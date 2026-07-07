from __future__ import annotations
from collections import defaultdict
from typing import Callable, Iterable, Optional
from .label_utils import label_to_key
from .ocr_utils import bbox_union
from .schema import AlignmentIssue, AlignmentResult, Annotation, CanonicalDocument, LabeledSpan, OCRToken

LabelEncoder = Callable[[str], str]

def char_overlap(token_start: int, token_end: int, span_start: int, span_end: int) -> int:
    return max(0, min(token_end, span_end) - max(token_start, span_start))

def token_overlap_ratio(token: OCRToken, annotation: Annotation) -> float:
    return char_overlap(token.start, token.end, annotation.start, annotation.end) / max(1, token.end-token.start)

def annotation_overlap_ratio(token: OCRToken, annotation: Annotation) -> float:
    return char_overlap(token.start, token.end, annotation.start, annotation.end) / max(1, annotation.end-annotation.start)

def tokens_for_annotation(tokens: list[OCRToken], annotation: Annotation, min_token_overlap: float = 0.5, allow_boundary_touch: bool = False) -> list[int]:
    matched=[]
    for i,t in enumerate(tokens):
        overlap = char_overlap(t.start,t.end,annotation.start,annotation.end)
        if overlap > 0 and overlap/max(1,t.end-t.start) >= min_token_overlap: matched.append(i)
        elif allow_boundary_touch and (t.end == annotation.start or t.start == annotation.end): matched.append(i)
    return matched

def default_label_encoder(label: str) -> str: return label_to_key(label)

def align_annotations_to_bio(tokens: list[OCRToken], annotations: Iterable[Annotation], labels_to_keep: Optional[set[str]] = None, min_token_overlap: float = 0.5, label_encoder: LabelEncoder = default_label_encoder, conflict_policy: str = 'keep_first') -> AlignmentResult:
    if conflict_policy not in {'keep_first','overwrite','error'}: raise ValueError('Invalid conflict_policy.')
    labels = ['O'] * len(tokens); issues=[]
    for ann in sorted(annotations, key=lambda a:(a.start,a.end,a.label)):
        if labels_to_keep is not None and ann.label not in labels_to_keep: continue
        idxs = tokens_for_annotation(tokens, ann, min_token_overlap)
        if not idxs:
            issues.append(AlignmentIssue('no_token_match', f'No OCR tokens matched {ann.label!r} span=({ann.start},{ann.end}) text={ann.text!r}', annotation=ann)); continue
        key = label_encoder(ann.label)
        for j,idx in enumerate(idxs):
            new = f"{'B' if j == 0 else 'I'}-{key}"
            if labels[idx] != 'O':
                issue = AlignmentIssue('label_conflict', f'Token already has {labels[idx]!r}; new label would be {new!r}.', annotation=ann, token_index=idx, token_text=tokens[idx].text)
                if conflict_policy == 'error': raise ValueError(issue.message)
                issues.append(issue)
                if conflict_policy == 'keep_first': continue
            labels[idx] = new
    return AlignmentResult(labels, issues)

def add_token_labels(doc: CanonicalDocument, labels_to_keep: Optional[set[str]] = None, min_token_overlap: float = 0.5, label_encoder: LabelEncoder = default_label_encoder, conflict_policy: str = 'keep_first', metadata_key: str = 'token_labels') -> CanonicalDocument:
    if doc.annotations is None: raise ValueError('doc.annotations is None; cannot create token labels.')
    res = align_annotations_to_bio(doc.tokens, doc.annotations, labels_to_keep, min_token_overlap, label_encoder, conflict_policy)
    md = dict(doc.metadata); md[metadata_key] = res.token_labels; md['alignment_issues'] = [x.to_dict() for x in res.issues]
    return doc.with_updates(metadata=md)

def bio_to_spans(tokens: list[OCRToken], bio_labels: list[str], source: str = 'gold', scores: Optional[list[float]] = None) -> list[LabeledSpan]:
    if len(tokens) != len(bio_labels): raise ValueError(f'tokens and labels length mismatch: {len(tokens)} vs {len(bio_labels)}')
    spans=[]; active_label=None; active_start=None
    def close(end_idx: int):
        nonlocal active_label, active_start
        if active_label is None or active_start is None or end_idx < active_start: return
        ts = tokens[active_start:end_idx+1]
        score = None if scores is None else sum(scores[active_start:end_idx+1])/max(1,len(ts))
        pages = {t.page for t in ts}
        spans.append(LabeledSpan(label=active_label, text=reconstruct_span_text(ts), start=ts[0].start, end=ts[-1].end, token_start=active_start, token_end=end_idx, page=next(iter(pages)) if len(pages)==1 else None, bbox=bbox_union([t.bbox for t in ts]), score=score, source=source))
        active_label=None; active_start=None
    for i,lab in enumerate(map(str,bio_labels)):
        if lab in {'O','-100'}:
            close(i-1); continue
        if '-' not in lab:
            close(i-1); active_label=lab; active_start=i; continue
        pref,name = lab.split('-',1)
        if pref == 'B': close(i-1); active_label=name; active_start=i
        elif pref == 'I':
            if active_label != name: close(i-1); active_label=name; active_start=i
        else: close(i-1)
    close(len(tokens)-1)
    return spans

def reconstruct_span_text(tokens: Iterable[OCRToken]) -> str:
    return ' '.join(t.text for t in tokens).strip()

def spans_to_field_dict(spans: Iterable[LabeledSpan], repeated_labels: Optional[set[str]] = None) -> dict[str, str|list[str]]:
    repeated_labels = repeated_labels or set(); grouped=defaultdict(list)
    for s in spans: grouped[s.label].append(s.text)
    return {lab: vals if lab in repeated_labels or len(vals)>1 else vals[0] for lab, vals in grouped.items()}

def group_line_item_spans_by_y(spans: Iterable[LabeledSpan], y_threshold: float = 12.0) -> list[list[LabeledSpan]]:
    items = [s for s in spans if s.label.startswith('line_item_') and s.bbox is not None and s.page is not None]
    def cy(s): return (s.bbox[1]+s.bbox[3])/2.0
    rows=[]
    for span in sorted(items, key=lambda s:(s.page,cy(s),s.bbox[0])):
        for row in rows:
            if row[0].page == span.page and abs(sum(cy(x) for x in row)/len(row) - cy(span)) <= y_threshold:
                row.append(span); row.sort(key=lambda s:s.bbox[0]); break
        else:
            rows.append([span])
    return rows
