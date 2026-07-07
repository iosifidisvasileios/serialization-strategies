from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Optional

from .label_utils import label_to_key
from .ocr_utils import bbox_union
from .schema import (
    AlignmentIssue,
    AlignmentResult,
    Annotation,
    CanonicalDocument,
    LabeledSpan,
    OCRToken,
)
from .value_normalization import (
    loose_value_match,
    normalize_extracted_value,
    normalize_for_matching,
    normalize_whitespace,
)


LabelEncoder = Callable[[str], str]


def char_overlap(token_start: int, token_end: int, span_start: int, span_end: int) -> int:
    start = max(token_start, span_start)
    end = min(token_end, span_end)
    return max(0, end - start)


def token_overlap_ratio(token: OCRToken, annotation: Annotation) -> float:
    overlap = char_overlap(token.start, token.end, annotation.start, annotation.end)
    return overlap / max(1, token.end - token.start)


def annotation_overlap_ratio(token: OCRToken, annotation: Annotation) -> float:
    overlap = char_overlap(token.start, token.end, annotation.start, annotation.end)
    return overlap / max(1, annotation.end - annotation.start)


def tokens_for_annotation(
    tokens: list[OCRToken],
    annotation: Annotation,
    min_token_overlap: float = 0.5,
    allow_boundary_touch: bool = False,
) -> list[int]:
    matched: list[int] = []

    for i, token in enumerate(tokens):
        overlap = char_overlap(token.start, token.end, annotation.start, annotation.end)
        if overlap > 0:
            ratio = overlap / max(1, token.end - token.start)
            if ratio >= min_token_overlap:
                matched.append(i)
            continue

        if allow_boundary_touch and (token.end == annotation.start or token.start == annotation.end):
            matched.append(i)

    return matched


def default_label_encoder(label: str) -> str:
    return label_to_key(label)


def reconstruct_span_text(tokens: Iterable[OCRToken]) -> str:
    return " ".join(t.text for t in tokens).strip()


def annotation_token_diagnostic(
    tokens: list[OCRToken],
    annotation: Annotation,
    min_token_overlap: float = 0.5,
) -> dict:
    indices = tokens_for_annotation(tokens, annotation, min_token_overlap=min_token_overlap)
    matched_tokens = [tokens[i] for i in indices]
    token_text = reconstruct_span_text(matched_tokens)

    return {
        "label": annotation.label,
        "start": annotation.start,
        "end": annotation.end,
        "annotation_text": annotation.text,
        "token_text": token_text,
        "n_tokens": len(indices),
        "token_indices": indices,
        "token_available": len(indices) > 0,
        "strict_match": normalize_whitespace(token_text) == normalize_whitespace(annotation.text),
        "loose_value_match": loose_value_match(annotation.label, annotation.text, token_text),
        "annotation_norm": normalize_for_matching(annotation.label, annotation.text),
        "token_norm": normalize_for_matching(annotation.label, token_text),
    }


def align_annotations_to_bio(
    tokens: list[OCRToken],
    annotations: Iterable[Annotation],
    labels_to_keep: Optional[set[str]] = None,
    min_token_overlap: float = 0.5,
    label_encoder: LabelEncoder = default_label_encoder,
    conflict_policy: str = "keep_first",
    drop_unmatched_annotations: bool = False,
) -> AlignmentResult:
    """Convert character-span annotations into BIO token labels.

    If an OCR token contains extra characters, e.g. "$150.00P-2" while the
    gold annotation is "150.00", this still labels the whole OCR token when
    the character offsets overlap. Value cleanup belongs in normalization.
    """
    if conflict_policy not in {"keep_first", "overwrite", "error"}:
        raise ValueError("conflict_policy must be one of: keep_first, overwrite, error")

    token_labels = ["O"] * len(tokens)
    issues: list[AlignmentIssue] = []

    sorted_annotations = sorted(annotations, key=lambda a: (a.start, a.end, a.label))

    for ann in sorted_annotations:
        if labels_to_keep is not None and ann.label not in labels_to_keep:
            continue

        token_indices = tokens_for_annotation(tokens, ann, min_token_overlap=min_token_overlap)
        if not token_indices:
            issues.append(
                AlignmentIssue(
                    issue_type="no_token_match",
                    message=(
                        f"No OCR tokens matched annotation {ann.label!r} "
                        f"span=({ann.start}, {ann.end}) text={ann.text!r}"
                    ),
                    annotation=ann,
                )
            )
            if drop_unmatched_annotations:
                continue
            continue

        encoded = label_encoder(ann.label)
        for j, idx in enumerate(token_indices):
            prefix = "B" if j == 0 else "I"
            new_label = f"{prefix}-{encoded}"

            if token_labels[idx] != "O":
                issue = AlignmentIssue(
                    issue_type="label_conflict",
                    message=(
                        f"Token already has label {token_labels[idx]!r}; "
                        f"new label would be {new_label!r}."
                    ),
                    annotation=ann,
                    token_index=idx,
                    token_text=tokens[idx].text,
                )
                if conflict_policy == "error":
                    raise ValueError(issue.message)
                issues.append(issue)
                if conflict_policy == "keep_first":
                    continue

            token_labels[idx] = new_label

    return AlignmentResult(token_labels=token_labels, issues=issues)


def add_token_labels(
    doc: CanonicalDocument,
    labels_to_keep: Optional[set[str]] = None,
    min_token_overlap: float = 0.5,
    label_encoder: LabelEncoder = default_label_encoder,
    conflict_policy: str = "keep_first",
    metadata_key: str = "token_labels",
    include_alignment_diagnostics: bool = True,
) -> CanonicalDocument:
    if doc.annotations is None:
        raise ValueError("Cannot add token labels: doc.annotations is None.")

    result = align_annotations_to_bio(
        tokens=doc.tokens,
        annotations=doc.annotations,
        labels_to_keep=labels_to_keep,
        min_token_overlap=min_token_overlap,
        label_encoder=label_encoder,
        conflict_policy=conflict_policy,
    )

    metadata = dict(doc.metadata)
    metadata[metadata_key] = result.token_labels
    metadata["alignment_issues"] = [issue.to_dict() for issue in result.issues]

    if include_alignment_diagnostics:
        metadata["alignment_diagnostics"] = [
            annotation_token_diagnostic(doc.tokens, ann, min_token_overlap=min_token_overlap)
            for ann in doc.annotations
            if labels_to_keep is None or ann.label in labels_to_keep
        ]

    return doc.with_updates(metadata=metadata)


def bio_to_spans(
    tokens: list[OCRToken],
    bio_labels: list[str],
    source: str = "gold",
    scores: Optional[list[float]] = None,
    normalize_values: bool = False,
) -> list[LabeledSpan]:
    if len(tokens) != len(bio_labels):
        raise ValueError(f"tokens and bio_labels must have same length: {len(tokens)} vs {len(bio_labels)}")

    spans: list[LabeledSpan] = []
    active_label: Optional[str] = None
    active_start: Optional[int] = None

    def close_span(end_idx: int) -> None:
        nonlocal active_label, active_start
        if active_label is None or active_start is None:
            return
        span_tokens = tokens[active_start : end_idx + 1]
        text = reconstruct_span_text(span_tokens)
        if normalize_values:
            text = normalize_extracted_value(active_label, text)
        bbox = bbox_union([t.bbox for t in span_tokens])
        score = None
        if scores is not None:
            span_scores = scores[active_start : end_idx + 1]
            score = sum(span_scores) / max(1, len(span_scores))
        pages = {t.page for t in span_tokens}
        page = next(iter(pages)) if len(pages) == 1 else None
        spans.append(
            LabeledSpan(
                label=active_label,
                text=text,
                start=span_tokens[0].start,
                end=span_tokens[-1].end,
                token_start=active_start,
                token_end=end_idx,
                page=page,
                bbox=bbox,
                score=score,
                source=source,
            )
        )
        active_label = None
        active_start = None

    for i, raw_label in enumerate(bio_labels):
        label = str(raw_label)
        if label == "O" or label == "-100":
            close_span(i - 1)
            continue
        if "-" not in label:
            close_span(i - 1)
            active_label = label
            active_start = i
            continue
        prefix, name = label.split("-", 1)
        if prefix == "B":
            close_span(i - 1)
            active_label = name
            active_start = i
        elif prefix == "I":
            if active_label == name:
                continue
            close_span(i - 1)
            active_label = name
            active_start = i
        else:
            close_span(i - 1)
    close_span(len(tokens) - 1)
    return spans


def spans_to_field_dict(
    spans: Iterable[LabeledSpan],
    repeated_labels: Optional[set[str]] = None,
    normalize_values: bool = False,
) -> dict[str, str | list[str]]:
    repeated_labels = repeated_labels or set()
    grouped: dict[str, list[str]] = defaultdict(list)
    for span in spans:
        value = normalize_extracted_value(span.label, span.text) if normalize_values else span.text
        grouped[span.label].append(value)

    output: dict[str, str | list[str]] = {}
    for label, values in grouped.items():
        if label in repeated_labels or len(values) > 1:
            output[label] = values
        else:
            output[label] = values[0]
    return output


def group_line_item_spans_by_y(spans: Iterable[LabeledSpan], y_threshold: float = 12.0) -> list[list[LabeledSpan]]:
    line_item_spans = [
        s for s in spans
        if s.label.startswith("line_item_") and s.bbox is not None and s.page is not None
    ]

    def center_y(span: LabeledSpan) -> float:
        assert span.bbox is not None
        return (span.bbox[1] + span.bbox[3]) / 2.0

    line_item_spans.sort(key=lambda s: (s.page if s.page is not None else -1, center_y(s), s.bbox[0]))
    rows: list[list[LabeledSpan]] = []
    for span in line_item_spans:
        placed = False
        for row in rows:
            row_page = row[0].page
            row_y = sum(center_y(s) for s in row) / len(row)
            if row_page == span.page and abs(row_y - center_y(span)) <= y_threshold:
                row.append(span)
                row.sort(key=lambda s: s.bbox[0] if s.bbox is not None else -1)
                placed = True
                break
        if not placed:
            rows.append([span])
    return rows


def normalized_gold_field_dict(doc: CanonicalDocument, repeated_labels: Optional[set[str]] = None) -> dict[str, str | list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for ann in doc.annotations or []:
        key = label_to_key(ann.label)
        grouped[key].append(normalize_extracted_value(ann.label, ann.text))

    output: dict[str, str | list[str]] = {}
    repeated_labels = repeated_labels or set()
    for label, values in grouped.items():
        if label in repeated_labels or len(values) > 1:
            output[label] = values
        else:
            output[label] = values[0]
    return output
