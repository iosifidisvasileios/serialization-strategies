from __future__ import annotations
import ast, json, re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Optional
from .schema import Annotation, CanonicalDocument

DEFAULT_LABEL_ALIASES = {
    'Line Item - Start Data': 'Line Item - Start Date',
    'line item - start data': 'Line Item - Start Date',
}

def normalize_whitespace(value: str) -> str:
    return ' '.join(str(value).strip().split())

def label_to_key(label: str) -> str:
    label = normalize_whitespace(label).replace('&', ' and ')
    label = re.sub(r'[^0-9A-Za-z]+', '_', label)
    return re.sub(r'_+', '_', label).strip('_').lower()

def normalize_label_name(label: str, aliases: Optional[Mapping[str, str]] = None, keep_human_readable: bool = True) -> str:
    aliases = aliases or DEFAULT_LABEL_ALIASES
    clean = normalize_whitespace(label)
    clean = aliases.get(clean, clean)
    return clean if keep_human_readable else label_to_key(clean)

def key_to_bio(key: str, prefix: str) -> str:
    if prefix not in {'B','I'}: raise ValueError(f'Invalid BIO prefix: {prefix!r}')
    return f'{prefix}-{key}'

def normalize_bio_label(label: str) -> str:
    if label == 'O': return label
    prefix, name = str(label).split('-', 1)
    if prefix not in {'B','I'}: raise ValueError(f'Invalid BIO label: {label!r}')
    return f'{prefix}-{label_to_key(name)}'

def parse_json_like(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)): return value
    if isinstance(value, bytes): value = value.decode('utf-8')
    if not isinstance(value, str): return value
    s = value.strip()
    if not s: return None
    for fn in (json.loads, ast.literal_eval):
        try: return fn(s)
        except Exception: pass
    return value

def parse_annotations(value: Any, aliases: Optional[Mapping[str,str]] = None, labels_to_keep: Optional[Iterable[str]] = None, keep_human_readable: bool = True) -> list[Annotation]:
    parsed = parse_json_like(value)
    if parsed is None: return []
    if isinstance(parsed, dict):
        for key in ('labels','annotations','entities'):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]; break
    if not isinstance(parsed, list): raise ValueError(f'Expected labels list, got {type(parsed)!r}')
    keep = None
    if labels_to_keep is not None:
        keep = {normalize_label_name(x, aliases, keep_human_readable) for x in labels_to_keep}
    anns = []
    for item in parsed:
        if not isinstance(item, Mapping): continue
        if not {'label','start','end'}.issubset(item): continue
        raw = str(item['label'])
        label = normalize_label_name(raw, aliases, keep_human_readable)
        if keep is not None and label not in keep: continue
        anns.append(Annotation(label=label, raw_label=raw, start=int(item['start']), end=int(item['end']), text=str(item.get('text','')), extra={k:v for k,v in dict(item).items() if k not in {'label','start','end','text'}}))
    return sorted(anns, key=lambda a: (a.start, a.end, a.label))

def collect_labels(documents: Iterable[CanonicalDocument]) -> Counter[str]:
    c = Counter()
    for doc in documents:
        c.update([a.label for a in doc.annotations or []])
    return c

def document_frequency(documents: Iterable[CanonicalDocument]) -> Counter[str]:
    c = Counter()
    for doc in documents:
        c.update({a.label for a in doc.annotations or []})
    return c

def labels_per_document(documents: Iterable[CanonicalDocument]) -> dict[str, set[str]]:
    return {doc.doc_id: {a.label for a in doc.annotations or []} for doc in documents}

def common_labels(documents: Iterable[CanonicalDocument]) -> set[str]:
    sets = [{a.label for a in doc.annotations or []} for doc in documents if doc.annotations is not None]
    return set.intersection(*sets) if sets else set()

def select_labels_by_doc_frequency(documents: Iterable[CanonicalDocument], min_doc_frequency: float|int) -> set[str]:
    docs = list(documents)
    freq = document_frequency(docs)
    n_docs = len([d for d in docs if d.annotations is not None])
    if isinstance(min_doc_frequency, float):
        if not (0 < min_doc_frequency <= 1): raise ValueError('Float min_doc_frequency must be in (0,1].')
        threshold = int(n_docs * min_doc_frequency + 0.999999)
    else:
        threshold = int(min_doc_frequency)
    return {label for label, count in freq.items() if count >= threshold}

def build_bio_label_list(labels: Iterable[str], model_safe: bool = True) -> list[str]:
    names = sorted({label_to_key(x) if model_safe else normalize_whitespace(x) for x in labels})
    out = ['O']
    for name in names:
        out.extend([f'B-{name}', f'I-{name}'])
    return out

def build_label_maps(bio_labels: Iterable[str]) -> tuple[dict[str,int], dict[int,str]]:
    label2id = {lab:i for i,lab in enumerate(bio_labels)}
    return label2id, {i:lab for lab,i in label2id.items()}

def is_line_item_label(label: str) -> bool: return label_to_key(label).startswith('line_item_')
def strip_line_item_prefix(label: str) -> str:
    k = label_to_key(label)
    return k[len('line_item_'):] if k.startswith('line_item_') else k

def group_annotations_by_label(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    d = defaultdict(list)
    for ann in annotations: d[ann.label].append(ann)
    return dict(d)

def annotation_summary(documents: Iterable[CanonicalDocument]) -> dict[str, Any]:
    docs = list(documents)
    return {'n_documents': len([d for d in docs if d.annotations is not None]), 'occurrence_counts': dict(sorted(collect_labels(docs).items())), 'document_frequency': dict(sorted(document_frequency(docs).items())), 'common_labels': sorted(common_labels(docs))}
