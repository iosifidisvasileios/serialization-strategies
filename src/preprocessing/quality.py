from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

try:
    import pandas as pd
except Exception:  # pandas is optional for library use
    pd = None

from .alignment import tokens_for_annotation
from .ocr_utils import bbox_union
from .schema import Annotation, CanonicalDocument, OCRToken
from .value_normalization import (
    loose_value_match,
    normalize_extracted_value,
    normalize_for_matching,
    normalize_whitespace,
)


@dataclass(slots=True)
class AnnotationAlignmentCheck:
    doc_id: str
    dataset_name: str
    label: str
    start: int
    end: int
    annotation_text: str
    token_text: str
    n_tokens: int
    token_indices: list[int]
    token_available: bool
    strict_match: bool
    loose_value_match: bool
    annotation_norm: str
    token_norm: str
    status: str
    bbox: Optional[list[int]] = None
    page: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentQualityReport:
    doc_id: str
    dataset_name: str
    n_tokens: int
    n_annotations: int
    token_available_rate: float
    strict_match_rate: float
    loose_value_match_rate: float
    n_unavailable: int
    n_loose_mismatch: int
    checks: list[AnnotationAlignmentCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "dataset_name": self.dataset_name,
            "n_tokens": self.n_tokens,
            "n_annotations": self.n_annotations,
            "token_available_rate": self.token_available_rate,
            "strict_match_rate": self.strict_match_rate,
            "loose_value_match_rate": self.loose_value_match_rate,
            "n_unavailable": self.n_unavailable,
            "n_loose_mismatch": self.n_loose_mismatch,
            "checks": [c.to_dict() for c in self.checks],
        }


def reconstruct_token_text(tokens: list[OCRToken]) -> str:
    return " ".join(t.text for t in tokens).strip()


def annotation_alignment_check(
    doc: CanonicalDocument,
    annotation: Annotation,
    min_token_overlap: float = 0.5,
) -> AnnotationAlignmentCheck:
    indices = tokens_for_annotation(doc.tokens, annotation, min_token_overlap=min_token_overlap)
    matched_tokens = [doc.tokens[i] for i in indices]
    token_text = reconstruct_token_text(matched_tokens)

    strict = normalize_whitespace(token_text) == normalize_whitespace(annotation.text)
    loose = loose_value_match(annotation.label, annotation.text, token_text)
    token_available = len(indices) > 0

    if not token_available:
        status = "unavailable"
    elif loose:
        status = "ok"
    else:
        status = "loose_mismatch"

    bboxes = [t.bbox for t in matched_tokens]
    bbox = bbox_union(bboxes) if bboxes else None
    pages = {t.page for t in matched_tokens}
    page = next(iter(pages)) if len(pages) == 1 else None

    return AnnotationAlignmentCheck(
        doc_id=doc.doc_id,
        dataset_name=doc.dataset_name,
        label=annotation.label,
        start=annotation.start,
        end=annotation.end,
        annotation_text=annotation.text,
        token_text=token_text,
        n_tokens=len(indices),
        token_indices=indices,
        token_available=token_available,
        strict_match=strict,
        loose_value_match=loose,
        annotation_norm=normalize_for_matching(annotation.label, annotation.text),
        token_norm=normalize_for_matching(annotation.label, token_text),
        status=status,
        bbox=bbox,
        page=page,
    )


def assess_document_quality(doc: CanonicalDocument, min_token_overlap: float = 0.5) -> DocumentQualityReport:
    checks = [
        annotation_alignment_check(doc, ann, min_token_overlap=min_token_overlap)
        for ann in (doc.annotations or [])
    ]

    def mean_bool(values: Iterable[bool]) -> float:
        values = list(values)
        if not values:
            return 1.0
        return sum(bool(v) for v in values) / len(values)

    return DocumentQualityReport(
        doc_id=doc.doc_id,
        dataset_name=doc.dataset_name,
        n_tokens=len(doc.tokens),
        n_annotations=len(doc.annotations or []),
        token_available_rate=mean_bool(c.token_available for c in checks),
        strict_match_rate=mean_bool(c.strict_match for c in checks),
        loose_value_match_rate=mean_bool(c.loose_value_match for c in checks),
        n_unavailable=sum(1 for c in checks if not c.token_available),
        n_loose_mismatch=sum(1 for c in checks if c.token_available and not c.loose_value_match),
        checks=checks,
    )


def quality_report_to_dataframe(report: DocumentQualityReport):
    if pd is None:
        raise ImportError("pandas is required for quality_report_to_dataframe")
    return pd.DataFrame([c.to_dict() for c in report.checks])


def audit_documents(documents: Iterable[CanonicalDocument], min_token_overlap: float = 0.5):
    if pd is None:
        raise ImportError("pandas is required for audit_documents")

    rows: list[dict[str, Any]] = []
    for doc in documents:
        report = assess_document_quality(doc, min_token_overlap=min_token_overlap)
        rows.extend(c.to_dict() for c in report.checks)

    check_df = pd.DataFrame(rows)
    if check_df.empty:
        summary_df = pd.DataFrame(
            columns=["label", "n", "token_available_rate", "strict_match_rate", "loose_value_match_rate"]
        )
        return check_df, summary_df

    summary_df = (
        check_df.groupby("label")
        .agg(
            n=("label", "size"),
            token_available_rate=("token_available", "mean"),
            strict_match_rate=("strict_match", "mean"),
            loose_value_match_rate=("loose_value_match", "mean"),
        )
        .reset_index()
        .sort_values(["token_available_rate", "loose_value_match_rate", "label"])
    )
    return check_df, summary_df


def filter_annotations_for_training(
    doc: CanonicalDocument,
    min_token_overlap: float = 0.5,
    drop_unavailable: bool = True,
    drop_loose_mismatch: bool = False,
    attach_quality_report: bool = True,
) -> CanonicalDocument:
    """Return doc with annotations filtered by a consistent quality policy.

    Recommended FCC invoice policy:
        drop_unavailable=True
        drop_loose_mismatch=False
    """
    if doc.annotations is None:
        return doc

    kept: list[Annotation] = []
    for ann in doc.annotations:
        check = annotation_alignment_check(doc, ann, min_token_overlap=min_token_overlap)
        if drop_unavailable and not check.token_available:
            continue
        if drop_loose_mismatch and check.token_available and not check.loose_value_match:
            continue
        kept.append(ann)

    metadata = dict(doc.metadata)
    if attach_quality_report:
        report = assess_document_quality(doc, min_token_overlap=min_token_overlap)
        metadata["quality_report"] = report.to_dict()
        metadata["n_annotations_before_quality_filter"] = len(doc.annotations or [])
        metadata["n_annotations_after_quality_filter"] = len(kept)
        metadata["quality_filter_policy"] = {
            "min_token_overlap": min_token_overlap,
            "drop_unavailable": drop_unavailable,
            "drop_loose_mismatch": drop_loose_mismatch,
        }

    return doc.with_updates(annotations=kept, metadata=metadata)


def make_gold_span_records(
    doc: CanonicalDocument,
    min_token_overlap: float = 0.5,
    include_unavailable: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ann in doc.annotations or []:
        check = annotation_alignment_check(doc, ann, min_token_overlap=min_token_overlap)
        if not include_unavailable and not check.token_available:
            continue
        records.append(
            {
                "label": ann.label,
                "start": ann.start,
                "end": ann.end,
                "gold_text": ann.text,
                "matched_token_text": check.token_text,
                "normalized_gold": normalize_extracted_value(ann.label, ann.text),
                "normalized_token": normalize_extracted_value(ann.label, check.token_text),
                "token_indices": check.token_indices,
                "n_tokens": check.n_tokens,
                "token_available": check.token_available,
                "strict_match": check.strict_match,
                "loose_value_match": check.loose_value_match,
                "status": check.status,
                "bbox": check.bbox,
                "page": check.page,
            }
        )
    return records
