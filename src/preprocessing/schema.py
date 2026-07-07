from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

@dataclass(slots=True)
class OCRToken:
    text: str
    start: int
    end: int
    page: int
    bbox: list[int]  # [left, top, right, bottom], raw pixels
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    block_start: Optional[int] = None
    block_end: Optional[int] = None
    block_id: Optional[int] = None
    style: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def x0(self) -> int: return self.bbox[0]
    @property
    def y0(self) -> int: return self.bbox[1]
    @property
    def x1(self) -> int: return self.bbox[2]
    @property
    def y1(self) -> int: return self.bbox[3]
    @property
    def width(self) -> int: return max(0, self.x1 - self.x0)
    @property
    def height(self) -> int: return max(0, self.y1 - self.y0)
    @property
    def center_x(self) -> float: return (self.x0 + self.x1) / 2.0
    @property
    def center_y(self) -> float: return (self.y0 + self.y1) / 2.0

    def normalized_bbox(self, scale: int = 1000, clamp: bool = True) -> list[int]:
        if self.page_width is None or self.page_height is None:
            raise ValueError(f"Missing page size for token {self.text!r}; cannot normalize bbox.")
        vals = [
            round(scale * self.x0 / self.page_width),
            round(scale * self.y0 / self.page_height),
            round(scale * self.x1 / self.page_width),
            round(scale * self.y1 / self.page_height),
        ]
        if clamp:
            vals = [max(0, min(scale, int(v))) for v in vals]
        return [int(v) for v in vals]

    def with_updates(self, **kwargs: Any) -> 'OCRToken':
        return replace(self, **kwargs)
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class OCRBlock:
    text: str
    start: int
    end: int
    page: int
    bbox: list[int]
    block_id: int
    block_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)
    @property
    def center_y(self) -> float: return (self.bbox[1] + self.bbox[3]) / 2.0
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(slots=True)
class Annotation:
    label: str
    start: int
    end: int
    text: str
    raw_label: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    @property
    def length(self) -> int: return max(0, self.end - self.start)
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(slots=True)
class LabeledSpan:
    label: str
    text: str
    start: int
    end: int
    token_start: int
    token_end: int  # inclusive
    page: Optional[int] = None
    bbox: Optional[list[int]] = None
    score: Optional[float] = None
    source: str = 'gold'
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(slots=True)
class AlignmentIssue:
    issue_type: str
    message: str
    annotation: Optional[Annotation] = None
    token_index: Optional[int] = None
    token_text: Optional[str] = None
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.annotation is not None:
            d['annotation'] = self.annotation.to_dict()
        return d

@dataclass(slots=True)
class AlignmentResult:
    token_labels: list[str]
    issues: list[AlignmentIssue] = field(default_factory=list)
    @property
    def ok(self) -> bool: return len(self.issues) == 0
    def to_dict(self) -> dict[str, Any]:
        return {'token_labels': self.token_labels, 'issues': [x.to_dict() for x in self.issues], 'ok': self.ok}

@dataclass(slots=True)
class CanonicalDocument:
    doc_id: str
    dataset_name: str
    text: str
    tokens: list[OCRToken]
    blocks: list[OCRBlock] = field(default_factory=list)
    annotations: Optional[list[Annotation]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def with_updates(self, **kwargs: Any) -> 'CanonicalDocument': return replace(self, **kwargs)
    def to_dict(self) -> dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'dataset_name': self.dataset_name,
            'text': self.text,
            'tokens': [t.to_dict() for t in self.tokens],
            'blocks': [b.to_dict() for b in self.blocks],
            'annotations': None if self.annotations is None else [a.to_dict() for a in self.annotations],
            'metadata': self.metadata,
        }

def empty_document(doc_id: str, dataset_name: str) -> CanonicalDocument:
    return CanonicalDocument(doc_id=doc_id, dataset_name=dataset_name, text='', tokens=[], blocks=[], annotations=None, metadata={})
