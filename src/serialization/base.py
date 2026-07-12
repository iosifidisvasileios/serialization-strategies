from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ..preprocessing.schema import CanonicalDocument, OCRToken


IGNORE_LABEL = -100


def _bio_entity_memberships(
    token_labels: Sequence[str],
) -> list[Optional[tuple[str, int]]]:
    """Map canonical BIO labels to stable entity-instance identifiers.

    Invalid/orphan ``I`` labels start a new entity instance.  Keeping the
    instance id separate from the entity type lets serializers reorder tokens
    without accidentally joining two adjacent entities of the same type.
    """
    memberships: list[Optional[tuple[str, int]]] = []
    active_entity: Optional[str] = None
    active_instance = -1

    for raw_label in token_labels:
        label = str(raw_label)
        if label == "O" or label == str(IGNORE_LABEL) or "-" not in label:
            memberships.append(None)
            active_entity = None
            continue

        prefix, entity = label.split("-", 1)
        if not entity:
            memberships.append(None)
            active_entity = None
            continue

        if prefix != "I" or active_entity != entity:
            active_instance += 1
        active_entity = entity
        memberships.append((entity, active_instance))

    return memberships


def rebuild_bio_labels_for_serialized_order(
    token_labels: Sequence[str],
    source_token_indices: Sequence[Optional[int]],
    *,
    ignore_label: int = IGNORE_LABEL,
) -> list[str | int]:
    """Re-encode canonical BIO labels in the serializer's emitted order.

    Synthetic layout items are transparent and retain ``ignore_label``.  A
    reordered or separated entity run always starts with ``B``; adjacent real
    tokens from the same canonical entity instance continue with ``I``.
    """
    memberships = _bio_entity_memberships(token_labels)
    output: list[str | int] = []
    previous_membership: Optional[tuple[str, int]] = None

    for source_index in source_token_indices:
        if source_index is None:
            output.append(ignore_label)
            continue
        if source_index < 0 or source_index >= len(memberships):
            raise IndexError(
                f"source_token_index {source_index} is outside 0..{len(memberships) - 1}."
            )

        membership = memberships[source_index]
        if membership is None:
            output.append("O")
            previous_membership = None
            continue

        entity, _ = membership
        prefix = "I" if membership == previous_membership else "B"
        output.append(f"{prefix}-{entity}")
        previous_membership = membership

    return output


@dataclass(slots=True)
class SerializedItem:
    """One emitted token/item in a serialized token-classification sequence.

    source_token_index is None for synthetic layout tokens such as [PAGE_0].
    Real OCR tokens carry their original index so predictions can be mapped
    back to the canonical token stream.
    """

    text: str
    source_token_index: Optional[int]
    is_layout_token: bool = False
    layout_role: Optional[str] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseSerializer(ABC):
    """Base class for token-level serialization strategies.

    Output schema for token classification:
        {
          "id": str,
          "dataset": str,
          "serializer": str,
          "tokens": list[str],
          "labels": list[str | int],          # training only
          "loss_mask": list[bool],            # True only for real OCR tokens
          "source_token_indices": list[int|None],
          "pages": list[int|None],
          "bboxes": list[list[int]|None],
          "normalized_bboxes": list[list[int]|None],
          "offsets": list[list[int]|None],
          "layout_roles": list[str|None],
          "metadata": dict
        }

    Synthetic layout tokens receive labels=-100 and loss_mask=False.
    """

    name = "base"

    def __init__(
        self,
        token_label_key: str = "token_labels",
        ignore_label: int = IGNORE_LABEL,
        include_bboxes: bool = True,
        include_normalized_bboxes: bool = True,
        include_offsets: bool = True,
        include_metadata: bool = True,
    ) -> None:
        self.token_label_key = token_label_key
        self.ignore_label = ignore_label
        self.include_bboxes = include_bboxes
        self.include_normalized_bboxes = include_normalized_bboxes
        self.include_offsets = include_offsets
        self.include_metadata = include_metadata

    def serialize_train(self, doc: CanonicalDocument) -> dict[str, Any]:
        token_labels = self._get_token_labels(doc)
        items = self.make_items(doc)
        return self._record_from_items(doc, items, token_labels=token_labels)

    def serialize_inference(self, doc: CanonicalDocument) -> dict[str, Any]:
        items = self.make_items(doc)
        return self._record_from_items(doc, items, token_labels=None)

    @abstractmethod
    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        """Return serialized items for a canonical document."""

    def _record_from_items(
        self,
        doc: CanonicalDocument,
        items: Sequence[SerializedItem],
        token_labels: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        tokens = [item.text for item in items]
        source_indices = [item.source_token_index for item in items]
        loss_mask = [idx is not None for idx in source_indices]
        layout_roles = [item.layout_role for item in items]
        item_attrs = [item.attrs for item in items]

        record: dict[str, Any] = {
            "id": doc.doc_id,
            "dataset": doc.dataset_name,
            "serializer": self.name,
            "tokens": tokens,
            "source_token_indices": source_indices,
            "loss_mask": loss_mask,
            "layout_roles": layout_roles,
            "item_attrs": item_attrs,
            "original_token_count": len(doc.tokens),
            "serialized_token_count": len(items),
        }

        if token_labels is not None:
            if len(token_labels) != len(doc.tokens):
                raise ValueError(
                    f"Expected {len(doc.tokens)} BIO labels for document {doc.doc_id!r}; "
                    f"got {len(token_labels)}. Run add_token_labels/filtering before serialization."
                )

            record["original_token_labels"] = list(token_labels)
            record["labels"] = rebuild_bio_labels_for_serialized_order(
                token_labels,
                source_indices,
                ignore_label=self.ignore_label,
            )

        if self.include_offsets:
            record["offsets"] = [
                self._token_offsets(doc.tokens[idx]) if idx is not None else None
                for idx in source_indices
            ]

        record["pages"] = [
            doc.tokens[idx].page if idx is not None else item.attrs.get("page")
            for item, idx in zip(items, source_indices)
        ]

        if self.include_bboxes:
            record["bboxes"] = [
                doc.tokens[idx].bbox if idx is not None else None
                for idx in source_indices
            ]

        if self.include_normalized_bboxes:
            record["normalized_bboxes"] = [
                self._safe_normalized_bbox(doc.tokens[idx]) if idx is not None else None
                for idx in source_indices
            ]

        if self.include_metadata:
            record["metadata"] = self._safe_metadata(doc)

        return record

    def _get_token_labels(self, doc: CanonicalDocument) -> list[str]:
        labels = doc.metadata.get(self.token_label_key)
        if labels is None:
            raise ValueError(
                f"Document {doc.doc_id!r} has no metadata[{self.token_label_key!r}]. "
                "Run add_token_labels(...) before serialize_train(...)."
            )
        labels = list(labels)
        if len(labels) != len(doc.tokens):
            raise ValueError(
                f"Document {doc.doc_id!r}: {len(labels)} labels for {len(doc.tokens)} tokens."
            )
        return labels

    @staticmethod
    def _token_offsets(token: OCRToken) -> list[int]:
        return [int(token.start), int(token.end)]

    @staticmethod
    def _safe_normalized_bbox(token: OCRToken) -> Optional[list[int]]:
        try:
            return token.normalized_bbox()
        except Exception:
            return None

    @staticmethod
    def _safe_metadata(doc: CanonicalDocument) -> dict[str, Any]:
        # Avoid writing large OCR payloads into every processed record.
        return {k: v for k, v in doc.metadata.items() if k != "raw_ocr"}

    @staticmethod
    def layout_item(
        text: str,
        role: str,
        page: Optional[int] = None,
        **attrs: Any,
    ) -> SerializedItem:
        if page is not None:
            attrs = {**attrs, "page": page}
        return SerializedItem(
            text=text,
            source_token_index=None,
            is_layout_token=True,
            layout_role=role,
            attrs=attrs,
        )

    @staticmethod
    def token_item(
        token_index: int,
        token_text: str,
        **attrs: Any,
    ) -> SerializedItem:
        return SerializedItem(
            text=token_text,
            source_token_index=token_index,
            is_layout_token=False,
            layout_role="ocr_token",
            attrs=attrs,
        )


def real_token_items(doc: CanonicalDocument) -> list[SerializedItem]:
    return [
        BaseSerializer.token_item(i, token.text)
        for i, token in enumerate(doc.tokens)
    ]


def collapse_serialized_labels_to_original(
    record: dict[str, Any],
    serialized_labels: Sequence[str],
    original_token_count: Optional[int] = None,
    conflict_policy: str = "first_non_o",
) -> list[str]:
    """Map serialized predictions back to the original OCR token sequence.

    This is used when a serializer emits synthetic layout tokens. Predictions
    for layout tokens are ignored because source_token_index is None.

    conflict_policy:
        - first_non_o: keep first non-O prediction per original token.
        - last: use last prediction per original token.
        - error: raise if conflicting non-O predictions occur.
    """
    if conflict_policy not in {"first_non_o", "last", "error"}:
        raise ValueError("conflict_policy must be: first_non_o, last, or error")

    source_indices = record.get("source_token_indices")
    if source_indices is None:
        raise KeyError("record must contain source_token_indices")

    if len(source_indices) != len(serialized_labels):
        raise ValueError(
            f"source_token_indices and serialized_labels length mismatch: "
            f"{len(source_indices)} vs {len(serialized_labels)}"
        )

    n = original_token_count or record.get("original_token_count")
    if n is None:
        real_indices = [i for i in source_indices if i is not None]
        n = max(real_indices) + 1 if real_indices else 0

    output = ["O"] * int(n)

    for source_idx, label in zip(source_indices, serialized_labels):
        if source_idx is None:
            continue

        label = str(label)
        if conflict_policy == "last":
            output[source_idx] = label
            continue

        if conflict_policy == "first_non_o":
            if output[source_idx] == "O" and label != "O":
                output[source_idx] = label
            elif output[source_idx] == "O" and label == "O":
                output[source_idx] = label
            continue

        if conflict_policy == "error":
            if output[source_idx] != "O" and label != "O" and output[source_idx] != label:
                raise ValueError(
                    f"Conflicting predictions for token {source_idx}: "
                    f"{output[source_idx]!r} vs {label!r}"
                )
            if label != "O":
                output[source_idx] = label

    return output


def iter_real_serialized_positions(record: dict[str, Any]) -> Iterable[tuple[int, int]]:
    """Yield (serialized_position, original_token_index) for real OCR tokens."""
    for pos, idx in enumerate(record.get("source_token_indices", [])):
        if idx is not None:
            yield pos, int(idx)
