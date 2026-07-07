from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Optional

from ..preprocessing.label_utils import label_to_key
from ..preprocessing.ocr_utils import token_row_col_buckets
from ..preprocessing.schema import CanonicalDocument
from ..preprocessing.value_normalization import normalize_extracted_value


class T5JsonSerializer:
    """Serialize OCR documents for T5-style OCR -> JSON extraction.

    Unlike token-classification serializers, this produces an input string and
    a normalized JSON target. OCR text is preserved in the input; target values
    are cleaned with value_normalization.py.
    """

    name = "t5_json"

    def __init__(
        self,
        labels_to_extract: Optional[list[str]] = None,
        include_dataset_name: bool = True,
        include_schema_prompt: bool = True,
        include_page_tokens: bool = True,
        include_layout_tokens: bool = False,
        max_tokens: Optional[int] = None,
        target_mode: str = "field_dict",
        sort_target_keys: bool = True,
    ) -> None:
        if target_mode not in {"field_dict", "line_item_records"}:
            raise ValueError("target_mode must be 'field_dict' or 'line_item_records'.")

        self.labels_to_extract = labels_to_extract
        self.include_dataset_name = include_dataset_name
        self.include_schema_prompt = include_schema_prompt
        self.include_page_tokens = include_page_tokens
        self.include_layout_tokens = include_layout_tokens
        self.max_tokens = max_tokens
        self.target_mode = target_mode
        self.sort_target_keys = sort_target_keys

    def serialize_train(self, doc: CanonicalDocument) -> dict[str, Any]:
        return {
            "id": doc.doc_id,
            "dataset": doc.dataset_name,
            "serializer": self.name,
            "input_text": self.make_input(doc),
            "target_text": json.dumps(
                self.make_target(doc),
                ensure_ascii=False,
                sort_keys=self.sort_target_keys,
            ),
        }

    def serialize_inference(self, doc: CanonicalDocument) -> dict[str, Any]:
        return {
            "id": doc.doc_id,
            "dataset": doc.dataset_name,
            "serializer": self.name,
            "input_text": self.make_input(doc),
        }

    def make_input(self, doc: CanonicalDocument) -> str:
        parts: list[str] = []

        if self.include_dataset_name:
            parts.append(f"Dataset: {doc.dataset_name}")

        labels = self.labels_to_extract
        if labels is None and doc.annotations:
            labels = sorted({ann.label for ann in doc.annotations})

        if self.include_schema_prompt and labels:
            parts.append("Extract the following fields as JSON:")
            parts.append(", ".join(labels))

        parts.append("Document:")
        parts.append(self.serialize_document_tokens(doc))

        return "\n".join(parts)

    def serialize_document_tokens(self, doc: CanonicalDocument) -> str:
        tokens = doc.tokens[: self.max_tokens] if self.max_tokens is not None else doc.tokens

        output: list[str] = []
        prev_page = None

        for token in tokens:
            if self.include_page_tokens and token.page != prev_page:
                output.append(f"[PAGE_{token.page}]")
                prev_page = token.page

            if self.include_layout_tokens:
                try:
                    row, col = token_row_col_buckets(token)
                    output.extend([f"[R_{row}]", f"[C_{col}]"])
                except Exception:
                    output.extend(["[R_UNK]", "[C_UNK]"])

            output.append(token.text)

        return " ".join(output)

    def make_target(self, doc: CanonicalDocument) -> dict[str, Any]:
        if self.target_mode == "line_item_records":
            return self.make_line_item_record_target(doc)
        return self.make_field_dict_target(doc)

    def make_field_dict_target(self, doc: CanonicalDocument) -> dict[str, Any]:
        grouped: dict[str, list[str]] = defaultdict(list)
        keep = set(self.labels_to_extract) if self.labels_to_extract is not None else None

        for ann in doc.annotations or []:
            if keep is not None and ann.label not in keep:
                continue

            key = label_to_key(ann.label)
            grouped[key].append(normalize_extracted_value(ann.label, ann.text))

        result: dict[str, Any] = {}

        for key, values in sorted(grouped.items()):
            result[key] = values[0] if len(values) == 1 else values

        return result

    def make_line_item_record_target(self, doc: CanonicalDocument) -> dict[str, Any]:
        """Build scalar fields plus line_items grouped by occurrence order.

        This is a simple sequence-order grouping. Later, for invoices, you may
        replace it with geometry-aware row grouping if you want row-level line
        item metrics.
        """
        scalar: dict[str, list[str]] = defaultdict(list)
        line_fields: dict[str, list[str]] = defaultdict(list)
        keep = set(self.labels_to_extract) if self.labels_to_extract is not None else None

        for ann in doc.annotations or []:
            if keep is not None and ann.label not in keep:
                continue

            key = label_to_key(ann.label)
            value = normalize_extracted_value(ann.label, ann.text)

            if key.startswith("line_item_"):
                field = key[len("line_item_") :]
                line_fields[field].append(value)
            else:
                scalar[key].append(value)

        result: dict[str, Any] = {}

        for key, values in sorted(scalar.items()):
            result[key] = values[0] if len(values) == 1 else values

        if line_fields:
            n_items = max(len(values) for values in line_fields.values())
            items: list[dict[str, str]] = []

            for i in range(n_items):
                item: dict[str, str] = {}
                for field, values in sorted(line_fields.items()):
                    if i < len(values):
                        item[field] = values[i]
                items.append(item)

            result["line_items"] = items

        return result
