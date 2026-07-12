from __future__ import annotations

import unittest

from src.preprocessing.schema import CanonicalDocument, OCRToken
from src.serialization import (
    KeyValueAnchorPairsSerializer,
    KeyValueRowPairsSerializer,
    PrecedenceGraphOrderSerializer,
    TOKEN_CLASSIFICATION_SERIALIZERS,
)


def token(text: str, start: int, x0: int, y0: int, x1: int, y1: int) -> OCRToken:
    return OCRToken(
        text=text,
        start=start,
        end=start + len(text),
        page=0,
        bbox=[x0, y0, x1, y1],
        page_width=1000,
        page_height=1000,
    )


def document(tokens: list[OCRToken]) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id="doc",
        dataset_name="sample",
        text=" ".join(item.text for item in tokens),
        tokens=tokens,
        metadata={"token_labels": ["O"] * len(tokens)},
    )


def real_sources(record: dict) -> list[int]:
    return [source for source in record["source_token_indices"] if source is not None]


class NewStrategyTests(unittest.TestCase):
    def test_strategies_are_registered(self):
        for name in (
            "precedence_graph_order",
            "key_value_row_pairs",
            "key_value_anchor_pairs",
        ):
            self.assertIn(name, TOKEN_CLASSIFICATION_SERIALIZERS)

    def test_precedence_graph_repairs_geometric_order(self):
        doc = document(
            [
                token("second", 0, 10, 100, 70, 120),
                token("first", 10, 10, 10, 60, 30),
            ]
        )
        record = PrecedenceGraphOrderSerializer().serialize_train(doc)
        self.assertEqual(real_sources(record), [1, 0])
        self.assertCountEqual(real_sources(record), range(len(doc.tokens)))

    def test_row_gap_groups_key_and_value_without_losing_tokens(self):
        doc = document(
            [
                token("Invoice", 0, 10, 10, 80, 30),
                token("number", 8, 90, 10, 150, 30),
                token("12345", 15, 500, 10, 550, 30),
            ]
        )
        record = KeyValueRowPairsSerializer().serialize_train(doc)
        self.assertIn("field", record["layout_roles"])
        self.assertIn("key", record["layout_roles"])
        self.assertIn("value", record["layout_roles"])
        self.assertCountEqual(real_sources(record), range(len(doc.tokens)))

    def test_anchor_groups_right_value_without_losing_tokens(self):
        doc = document(
            [
                token("Name:", 0, 10, 10, 70, 30),
                token("Alice", 6, 100, 10, 150, 30),
            ]
        )
        record = KeyValueAnchorPairsSerializer().serialize_train(doc)
        self.assertIn("key", record["layout_roles"])
        self.assertIn("value", record["layout_roles"])
        self.assertEqual(real_sources(record), [0, 1])

    def test_anchor_can_group_aligned_value_below(self):
        doc = document(
            [
                token("Address:", 0, 10, 10, 90, 30),
                token("Main", 9, 12, 50, 55, 70),
                token("Street", 14, 60, 50, 110, 70),
            ]
        )
        record = KeyValueAnchorPairsSerializer().serialize_train(doc)
        orientations = [
            attrs.get("orientation")
            for role, attrs in zip(record["layout_roles"], record["item_attrs"])
            if role == "field"
        ]
        self.assertEqual(orientations, ["below"])
        self.assertEqual(real_sources(record), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
