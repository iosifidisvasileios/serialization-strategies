from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "training"))

from src.training.data_pipeline import DataPipeline  # noqa: E402
from src.training.experiment_config import ExperimentConfig, ModelSpec  # noqa: E402
from src.training.layout_model import (  # noqa: E402
    ALL_LAYOUT_TOKENS,
    NumericLayoutTokenClassifier,
    canonical_layout_token,
)
from src.training.ocr_metrics import (  # noqa: E402
    aggregate_ocr_predictions,
    entity_metrics,
    rebuild_bio_for_source_order,
    repair_bio_label_ids,
)
from src.serialization.base import rebuild_bio_labels_for_serialized_order  # noqa: E402


class TinyEncoding(dict):
    def __init__(self, words):
        super().__init__(input_ids=list(range(1, len(words) + 1)))
        self._word_ids = list(range(len(words)))

    def word_ids(self, batch_index=None):
        del batch_index
        return list(self._word_ids)


class TinyTokenizer:
    name_or_path = "tiny"

    def __len__(self):
        return 100 + len(ALL_LAYOUT_TOKENS)

    @staticmethod
    def num_special_tokens_to_add(pair=False):
        del pair
        return 2

    def __call__(self, words, **kwargs):
        del kwargs
        return TinyEncoding(words)


class FakeDataset:
    def __init__(self, columns):
        self.columns = columns
        self.column_names = list(columns)

    def __len__(self):
        return len(self.columns["metric_doc_key"])

    def __getitem__(self, key):
        return self.columns[key]


class LayoutFixTests(unittest.TestCase):
    def test_reordered_entity_gets_valid_bio(self):
        canonical = ["B-X", "I-X"]
        sources = [1, None, 0]
        expected = ["B-X", -100, "I-X"]
        self.assertEqual(
            rebuild_bio_labels_for_serialized_order(canonical, sources), expected
        )
        self.assertEqual(
            rebuild_bio_for_source_order(canonical, sources, ignore_label=-100), expected
        )

    def test_separated_same_type_entities_do_not_join(self):
        canonical = ["B-X", "I-X", "O", "B-X"]
        sources = [0, 3, 1]
        self.assertEqual(
            rebuild_bio_for_source_order(canonical, sources, ignore_label=-100),
            ["B-X", "B-X", "B-X"],
        )

    def test_window_repair_and_ignored_subtokens(self):
        id2label = {0: "O", 1: "B-X", 2: "I-X"}
        label2id = {label: idx for idx, label in id2label.items()}
        self.assertEqual(
            repair_bio_label_ids(
                [2, -100, 2, 0, 2],
                id2label=id2label,
                label2id=label2id,
                ignore_label=-100,
            ),
            [1, -100, 2, 0, 1],
        )

    def test_atomic_vocabulary_is_bounded_by_roles(self):
        self.assertEqual(canonical_layout_token("compact_bbox"), "[LAYOUT_BBOX]")
        self.assertEqual(canonical_layout_token("coord_suffix"), "[LAYOUT_COORD]")
        self.assertEqual(canonical_layout_token("unseen_role"), "[LAYOUT_UNKNOWN]")
        self.assertLess(len(ALL_LAYOUT_TOKENS), 32)

    def test_shared_windows_equalize_plain_and_heavy_strategies(self):
        cfg = ExperimentConfig(
            project_root=ROOT,
            max_length=8,
            word_window_size=0,
            n_folds=2,
            enforce_minimum_versions=False,
        )
        pipeline = DataPipeline(cfg)
        pipeline.state.label_list = ["O", "B-X", "I-X"]
        pipeline.state.label2id = {label: idx for idx, label in enumerate(pipeline.state.label_list)}
        pipeline.state.id2label = {idx: label for label, idx in pipeline.state.label2id.items()}
        plain = {
            "id": "doc",
            "tokens": ["a", "b", "c", "d"],
            "labels": ["B-X", "I-X", "O", "O"],
            "source_token_indices": [0, 1, 2, 3],
            "layout_roles": ["ocr_token"] * 4,
            "normalized_bboxes": [[0, 0, 1, 1]] * 4,
            "pages": [0] * 4,
            "original_token_count": 4,
        }
        heavy_tokens = []
        heavy_labels = []
        heavy_sources = []
        heavy_roles = []
        heavy_boxes = []
        heavy_pages = []
        for source, text in enumerate(["a", "b", "c", "d"]):
            heavy_tokens.extend([f"[R_{source}]", f"[C_{source}]", text])
            heavy_labels.extend([-100, -100, plain["labels"][source]])
            heavy_sources.extend([None, None, source])
            heavy_roles.extend(["row_bucket", "col_bucket", "ocr_token"])
            heavy_boxes.extend([None, None, [0, 0, 1, 1]])
            heavy_pages.extend([0, 0, 0])
        heavy = {
            "id": "doc",
            "tokens": heavy_tokens,
            "labels": heavy_labels,
            "source_token_indices": heavy_sources,
            "layout_roles": heavy_roles,
            "normalized_bboxes": heavy_boxes,
            "pages": heavy_pages,
            "original_token_count": 4,
        }
        pipeline.state.records_by_dataset_strategy = {
            "sample": {"plain_text": [plain], "rowcol_bucket": [heavy]}
        }
        examples = pipeline._ensure_fair_examples(
            "sample", ModelSpec("tiny", "tiny"), TinyTokenizer()
        )
        self.assertEqual(len(examples["plain_text"]), len(examples["rowcol_bucket"]))
        self.assertEqual(
            [(row["word_start"], row["word_end"]) for row in examples["plain_text"]],
            [(row["word_start"], row["word_end"]) for row in examples["rowcol_bucket"]],
        )

    def test_source_window_lookup_handles_reordered_tokens(self):
        views = [
            {"source": 2, "active": {}, "prefix": [], "real": 0, "suffix": []},
            {"source": 0, "active": {}, "prefix": [], "real": 1, "suffix": []},
            {"source": 1, "active": {}, "prefix": [], "real": 2, "suffix": []},
        ]
        lookup = DataPipeline._source_view_lookup(views)
        self.assertEqual(DataPipeline._render_source_window(lookup, {0}), [1])

    def test_ocr_aggregation_deduplicates_and_rejoins_entities(self):
        columns = {
            "metric_source_indices": [[0, -1, -1], [1, -1, -1], [0, -1, -1]],
            "metric_gold_labels": [[1, -100, -100], [2, -100, -100], [1, -100, -100]],
            "metric_doc_key": ["doc", "doc", "doc"],
            "metric_record_index": [0, 0, 0],
            "metric_original_token_count": [2, 2, 2],
        }
        predictions = np.asarray([[1, 0, 0], [2, 0, 0], [0, 0, 0]])
        result = aggregate_ocr_predictions(
            predictions,
            FakeDataset(columns),
            id2label={0: "O", 1: "B-X", 2: "I-X"},
            label2id={"O": 0, "B-X": 1, "I-X": 2},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.y_true.tolist(), [1, 2])
        self.assertEqual(result.y_pred.tolist(), [1, 2])
        metrics = entity_metrics(result.sequences)
        self.assertEqual(metrics["n_gold_entities"], 1)
        self.assertEqual(metrics["entity_micro_f1"], 1.0)

    def test_ocr_token_metrics_do_not_forgive_b_i_error(self):
        columns = {
            "metric_source_indices": [[0, 1]],
            "metric_gold_labels": [[1, 2]],
            "metric_doc_key": ["doc"],
            "metric_record_index": [0],
            "metric_original_token_count": [2],
        }
        result = aggregate_ocr_predictions(
            np.asarray([[2, 2]]),
            FakeDataset(columns),
            id2label={0: "O", 1: "B-X", 2: "I-X"},
            label2id={"O": 0, "B-X": 1, "I-X": 2},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.y_true.tolist(), [1, 2])
        self.assertEqual(result.y_pred.tolist(), [2, 2])
        self.assertNotEqual(result.y_true.tolist(), result.y_pred.tolist())

    def test_numeric_layout_projection_starts_as_noop_and_learns(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace()
                self.embedding = torch.nn.Embedding(16, 8)
                self.classifier = torch.nn.Linear(8, 3)

            def get_input_embeddings(self):
                return self.embedding

            def resize_token_embeddings(self, size):
                del size

            def forward(self, inputs_embeds=None, attention_mask=None, labels=None, return_dict=None):
                del attention_mask, labels, return_dict
                return SimpleNamespace(logits=self.classifier(inputs_embeds))

        base = TinyModel()
        wrapped = NumericLayoutTokenClassifier(base)
        input_ids = torch.tensor([[1, 2]])
        baseline = base(inputs_embeds=base.get_input_embeddings()(input_ids)).logits
        output = wrapped(
            input_ids=input_ids,
            bbox=torch.tensor([[[0, 0, 100, 100], [100, 100, 200, 200]]]),
            bbox_mask=torch.tensor([[1, 1]]),
            page_ids=torch.tensor([[0, 0]]),
        ).logits
        self.assertTrue(torch.allclose(baseline, output))
        output.sum().backward()
        self.assertIsNotNone(wrapped.layout_projection.weight.grad)
        self.assertGreater(float(wrapped.layout_projection.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
