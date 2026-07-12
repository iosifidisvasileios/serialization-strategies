from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def split_bio_label(label: str) -> tuple[str, str | None]:
    label = str(label)
    if label == "O" or "-" not in label:
        return ("O", None)
    prefix, entity = label.split("-", 1)
    if prefix not in {"B", "I"} or not entity:
        return ("O", None)
    return (prefix, entity)


def repair_bio_strings(
    labels: Sequence[str], source_indices: Sequence[int] | None = None
) -> list[str]:
    """Return a valid BIO sequence, breaking entities across source-index gaps."""
    repaired: list[str] = []
    active_entity: str | None = None
    previous_source: int | None = None
    for position, raw_label in enumerate(labels):
        source_index = position if source_indices is None else int(source_indices[position])
        if previous_source is not None and source_index != previous_source + 1:
            active_entity = None
        prefix, entity = split_bio_label(str(raw_label))
        if entity is None:
            repaired.append("O")
            active_entity = None
        elif prefix == "I" and active_entity == entity:
            repaired.append(f"I-{entity}")
            active_entity = entity
        else:
            repaired.append(f"B-{entity}")
            active_entity = entity
        previous_source = source_index
    return repaired


def repair_bio_label_ids(
    label_ids: Sequence[int],
    *,
    id2label: Mapping[int, str],
    label2id: Mapping[str, int],
    ignore_label: int,
) -> list[int]:
    """Repair BIO at a model window boundary while keeping ignored items transparent."""
    output: list[int] = []
    active_entity: str | None = None
    for raw_id in label_ids:
        label_id = int(raw_id)
        if label_id == ignore_label:
            output.append(ignore_label)
            continue
        prefix, entity = split_bio_label(id2label[label_id])
        if entity is None:
            output.append(label2id["O"])
            active_entity = None
        elif prefix == "I" and active_entity == entity:
            output.append(label2id.get(f"I-{entity}", label_id))
            active_entity = entity
        else:
            output.append(label2id.get(f"B-{entity}", label_id))
            active_entity = entity
    return output


def canonical_labels_from_record(record: Mapping[str, Any], ignore_label: int) -> list[str]:
    """Recover canonical OCR-token gold labels from old or new serialized records."""
    original = record.get("original_token_labels")
    if isinstance(original, list):
        return [str(label) for label in original]

    raw_labels = list(record.get("labels", []))
    raw_sources = record.get("source_token_indices")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(raw_labels):
        # Backward-compatible fallback for records that predate source maps.
        real_labels = [
            str(label)
            for label in raw_labels
            if label not in {ignore_label, str(ignore_label), None}
        ]
        return real_labels

    real_sources = [int(source) for source in raw_sources if source is not None]
    n_tokens = int(record.get("original_token_count") or (max(real_sources) + 1 if real_sources else 0))
    canonical = ["O"] * n_tokens
    for source, label in zip(raw_sources, raw_labels):
        if source is None or label in {ignore_label, str(ignore_label), None}:
            continue
        source_index = int(source)
        if 0 <= source_index < n_tokens:
            canonical[source_index] = str(label)
    return repair_bio_strings(canonical, list(range(n_tokens)))


def rebuild_bio_for_source_order(
    canonical_labels: Sequence[str],
    source_indices: Sequence[int | None],
    *,
    ignore_label: int,
) -> list[str | int]:
    """Re-encode canonical entity instances in an arbitrary serialized order."""
    memberships: list[tuple[str, int] | None] = []
    active_entity: str | None = None
    instance = -1
    for label in canonical_labels:
        prefix, entity = split_bio_label(str(label))
        if entity is None:
            memberships.append(None)
            active_entity = None
            continue
        if prefix != "I" or active_entity != entity:
            instance += 1
        active_entity = entity
        memberships.append((entity, instance))

    output: list[str | int] = []
    previous: tuple[str, int] | None = None
    for source in source_indices:
        if source is None:
            output.append(ignore_label)
            continue
        source_index = int(source)
        if source_index < 0 or source_index >= len(memberships):
            raise IndexError(
                f"source_token_index {source_index} is outside 0..{len(memberships) - 1}."
            )
        membership = memberships[source_index]
        if membership is None:
            output.append("O")
            previous = None
            continue
        entity, _ = membership
        output.append(f"{'I' if membership == previous else 'B'}-{entity}")
        previous = membership
    return output


@dataclass
class OcrEvaluationData:
    y_true: np.ndarray
    y_pred: np.ndarray
    sequences: list[tuple[tuple[str, int], list[int], list[str], list[str]]]
    n_missing_ocr_tokens: int


def aggregate_ocr_predictions(
    predictions,
    dataset,
    *,
    id2label: Mapping[int, str],
    label2id: Mapping[str, int],
) -> OcrEvaluationData | None:
    """Collapse model positions/overlaps to one prediction per canonical OCR token."""
    required = {
        "metric_source_indices",
        "metric_gold_labels",
        "metric_doc_key",
        "metric_record_index",
        "metric_original_token_count",
    }
    if dataset is None or not required.issubset(set(getattr(dataset, "column_names", []))):
        return None

    prediction_ids = np.asarray(predictions)
    if prediction_ids.ndim == 3:
        prediction_ids = np.argmax(prediction_ids, axis=-1)
    if len(prediction_ids) != len(dataset):
        raise AssertionError(
            f"Metric dataset/prediction row mismatch: {len(dataset)} vs {len(prediction_ids)}."
        )

    votes: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    gold_by_key: dict[tuple[str, int, int], int] = {}
    expected_by_record: dict[tuple[str, int], int] = {}

    source_rows = dataset["metric_source_indices"]
    gold_rows = dataset["metric_gold_labels"]
    doc_keys = dataset["metric_doc_key"]
    record_indices = dataset["metric_record_index"]
    original_counts = dataset["metric_original_token_count"]
    for row_index, pred_row in enumerate(prediction_ids):
        record_key = (str(doc_keys[row_index]), int(record_indices[row_index]))
        expected_by_record[record_key] = int(original_counts[row_index])
        for position, (source, gold) in enumerate(zip(source_rows[row_index], gold_rows[row_index])):
            source_index = int(source)
            gold_id = int(gold)
            if source_index < 0 or gold_id < 0:
                continue
            key = (*record_key, source_index)
            existing = gold_by_key.get(key)
            if existing is not None and existing != gold_id:
                raise AssertionError(f"Inconsistent gold labels for OCR token {key}: {existing} vs {gold_id}.")
            gold_by_key[key] = gold_id
            votes[key].append(int(pred_row[position]))

    grouped: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
    for (doc_key, record_index, source_index), gold_id in gold_by_key.items():
        token_votes = votes[(doc_key, record_index, source_index)]
        counts = Counter(token_votes)
        best_count = max(counts.values())
        # list order is deterministic evaluation order and resolves ties.
        pred_id = next(value for value in token_votes if counts[value] == best_count)
        grouped[(doc_key, record_index)].append((source_index, gold_id, pred_id))

    sequences = []
    y_true: list[int] = []
    y_pred: list[int] = []
    missing = 0
    for record_key in sorted(expected_by_record):
        rows = sorted(grouped.get(record_key, []), key=lambda row: row[0])
        source_indices = [row[0] for row in rows]
        # Token-level metrics must retain the model's exact BIO class.  BIO
        # repair is appropriate only for span decoding: applying it to these
        # ids would incorrectly forgive an I-X prediction where B-X is gold.
        gold_ids = [row[1] for row in rows]
        pred_ids = [row[2] for row in rows]
        default_label = id2label.get(int(label2id["O"]), "O")
        gold_strings = [id2label.get(label_id, default_label) for label_id in gold_ids]
        pred_strings = [id2label.get(label_id, default_label) for label_id in pred_ids]
        gold_strings = repair_bio_strings(gold_strings, source_indices)
        pred_strings = repair_bio_strings(pred_strings, source_indices)
        y_true.extend(gold_ids)
        y_pred.extend(pred_ids)
        sequences.append((record_key, source_indices, gold_strings, pred_strings))
        missing += max(0, expected_by_record[record_key] - len(source_indices))

    return OcrEvaluationData(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        sequences=sequences,
        n_missing_ocr_tokens=missing,
    )


def bio_spans(labels: Sequence[str], source_indices: Sequence[int]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    active_entity: str | None = None
    active_start: int | None = None
    previous_source: int | None = None

    def close(end: int | None) -> None:
        nonlocal active_entity, active_start
        if active_entity is not None and active_start is not None and end is not None:
            spans.add((active_entity, active_start, end))
        active_entity = None
        active_start = None

    for label, source_index in zip(labels, source_indices):
        if previous_source is not None and source_index != previous_source + 1:
            close(previous_source)
        prefix, entity = split_bio_label(label)
        if entity is None:
            close(previous_source)
        elif prefix == "B" or active_entity != entity:
            close(previous_source)
            active_entity = entity
            active_start = source_index
        previous_source = source_index
    close(previous_source)
    return spans


def entity_metrics(
    sequences: Sequence[tuple[tuple[str, int], list[int], list[str], list[str]]]
) -> dict[str, float | int]:
    gold_all: set[tuple[str, int, str, int, int]] = set()
    pred_all: set[tuple[str, int, str, int, int]] = set()
    for (doc_key, record_index), source_indices, gold, pred in sequences:
        gold_all.update(
            (doc_key, record_index, entity, start, end)
            for entity, start, end in bio_spans(gold, source_indices)
        )
        pred_all.update(
            (doc_key, record_index, entity, start, end)
            for entity, start, end in bio_spans(pred, source_indices)
        )

    correct = gold_all & pred_all
    precision = len(correct) / len(pred_all) if pred_all else 0.0
    recall = len(correct) / len(gold_all) if gold_all else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    entity_types = sorted({row[2] for row in gold_all | pred_all})
    type_f1s = []
    for entity in entity_types:
        gold_type = {row for row in gold_all if row[2] == entity}
        pred_type = {row for row in pred_all if row[2] == entity}
        correct_type = gold_type & pred_type
        p = len(correct_type) / len(pred_type) if pred_type else 0.0
        r = len(correct_type) / len(gold_type) if gold_type else 0.0
        type_f1s.append(2 * p * r / (p + r) if p + r else 0.0)

    return {
        "entity_micro_precision": float(precision),
        "entity_micro_recall": float(recall),
        "entity_micro_f1": float(f1),
        "entity_macro_f1": float(np.mean(type_f1s)) if type_f1s else 0.0,
        "n_gold_entities": len(gold_all),
        "n_pred_entities": len(pred_all),
        "n_correct_entities": len(correct),
    }
