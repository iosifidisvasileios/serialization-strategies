from __future__ import annotations
import json
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from transformers import AutoTokenizer

# Add parent directories to path for direct script execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src" / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "training"))

try:
    from .experiment_config import ExperimentConfig, ModelSpec
    from .layout_model import ALL_LAYOUT_TOKENS, canonical_layout_token
    from .ocr_metrics import (
        canonical_labels_from_record,
        rebuild_bio_for_source_order,
        repair_bio_label_ids,
    )
except ImportError:
    from experiment_config import ExperimentConfig, ModelSpec
    from layout_model import ALL_LAYOUT_TOKENS, canonical_layout_token
    from ocr_metrics import (
        canonical_labels_from_record,
        rebuild_bio_for_source_order,
        repair_bio_label_ids,
    )


@dataclass
class ExperimentData:
    records_by_dataset_strategy: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    selected_strategy_files_by_dataset: dict[str, dict[str, Path]] = field(default_factory=dict)
    selected_datasets: list[str] = field(default_factory=list)
    selected_strategies_by_dataset: dict[str, list[str]] = field(default_factory=dict)
    label_list: list[str] = field(default_factory=list)
    label2id: dict[str, int] = field(default_factory=dict)
    id2label: dict[int, str] = field(default_factory=dict)
    non_o_label_ids: list[int] = field(default_factory=list)
    cv_split_assignments_by_dataset: dict[str, dict[int, dict[str, str]]] = field(
        default_factory=dict
    )
    doc_strata_by_dataset: dict[str, dict[str, str]] = field(default_factory=dict)
    cv_assignment_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    split_plan_paths: dict[str, Path] = field(default_factory=dict)
    examples_by_dataset_strategy: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    fair_window_plan_cache: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    tokenization_summary_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TokenizedCorpus:
    dataset: Dataset
    indices_by_doc_key: dict[str, list[int]]
    summary: dict[str, Any]


class DataPipeline:
    """Load, split, chunk, tokenize, and expose fold-specific datasets."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.data = ExperimentData()
        self.state = self.data

    def qprint(self, *args, **kwargs) -> None:
        if self.cfg.show_run_progress:
            print(*args, **kwargs)

    @contextmanager
    def quiet_section(self, enabled: bool = True):
        if not enabled:
            yield
            return
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield

    def prepare(self) -> None:
        read_summary = self.load_records()
        read_summary.to_csv(self.cfg.results_root / "dataset_read_summary.csv", index=False)
        label_df = self.build_label_maps()
        label_df.to_csv(self.cfg.results_root / "label_map.csv", index=False)
        split_summary = self.build_all_cv_assignments()
        split_summary.to_csv(self.cfg.results_root / "strategy_split_summary.csv", index=False)
        chunk_summary = self.build_examples()
        chunk_summary.to_csv(self.cfg.results_root / "base_chunk_summary.csv", index=False)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON at {path}:{line_idx + 1}") from e
        return records

    @staticmethod
    def discover_strategy_files(processed_root: Path) -> dict[str, Path]:
        strategy_files = {}
        for path in sorted(processed_root.glob("*/all.jsonl")):
            strategy_files[path.parent.name] = path
        return strategy_files

    def discover_dataset_roots(self) -> dict[str, Path]:
        if not self.cfg.processed_datasets_root.exists():
            raise FileNotFoundError(
                f"Processed datasets root does not exist: {self.cfg.processed_datasets_root}"
            )
        dataset_roots: dict[str, Path] = {}
        for candidate_root in sorted(self.cfg.processed_datasets_root.iterdir()):
            if candidate_root.is_dir() and self.discover_strategy_files(candidate_root):
                dataset_roots.setdefault(candidate_root.name, candidate_root)
        if self.cfg.datasets_to_run != "all":
            requested = set(self.cfg.datasets_to_run)
            missing = sorted(requested.difference(dataset_roots))
            if missing:
                raise AssertionError(f"Requested datasets not found: {missing}")
            dataset_roots = {
                name: root for name, root in dataset_roots.items() if name in requested
            }
        if not dataset_roots:
            raise AssertionError(
                "No processed datasets found. Expected folders under data/processed containing */all.jsonl."
            )
        return dataset_roots

    def load_records(self) -> pd.DataFrame:
        dataset_roots = self.discover_dataset_roots()
        read_summary_rows = []
        for dataset_name, dataset_root in dataset_roots.items():
            available_strategy_files = self.discover_strategy_files(dataset_root)
            print(f"\nDataset: {dataset_name}")
            print("Available strategies:")
            for name, path in available_strategy_files.items():
                print(f"  {name:24s} -> {path}")
            if self.cfg.strategies_to_run == "all":
                selected_strategy_files = available_strategy_files
            else:
                selected_strategy_files = {
                    name: available_strategy_files[name]
                    for name in self.cfg.strategies_to_run
                    if name in available_strategy_files
                }
            if not selected_strategy_files:
                print(f"Skipping dataset {dataset_name!r}; no selected strategies found.")
                continue

            records_by_strategy = {
                strategy: self.read_jsonl(path)
                for strategy, path in selected_strategy_files.items()
            }
            doc_keys_by_strategy = {
                strategy: {
                    self.record_doc_key(record, "record", record_index)
                    for record_index, record in enumerate(records)
                }
                for strategy, records in records_by_strategy.items()
            }
            common_doc_keys = set.intersection(*doc_keys_by_strategy.values())
            if not common_doc_keys:
                raise AssertionError(
                    f"Selected strategies for dataset {dataset_name!r} have no common documents."
                )
            if any(keys != common_doc_keys for keys in doc_keys_by_strategy.values()):
                dropped = {
                    strategy: len(keys - common_doc_keys)
                    for strategy, keys in doc_keys_by_strategy.items()
                }
                warnings.warn(
                    f"Restricting dataset {dataset_name!r} to the common strategy cohort; "
                    f"strategy-only document counts: {dropped}.",
                    RuntimeWarning,
                )

            selected_doc_keys = sorted(common_doc_keys)
            if self.cfg.sampling is not None:
                rng = random.Random(self.cfg.split_seed)
                rng.shuffle(selected_doc_keys)
                selected_doc_keys = selected_doc_keys[: self.cfg.sampling]
            selected_doc_key_set = set(selected_doc_keys)

            self.state.selected_strategy_files_by_dataset[dataset_name] = selected_strategy_files
            self.state.records_by_dataset_strategy[dataset_name] = {}
            for strategy, path in selected_strategy_files.items():
                records = [
                    record
                    for record_index, record in enumerate(records_by_strategy[strategy])
                    if self.record_doc_key(record, "record", record_index)
                    in selected_doc_key_set
                ]
                self.state.records_by_dataset_strategy[dataset_name][strategy] = records
                read_summary_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "strategy": strategy,
                        "n_records": len(records),
                        "path": str(path),
                    }
                )
        if not self.state.records_by_dataset_strategy:
            raise AssertionError("No records loaded after dataset/strategy filtering.")
        self.state.selected_datasets = sorted(self.state.records_by_dataset_strategy.keys())
        self.state.selected_strategies_by_dataset = {
            dataset_name: sorted(records_by_strategy.keys())
            for dataset_name, records_by_strategy in self.state.records_by_dataset_strategy.items()
        }
        return pd.DataFrame(read_summary_rows)

    def is_real_label(self, x: Any) -> bool:
        return isinstance(x, str) and x != "" and (x != str(self.cfg.ignore_label))

    @staticmethod
    def sort_bio_labels(labels: set[str]) -> list[str]:
        labels = set(labels)
        labels.add("O")

        def sort_key(label: str):
            if label == "O":
                return ("", 0, "")
            if "-" in label:
                prefix, entity = label.split("-", 1)
                prefix_rank = {"B": 0, "I": 1}.get(prefix, 9)
                return (entity, prefix_rank, label)
            return (label, 9, label)

        return ["O"] + sorted([x for x in labels if x != "O"], key=sort_key)

    def build_label_maps(self) -> pd.DataFrame:
        label_set = set()
        for records_by_strategy in self.state.records_by_dataset_strategy.values():
            for records in records_by_strategy.values():
                for rec in records:
                    for label in canonical_labels_from_record(rec, self.cfg.ignore_label):
                        if self.is_real_label(label):
                            label_set.add(label)
        self.state.label_list = self.sort_bio_labels(label_set)
        self.state.label2id = {label: i for i, label in enumerate(self.state.label_list)}
        self.state.id2label = {i: label for label, i in self.state.label2id.items()}
        self.state.non_o_label_ids = [i for i, label in self.state.id2label.items() if label != "O"]
        print("n_labels:", len(self.state.label_list))
        print(self.state.label_list)
        return pd.DataFrame(
            {"label_id": list(range(len(self.state.label_list))), "label": self.state.label_list}
        )

    @staticmethod
    def record_doc_key(rec: dict, fallback_prefix: str, fallback_index: int) -> str:
        for key in ["doc_key", "id", "doc_id"]:
            value = rec.get(key)
            if value is not None and str(value) != "":
                return str(value)
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ["doc_key", "id", "doc_id", "original_filename"]:
                value = metadata.get(key)
                if value is not None and str(value) != "":
                    return str(value)
        return f"{fallback_prefix}::fallback::{fallback_index}"

    def normalize_label_name(self, label: Any) -> Optional[str]:
        if not self.is_real_label(label):
            return None
        if label == "O":
            return "O"
        if isinstance(label, str) and "-" in label:
            return label.split("-", 1)[1]
        return str(label)

    def build_doc_strata(self, records_by_strategy: dict[str, list[dict]]) -> dict[str, str]:
        entity_counts_by_doc = defaultdict(Counter)
        for strategy, records in records_by_strategy.items():
            for rec_idx, rec in enumerate(records):
                doc_key = self.record_doc_key(rec, "record", rec_idx)
                for label in rec.get("labels", []):
                    normalized = self.normalize_label_name(label)
                    if normalized is None or normalized == "O":
                        continue
                    entity_counts_by_doc[doc_key][normalized] += 1
        all_doc_keys = sorted(
            {
                self.record_doc_key(rec, "record", rec_idx)
                for strategy, records in records_by_strategy.items()
                for rec_idx, rec in enumerate(records)
            }
        )
        strata = {}
        for doc_key in all_doc_keys:
            counts = entity_counts_by_doc.get(doc_key, Counter())
            if not counts:
                strata[doc_key] = "only_o"
            else:
                dominant_entity, _ = counts.most_common(1)[0]
                strata[doc_key] = f"has_entities:{dominant_entity}"
        return strata

    @staticmethod
    def can_use_stratification(labels: list[str], min_count: int) -> bool:
        counts = Counter(labels)
        return len(counts) > 1 and min(counts.values()) >= min_count

    def split_train_validation(
        self, trainval_keys: list[str], key_to_stratum: dict[str, str], seed: int
    ) -> tuple[set[str], set[str], str]:
        """Split the outer-training folds into train and validation documents.

        The outer StratifiedKFold already defines the held-out test fold.
        Validation is therefore sampled only from the k-1 outer-training folds.
        By default, validation receives 10% of the outer-training documents.
        Stratification is used when the document strata are sufficiently populated;
        otherwise the code falls back to a deterministic shuffled split.
        """
        if not 0.0 < self.cfg.validation_fraction_of_train < 1.0:
            raise ValueError(
                f"validation_fraction_of_train must be between 0 and 1, got {self.cfg.validation_fraction_of_train}."
            )
        if len(trainval_keys) < 2:
            raise AssertionError(
                "Each outer training partition must contain at least two documents to create a validation split."
            )
        y = [key_to_stratum[k] for k in trainval_keys]
        n_validation = int(round(len(trainval_keys) * self.cfg.validation_fraction_of_train))
        n_validation = max(1, min(n_validation, len(trainval_keys) - 1))
        n_classes = len(set(y))
        can_stratify = (
            self.can_use_stratification(y, min_count=2)
            and n_validation >= n_classes
            and (len(trainval_keys) - n_validation >= n_classes)
        )
        validation_size: int | float = n_validation
        if can_stratify:
            train_keys, validation_keys = train_test_split(
                trainval_keys,
                test_size=validation_size,
                random_state=seed,
                shuffle=True,
                stratify=y,
            )
            inner_splitter_name = "StratifiedShuffleSplit"
        else:
            train_keys, validation_keys = train_test_split(
                trainval_keys,
                test_size=validation_size,
                random_state=seed,
                shuffle=True,
                stratify=None,
            )
            inner_splitter_name = "ShuffleSplit"
        return (set(train_keys), set(validation_keys), inner_splitter_name)

    def build_cv_split_assignments(
        self, records_by_strategy: dict[str, list[dict]], n_folds: int, seed: int
    ):
        key_to_stratum = self.build_doc_strata(records_by_strategy)
        doc_keys = np.array(sorted(key_to_stratum.keys()))
        if len(doc_keys) < n_folds:
            raise ValueError(
                f"Not enough documents for {n_folds}-fold CV: found {len(doc_keys)} documents."
            )
        y = np.array([key_to_stratum[k] for k in doc_keys])
        if self.can_use_stratification(y.tolist(), min_count=n_folds):
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(doc_keys, y)
            outer_splitter_name = "StratifiedKFold"
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(doc_keys)
            outer_splitter_name = "KFold"
        assignments_by_fold = {}
        rows = []
        for fold_index, (trainval_idx, test_idx) in enumerate(split_iter):
            trainval_keys = sorted(doc_keys[trainval_idx].tolist())
            test_keys = set(doc_keys[test_idx].tolist())
            train_keys, validation_keys, inner_splitter_name = self.split_train_validation(
                trainval_keys=trainval_keys, key_to_stratum=key_to_stratum, seed=seed + fold_index
            )
            assignment = {}
            assignment.update({k: "train" for k in train_keys})
            assignment.update({k: "validation" for k in validation_keys})
            assignment.update({k: "test" for k in test_keys})
            assert set(assignment) == set(doc_keys.tolist())
            assert not validation_keys & test_keys
            assert not train_keys & validation_keys
            assert not train_keys & test_keys
            assignments_by_fold[fold_index] = assignment
            for doc_key, split in assignment.items():
                rows.append(
                    {
                        "fold": fold_index,
                        "doc_key": doc_key,
                        "split": split,
                        "stratum": key_to_stratum[doc_key],
                        "outer_splitter": outer_splitter_name,
                        "inner_validation_splitter": inner_splitter_name,
                        "validation_fraction_of_train": self.cfg.validation_fraction_of_train,
                    }
                )
        assignment_df = pd.DataFrame(rows).sort_values(["fold", "split", "doc_key"])
        return (assignments_by_fold, key_to_stratum, assignment_df)

    def build_all_cv_assignments(self) -> pd.DataFrame:
        for dataset_name, records_by_strategy in self.state.records_by_dataset_strategy.items():
            assignments, doc_strata, assignment_df = self.build_cv_split_assignments(
                records_by_strategy=records_by_strategy,
                n_folds=self.cfg.n_folds,
                seed=self.cfg.split_seed,
            )
            self.state.cv_split_assignments_by_dataset[dataset_name] = assignments
            self.state.doc_strata_by_dataset[dataset_name] = doc_strata
            self.state.cv_assignment_dfs[dataset_name] = assignment_df
            split_plan_path = (
                self.cfg.results_root
                / f"{self.safe_name(dataset_name)}_cv_doc_split_assignments.csv"
            )
            assignment_df.to_csv(split_plan_path, index=False)
            self.state.split_plan_paths[dataset_name] = split_plan_path
            print("Saved CV split plan to:", split_plan_path)
            print("Outer splitter used:", assignment_df["outer_splitter"].iloc[0])
            print(
                "Inner validation splitter(s):",
                sorted(assignment_df["inner_validation_splitter"].unique().tolist()),
            )
        strategy_split_rows = []
        for dataset_name, assignments_by_fold in self.state.cv_split_assignments_by_dataset.items():
            records_by_strategy = self.state.records_by_dataset_strategy[dataset_name]
            for fold_index, assignment in assignments_by_fold.items():
                for strategy, records in records_by_strategy.items():
                    strategy_doc_keys = sorted(
                        {self.record_doc_key(rec, "record", i) for i, rec in enumerate(records)}
                    )
                    counts = Counter((assignment[k] for k in strategy_doc_keys))
                    n_docs = len(strategy_doc_keys)
                    strategy_split_rows.append(
                        {
                            "dataset_name": dataset_name,
                            "fold": fold_index,
                            "strategy": strategy,
                            "n_docs": n_docs,
                            "train_docs": counts.get("train", 0),
                            "validation_docs": counts.get("validation", 0),
                            "test_docs": counts.get("test", 0),
                            "observed_train_fraction": (
                                counts.get("train", 0) / n_docs if n_docs else 0
                            ),
                            "observed_validation_fraction": (
                                counts.get("validation", 0) / n_docs if n_docs else 0
                            ),
                            "observed_test_fraction": (
                                counts.get("test", 0) / n_docs if n_docs else 0
                            ),
                        }
                    )
        return pd.DataFrame(strategy_split_rows)

    def normalize_record_labels(self, raw_labels: list) -> list[int]:
        label_ids = []
        for label in raw_labels:
            if (
                label == self.cfg.ignore_label
                or label == str(self.cfg.ignore_label)
                or label is None
            ):
                label_ids.append(self.cfg.ignore_label)
            elif isinstance(label, str):
                label_ids.append(self.state.label2id[label])
            else:
                label_ids.append(self.cfg.ignore_label)
        return label_ids

    def _training_record_view(self, record: dict) -> dict[str, Any]:
        """Normalize old/new serialized records without requiring regeneration."""
        tokens = list(record.get("tokens", []))
        raw_labels = list(record.get("labels", []))
        if len(tokens) != len(raw_labels):
            raise ValueError(
                f"Serialized record has {len(tokens)} tokens but {len(raw_labels)} labels."
            )
        n_items = len(tokens)

        sources = record.get("source_token_indices")
        if not isinstance(sources, list) or len(sources) != n_items:
            sources = []
            next_source = 0
            for label in raw_labels:
                if label in {self.cfg.ignore_label, str(self.cfg.ignore_label), None}:
                    sources.append(None)
                else:
                    sources.append(next_source)
                    next_source += 1
        sources = [None if value is None else int(value) for value in sources]

        roles = record.get("layout_roles")
        if not isinstance(roles, list) or len(roles) != n_items:
            roles = ["ocr_token" if source is not None else "layout" for source in sources]
        roles = [str(role or ("ocr_token" if source is not None else "layout")) for role, source in zip(roles, sources)]

        canonical_labels = canonical_labels_from_record(record, self.cfg.ignore_label)
        serialized_labels = rebuild_bio_for_source_order(
            canonical_labels,
            sources,
            ignore_label=self.cfg.ignore_label,
        )

        normalized_tokens = []
        for token, source, role in zip(tokens, sources, roles):
            if source is None or role != "ocr_token":
                role_token = canonical_layout_token(role)
                value_token = str(token).strip()
                normalized_tokens.append(
                    role_token
                    if not value_token or value_token == role_token
                    else f"{role_token} {value_token}"
                )
            else:
                text = str(token)
                normalized_tokens.append(text if text.strip() else "<EMPTY_OCR>")

        def aligned_list(key: str, default):
            values = record.get(key)
            if not isinstance(values, list) or len(values) != n_items:
                return [default() if callable(default) else default for _ in range(n_items)]
            return list(values)

        return {
            "tokens": normalized_tokens,
            "labels": serialized_labels,
            "source_token_indices": sources,
            "layout_roles": roles,
            "item_attrs": aligned_list("item_attrs", dict),
            "pages": aligned_list("pages", None),
            "normalized_bboxes": aligned_list("normalized_bboxes", None),
            "canonical_labels": canonical_labels,
            "original_token_count": len(canonical_labels),
        }

    @staticmethod
    def _record_source_views(view: dict[str, Any]) -> list[dict[str, Any]]:
        """Attach layout items to real OCR tokens and track active group markers."""
        persistent_roles = {"page", "block", "line", "column", "xycut_region"}
        active: dict[str, int] = {}
        pending_prefix: list[int] = []
        views: list[dict[str, Any]] = []
        seen_sources: set[int] = set()

        for position, (source, role) in enumerate(
            zip(view["source_token_indices"], view["layout_roles"])
        ):
            if source is None:
                if role in persistent_roles:
                    if role == "page":
                        active.clear()
                    elif role in {"block", "column", "xycut_region"}:
                        active.pop("line", None)
                    active[role] = position
                elif role == "coord_suffix" and views:
                    views[-1]["suffix"].append(position)
                else:
                    pending_prefix.append(position)
                continue

            source_index = int(source)
            if source_index in seen_sources:
                raise AssertionError(f"Duplicate source_token_index {source_index} in one record.")
            seen_sources.add(source_index)
            views.append(
                {
                    "source": source_index,
                    "active": dict(active),
                    "prefix": pending_prefix,
                    "real": position,
                    "suffix": [],
                }
            )
            pending_prefix = []

        if pending_prefix and views:
            views[-1]["suffix"].extend(pending_prefix)
        return views

    @staticmethod
    def _source_view_lookup(views: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {int(source_view["source"]): source_view for source_view in views}

    @staticmethod
    def _render_source_window(
        source_views_by_id: dict[int, dict[str, Any]],
        selected_sources: set[int],
    ) -> list[int]:
        priority = ("page", "block", "column", "xycut_region", "line")
        positions: list[int] = []
        previous_active: dict[str, int] = {}
        selected_views = sorted(
            (source_views_by_id[source] for source in selected_sources),
            key=lambda source_view: int(source_view["real"]),
        )
        for source_view in selected_views:
            active = source_view["active"]
            for role in priority:
                marker_position = active.get(role)
                if marker_position is not None and previous_active.get(role) != marker_position:
                    positions.append(marker_position)
            positions.extend(source_view["prefix"])
            positions.append(source_view["real"])
            positions.extend(source_view["suffix"])
            previous_active = dict(active)
        return positions

    @staticmethod
    def _wordpiece_lengths(tokenizer, words: list[str]) -> list[int]:
        if not words:
            return []
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=False,
        )
        lengths = [0] * len(words)
        for word_id in encoded.word_ids():
            if word_id is not None:
                lengths[int(word_id)] += 1
        return lengths

    def _record_slots(self, strategy: str, records: list[dict]) -> dict[tuple[str, int], tuple[int, dict]]:
        occurrences: dict[str, int] = defaultdict(int)
        slots = {}
        for record_index, record in enumerate(records):
            doc_key = self.record_doc_key(record, "record", record_index)
            occurrence = occurrences[doc_key]
            occurrences[doc_key] += 1
            slots[(doc_key, occurrence)] = (record_index, record)
        return slots

    def _example_from_positions(
        self,
        *,
        dataset_name: str,
        strategy: str,
        doc_key: str,
        record_index: int,
        chunk_index: int,
        source_ids: list[int],
        positions: list[int],
        view: dict[str, Any],
    ) -> dict[str, Any]:
        tokens = [view["tokens"][position] for position in positions]
        labels = [view["labels"][position] for position in positions]
        word_label_ids = repair_bio_label_ids(
            self.normalize_record_labels(labels),
            id2label=self.state.id2label,
            label2id=self.state.label2id,
            ignore_label=self.cfg.ignore_label,
        )
        word_sources = [view["source_token_indices"][position] for position in positions]
        metric_gold_ids = []
        word_bboxes = []
        word_bbox_masks = []
        word_pages = []
        for position, source in zip(positions, word_sources):
            if source is None:
                metric_gold_ids.append(self.cfg.ignore_label)
            else:
                label = view["canonical_labels"][int(source)]
                metric_gold_ids.append(self.state.label2id.get(label, self.state.label2id["O"]))

            raw_bbox = None if strategy == "plain_text" else view["normalized_bboxes"][position]
            if (
                isinstance(raw_bbox, (list, tuple))
                and len(raw_bbox) == 4
                and all(isinstance(value, (int, float, np.integer, np.floating)) for value in raw_bbox)
            ):
                bbox = [max(0, min(1000, int(round(value)))) for value in raw_bbox]
                valid_bbox = bbox[2] >= bbox[0] and bbox[3] >= bbox[1]
            else:
                bbox = [0, 0, 0, 0]
                valid_bbox = False
            word_bboxes.append(bbox if valid_bbox else [0, 0, 0, 0])
            word_bbox_masks.append(1 if valid_bbox else 0)
            if strategy == "plain_text":
                page = -1
            else:
                try:
                    page = int(view["pages"][position])
                except (TypeError, ValueError):
                    page = -1
            word_pages.append(page)

        return {
            "dataset_name": dataset_name,
            "strategy": strategy,
            "doc_key": doc_key,
            "record_index": record_index,
            "chunk_index": chunk_index,
            "source_example_id": f"{doc_key}::{record_index}::{chunk_index}",
            "word_start": source_ids[0],
            "word_end": source_ids[-1] + 1,
            "tokens": tokens,
            "word_label_ids": word_label_ids,
            "word_source_token_indices": [(-1 if value is None else int(value)) for value in word_sources],
            "word_metric_gold_label_ids": metric_gold_ids,
            "word_bboxes": word_bboxes,
            "word_bbox_masks": word_bbox_masks,
            "word_page_ids": word_pages,
            "metric_doc_key": doc_key,
            "metric_record_index": record_index,
            "metric_original_token_count": view["original_token_count"],
        }

    def _ensure_fair_examples(
        self,
        dataset_name: str,
        model_spec: ModelSpec,
        tokenizer,
    ) -> dict[str, list[dict]]:
        max_length = self.effective_max_length(tokenizer)
        cache_key = (
            dataset_name,
            model_spec.name,
            model_spec.model_name_or_path,
            getattr(tokenizer, "name_or_path", ""),
            len(tokenizer),
            max_length,
        )
        cached = self.state.fair_window_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        records_by_strategy = self.state.records_by_dataset_strategy[dataset_name]
        strategies = sorted(records_by_strategy)
        slot_maps = {
            strategy: self._record_slots(strategy, records_by_strategy[strategy])
            for strategy in strategies
        }
        slot_sets = [set(slots) for slots in slot_maps.values()]
        common_slots = set.intersection(*slot_sets) if slot_sets else set()
        if not common_slots:
            raise AssertionError(f"{dataset_name}: strategies have no common document records.")
        if any(slots != common_slots for slots in slot_sets):
            warnings.warn(
                f"{dataset_name}: strategy record sets differ; fair comparison uses "
                f"the {len(common_slots)} common records only.",
                RuntimeWarning,
            )

        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        usable_length = max_length - special_count
        if usable_length <= 0:
            raise ValueError(f"max_length={max_length} leaves no model content positions.")

        output: dict[str, list[dict]] = {strategy: [] for strategy in strategies}
        for doc_key, occurrence in sorted(common_slots):
            prepared = {}
            source_sets = []
            reference_labels = None
            for strategy in strategies:
                record_index, record = slot_maps[strategy][(doc_key, occurrence)]
                view = self._training_record_view(record)
                source_views = self._source_view_lookup(self._record_source_views(view))
                source_set = set(source_views)
                source_sets.append(source_set)
                if reference_labels is None:
                    reference_labels = view["canonical_labels"]
                elif view["canonical_labels"] != reference_labels:
                    raise AssertionError(
                        f"{dataset_name}/{doc_key}: canonical gold labels differ across strategies."
                    )
                position_lengths = self._wordpiece_lengths(tokenizer, view["tokens"])
                prepared[strategy] = {
                    "record_index": record_index,
                    "view": view,
                    "source_views": source_views,
                    "position_lengths": position_lengths,
                }

            common_sources = set.intersection(*source_sets) if source_sets else set()
            if not common_sources:
                continue
            if any(sources != common_sources for sources in source_sets):
                raise AssertionError(
                    f"{dataset_name}/{doc_key}: source-token sets differ across strategies."
                )
            expected_sources = set(range(len(reference_labels)))
            if common_sources != expected_sources:
                missing = sorted(expected_sources - common_sources)
                extra = sorted(common_sources - expected_sources)
                raise AssertionError(
                    f"{dataset_name}/{doc_key}: incomplete OCR source coverage; "
                    f"missing={missing[:10]}, extra={extra[:10]}."
                )
            source_ids = sorted(common_sources)

            start = 0
            chunk_index = 0
            while start < len(source_ids):
                end = start
                while end < len(source_ids):
                    if self.cfg.word_window_size > 0 and end - start >= self.cfg.word_window_size:
                        break
                    candidate_sources = set(source_ids[start : end + 1])
                    fits = True
                    for strategy in strategies:
                        item = prepared[strategy]
                        positions = self._render_source_window(
                            item["source_views"], candidate_sources
                        )
                        n_pieces = sum(item["position_lengths"][position] for position in positions)
                        if n_pieces > usable_length:
                            fits = False
                            break
                    if not fits:
                        break
                    end += 1

                if end == start:
                    raise ValueError(
                        f"{dataset_name}/{doc_key}: OCR token {source_ids[start]} plus its layout "
                        f"requires more than {usable_length} subtokens."
                    )

                # Move an entity wholly into the next window only when its B-
                # token is inside this window.  If it starts at/before `start`,
                # it is longer than the available capacity, so keep the maximal
                # fitting window and repair the next window's initial I- to B-.
                if end < len(source_ids):
                    next_label = str(reference_labels[source_ids[end]])
                    if next_label.startswith("I-"):
                        entity = next_label[2:]
                        entity_start = end
                        while (
                            entity_start > start
                            and str(reference_labels[source_ids[entity_start]])
                            == f"I-{entity}"
                        ):
                            entity_start -= 1
                        if (
                            entity_start > start
                            and str(reference_labels[source_ids[entity_start]])
                            == f"B-{entity}"
                        ):
                            end = entity_start

                window_sources = source_ids[start:end]
                selected = set(window_sources)
                for strategy in strategies:
                    item = prepared[strategy]
                    positions = self._render_source_window(item["source_views"], selected)
                    output[strategy].append(
                        self._example_from_positions(
                            dataset_name=dataset_name,
                            strategy=strategy,
                            doc_key=doc_key,
                            record_index=item["record_index"],
                            chunk_index=chunk_index,
                            source_ids=window_sources,
                            positions=positions,
                            view=item["view"],
                        )
                    )

                if end >= len(source_ids):
                    break
                overlap_words = self.cfg.word_window_stride
                if overlap_words <= 0 and self.cfg.tokenizer_stride > 0:
                    accumulated = 0
                    cursor = end
                    while cursor > start and accumulated < self.cfg.tokenizer_stride:
                        cursor -= 1
                        source = source_ids[cursor]
                        worst = 0
                        for strategy in strategies:
                            item = prepared[strategy]
                            positions = self._render_source_window(
                                item["source_views"], {source}
                            )
                            worst = max(
                                worst,
                                sum(item["position_lengths"][position] for position in positions),
                            )
                        accumulated += worst
                    overlap_words = end - cursor
                start = max(start + 1, end - overlap_words)
                chunk_index += 1

        chunk_counts = {strategy: len(examples) for strategy, examples in output.items()}
        if len(set(chunk_counts.values())) != 1:
            raise AssertionError(f"Fair chunk planning produced unequal counts: {chunk_counts}")

        self.state.fair_window_plan_cache[cache_key] = output
        return output

    def iter_word_windows(self, n_tokens: int):
        if n_tokens <= 0:
            return
        if self.cfg.word_window_size == 0:
            yield (0, n_tokens)
            return
        step = self.cfg.word_window_size - self.cfg.word_window_stride
        start = 0
        while start < n_tokens:
            end = min(n_tokens, start + self.cfg.word_window_size)
            yield (start, end)
            if end >= n_tokens:
                break
            start += step

    def make_base_examples_for_strategy(
        self, dataset_name: str, strategy: str, records: list[dict]
    ) -> list[dict]:
        examples = []
        for rec_idx, rec in enumerate(records):
            tokens = rec.get("tokens", [])
            labels = rec.get("labels", [])
            if len(tokens) != len(labels):
                raise ValueError(
                    f"{dataset_name}/{strategy} record {rec_idx}: len(tokens)={len(tokens)} but len(labels)={len(labels)}"
                )
            if not tokens:
                continue
            doc_key = self.record_doc_key(rec, "record", rec_idx)
            word_label_ids = self.normalize_record_labels(labels)
            for chunk_idx, (start, end) in enumerate(self.iter_word_windows(len(tokens))):
                chunk_tokens = [str(t) for t in tokens[start:end]]
                chunk_label_ids = word_label_ids[start:end]
                if self.cfg.drop_chunks_with_no_trainable_tokens and all(
                    (x == self.cfg.ignore_label for x in chunk_label_ids)
                ):
                    continue
                examples.append(
                    {
                        "dataset_name": dataset_name,
                        "strategy": strategy,
                        "doc_key": doc_key,
                        "record_index": rec_idx,
                        "chunk_index": chunk_idx,
                        "source_example_id": f"{doc_key}::{rec_idx}::{chunk_idx}",
                        "word_start": start,
                        "word_end": end,
                        "tokens": chunk_tokens,
                        "word_label_ids": chunk_label_ids,
                    }
                )
        return examples

    def build_examples(self) -> pd.DataFrame:
        self.state.examples_by_dataset_strategy = {
            dataset_name: {
                strategy: self.make_base_examples_for_strategy(dataset_name, strategy, records)
                for strategy, records in records_by_strategy.items()
            }
            for dataset_name, records_by_strategy in self.state.records_by_dataset_strategy.items()
        }
        chunk_summary_rows = []
        for dataset_name, examples_by_strategy in self.state.examples_by_dataset_strategy.items():
            for strategy, examples in examples_by_strategy.items():
                chunk_summary_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "strategy": strategy,
                        "n_base_word_chunks": len(examples),
                        "n_docs": len({ex["doc_key"] for ex in examples}),
                        "mean_base_chunk_words": (
                            float(np.mean([len(ex["tokens"]) for ex in examples]))
                            if examples
                            else 0.0
                        ),
                        "max_base_chunk_words": max(
                            [len(ex["tokens"]) for ex in examples], default=0
                        ),
                    }
                )
        if not all(
            (
                len(examples) > 0
                for examples_by_strategy in self.state.examples_by_dataset_strategy.values()
                for examples in examples_by_strategy.values()
            )
        ):
            raise AssertionError("At least one dataset/strategy has no examples.")
        return pd.DataFrame(chunk_summary_rows).sort_values(["dataset_name", "strategy"])

    def build_tokenizer(self, model_spec: ModelSpec):
        tokenizer_name = model_spec.tokenizer_name_or_path or model_spec.model_name_or_path
        tokenizer_kwargs = dict(model_spec.tokenizer_kwargs)
        tokenizer_kwargs.setdefault("use_fast", True)
        with self.quiet_section(self.cfg.quiet_training and self.cfg.suppress_model_load_warnings):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        if not tokenizer.is_fast:
            raise ValueError(
                f"Tokenizer for {model_spec.name} is not fast. Fast tokenizers are required because this pipeline uses word_ids() and overflow mappings."
            )
        # Added vocabulary items are atomic but remain ordinary input words, so
        # fast-tokenizer word_ids() keeps their word alignment for ignored labels.
        tokenizer.add_tokens(list(ALL_LAYOUT_TOKENS), special_tokens=False)
        for token in ALL_LAYOUT_TOKENS:
            token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(token_ids) != 1 or token_ids[0] == getattr(tokenizer, "unk_token_id", None):
                raise AssertionError(
                    f"Layout marker {token!r} is not one atomic non-UNK token for {model_spec.name}: {token_ids}."
                )
        return tokenizer

    def effective_max_length(self, tokenizer) -> int:
        tokenizer_limit = getattr(tokenizer, "model_max_length", self.cfg.max_length)
        if tokenizer_limit is None or tokenizer_limit > 1000000:
            return self.cfg.max_length
        return min(self.cfg.max_length, int(tokenizer_limit))

    def make_tokenize_and_align_labels_fn(self, tokenizer):
        max_length = self.effective_max_length(tokenizer)
        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        usable_length = max_length - special_tokens
        if usable_length <= 0:
            raise ValueError(
                f"max_length={max_length} leaves no room after {special_tokens} special tokens."
            )
        if self.cfg.tokenizer_stride >= usable_length:
            raise ValueError(
                f"tokenizer_stride={self.cfg.tokenizer_stride} must be smaller than the usable sequence length {usable_length}."
            )

        def tokenize_and_align_labels(batch):
            tokenized = tokenizer(
                batch["tokens"],
                is_split_into_words=True,
                truncation=False,
                max_length=max_length,
                return_offsets_mapping=True,
                padding="max_length",
            )
            aligned_labels: list[list[int]] = []
            aligned_bboxes: list[list[list[int]]] = []
            aligned_bbox_masks: list[list[int]] = []
            aligned_page_ids: list[list[int]] = []
            metric_source_rows: list[list[int]] = []
            metric_gold_rows: list[list[int]] = []
            metadata_rows: list[dict[str, Any]] = []
            keep_indices: list[int] = []

            for output_index in range(len(batch["tokens"])):
                word_ids = tokenized.word_ids(batch_index=output_index)
                word_label_ids = batch["word_label_ids"][output_index]
                word_sources = batch["word_source_token_indices"][output_index]
                word_metric_gold = batch["word_metric_gold_label_ids"][output_index]
                word_bboxes = batch["word_bboxes"][output_index]
                word_bbox_masks = batch["word_bbox_masks"][output_index]
                word_pages = batch["word_page_ids"][output_index]

                label_ids: list[int] = []
                bbox_row: list[list[int]] = []
                bbox_mask_row: list[int] = []
                page_row: list[int] = []
                metric_source_row: list[int] = []
                metric_gold_row: list[int] = []
                seen_words: set[int] = set()
                previous_word_idx = None

                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(self.cfg.ignore_label)
                        bbox_row.append([0, 0, 0, 0])
                        bbox_mask_row.append(0)
                        page_row.append(-1)
                        metric_source_row.append(-1)
                        metric_gold_row.append(self.cfg.ignore_label)
                        previous_word_idx = None
                        continue

                    word_idx = int(word_idx)
                    is_first_subtoken = word_idx != previous_word_idx
                    seen_words.add(word_idx)
                    bbox_row.append(list(word_bboxes[word_idx]))
                    bbox_mask_row.append(int(word_bbox_masks[word_idx]))
                    page_row.append(int(word_pages[word_idx]))
                    if is_first_subtoken:
                        label_ids.append(int(word_label_ids[word_idx]))
                        source_index = int(word_sources[word_idx])
                        gold_id = int(word_metric_gold[word_idx])
                        metric_source_row.append(source_index if source_index >= 0 else -1)
                        metric_gold_row.append(gold_id if source_index >= 0 else self.cfg.ignore_label)
                    else:
                        # One supervised/evaluated position per OCR/layout word.
                        label_ids.append(self.cfg.ignore_label)
                        metric_source_row.append(-1)
                        metric_gold_row.append(self.cfg.ignore_label)
                    previous_word_idx = word_idx

                if len(label_ids) != len(tokenized["input_ids"][output_index]):
                    raise AssertionError("Token and aligned feature lengths differ.")

                should_keep = not self.cfg.drop_chunks_with_no_trainable_tokens or any(
                    label != self.cfg.ignore_label for label in label_ids
                )
                if not should_keep:
                    continue
                keep_indices.append(output_index)
                aligned_labels.append(label_ids)
                aligned_bboxes.append(bbox_row)
                aligned_bbox_masks.append(bbox_mask_row)
                aligned_page_ids.append(page_row)
                metric_source_rows.append(metric_source_row)
                metric_gold_rows.append(metric_gold_row)
                metadata_rows.append(
                    {
                        "dataset_name": batch["dataset_name"][output_index],
                        "strategy": batch["strategy"][output_index],
                        "doc_key": batch["doc_key"][output_index],
                        "record_index": batch["record_index"][output_index],
                        "chunk_index": batch["chunk_index"][output_index],
                        "source_example_id": batch["source_example_id"][output_index],
                        "word_start": batch["word_start"][output_index],
                        "word_end": batch["word_end"][output_index],
                        "overflow_index": 0,
                        "n_source_words": len(batch["tokens"][output_index]),
                        "n_subtokens": int(sum(tokenized["attention_mask"][output_index])),
                        "metric_doc_key": batch["metric_doc_key"][output_index],
                        "metric_record_index": batch["metric_record_index"][output_index],
                        "metric_original_token_count": batch["metric_original_token_count"][output_index],
                    }
                )

            result = {
                key: [values[index] for index in keep_indices]
                for key, values in tokenized.items()
                if key != "offset_mapping"
            }
            result["labels"] = aligned_labels
            result["bbox"] = aligned_bboxes
            result["bbox_mask"] = aligned_bbox_masks
            result["page_ids"] = aligned_page_ids
            result["metric_source_indices"] = metric_source_rows
            result["metric_gold_labels"] = metric_gold_rows
            for key in metadata_rows[0].keys() if metadata_rows else []:
                result[key] = [row[key] for row in metadata_rows]
            return result

        return (tokenize_and_align_labels, max_length)

    def tokenize_dataset_strategy_once(
        self, dataset_name: str, strategy: str, model_spec: ModelSpec, tokenizer
    ) -> TokenizedCorpus:
        examples = self._ensure_fair_examples(dataset_name, model_spec, tokenizer)[strategy]
        base_dataset = Dataset.from_list(examples)
        tokenize_fn, effective_max_length = self.make_tokenize_and_align_labels_fn(tokenizer)
        map_kwargs: dict[str, Any] = {
            "batched": True,
            "batch_size": self.cfg.tokenization_batch_size,
            "remove_columns": base_dataset.column_names,
            "desc": (
                None
                if self.cfg.quiet_training
                else f"Tokenizing {dataset_name}/{strategy}/{model_spec.name}"
            ),
        }
        if self.cfg.tokenization_num_proc is not None:
            map_kwargs["num_proc"] = self.cfg.tokenization_num_proc
        with self.quiet_section(self.cfg.quiet_training):
            tokenized_dataset = base_dataset.map(tokenize_fn, **map_kwargs)
        if len(tokenized_dataset) == 0:
            raise AssertionError(
                f"{dataset_name}/{strategy}/{model_spec.name}: tokenization produced no chunks."
            )
        indices_by_doc_key: dict[str, list[int]] = defaultdict(list)
        for index, doc_key in enumerate(tokenized_dataset["doc_key"]):
            indices_by_doc_key[str(doc_key)].append(index)
        lengths = [int(x) for x in tokenized_dataset["n_subtokens"]]
        summary = {
            "dataset_name": dataset_name,
            "strategy": strategy,
            "model_name": model_spec.name,
            "model_name_or_path": model_spec.model_name_or_path,
            "effective_max_length": effective_max_length,
            "tokenizer_stride": self.cfg.tokenizer_stride,
            "n_base_word_chunks": len(examples),
            "n_tokenized_chunks": len(tokenized_dataset),
            "n_docs": len(indices_by_doc_key),
            "mean_subtokens": float(np.mean(lengths)) if lengths else 0.0,
            "max_subtokens": max(lengths, default=0),
            "n_overflow_chunks": 0,
            "shared_source_windows": True,
            "padded_subtokens_per_chunk": effective_max_length,
            "total_padded_subtokens": len(tokenized_dataset) * effective_max_length,
        }
        self.state.tokenization_summary_rows.append(summary)
        pd.DataFrame(self.state.tokenization_summary_rows).to_csv(
            self.cfg.results_root / "tokenization_summary.csv", index=False
        )
        self.qprint(
            f"Tokenized once: {dataset_name}/{strategy}/{model_spec.name}: {len(examples):,} base chunks -> {len(tokenized_dataset):,} model chunks."
        )
        return TokenizedCorpus(
            dataset=tokenized_dataset, indices_by_doc_key=dict(indices_by_doc_key), summary=summary
        )

    def make_fold_datasets_from_tokenized(
        self, dataset_name: str, strategy: str, fold_index: int, corpus: TokenizedCorpus
    ) -> dict[str, Dataset]:
        assignment = self.state.cv_split_assignments_by_dataset[dataset_name][fold_index]
        split_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        for doc_key, indices in corpus.indices_by_doc_key.items():
            if doc_key not in assignment:
                raise KeyError(
                    f"{dataset_name}/{strategy}: document {doc_key!r} is missing from the fold-{fold_index} assignment."
                )
            split_indices[assignment[doc_key]].extend(indices)
        metadata_columns = {
            "dataset_name",
            "strategy",
            "doc_key",
            "record_index",
            "chunk_index",
            "source_example_id",
            "word_start",
            "word_end",
            "overflow_index",
            "n_source_words",
            "n_subtokens",
        }
        datasets: dict[str, Dataset] = {}
        for split_name, indices in split_indices.items():
            if not indices:
                raise AssertionError(
                    f"{dataset_name}/{strategy} fold {fold_index}: no {split_name} chunks."
                )
            split_dataset = corpus.dataset.select(sorted(indices))
            columns_to_remove = [
                column for column in split_dataset.column_names if column in metadata_columns
            ]
            if columns_to_remove:
                split_dataset = split_dataset.remove_columns(columns_to_remove)
            datasets[split_name] = split_dataset
        return datasets

    @staticmethod
    def safe_name(value: Any) -> str:
        text = str(value)
        return "".join((ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text))
