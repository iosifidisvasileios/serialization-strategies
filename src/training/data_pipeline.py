from __future__ import annotations
import json
import os
import random
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
from experiment_config import ExperimentConfig, ModelSpec


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
            self.state.selected_strategy_files_by_dataset[dataset_name] = selected_strategy_files
            self.state.records_by_dataset_strategy[dataset_name] = {}
            for strategy, path in selected_strategy_files.items():
                records = self.read_jsonl(path)
                if self.cfg.sampling is not None:
                    rng = random.Random(self.cfg.split_seed)
                    records = records.copy()
                    rng.shuffle(records)
                    records = records[: self.cfg.sampling]
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
                    for label in rec.get("labels", []):
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
                doc_key = self.record_doc_key(rec, strategy, rec_idx)
                for label in rec.get("labels", []):
                    normalized = self.normalize_label_name(label)
                    if normalized is None or normalized == "O":
                        continue
                    entity_counts_by_doc[doc_key][normalized] += 1
        all_doc_keys = sorted(
            {
                self.record_doc_key(rec, strategy, rec_idx)
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
                        {self.record_doc_key(rec, strategy, i) for i, rec in enumerate(records)}
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
            doc_key = self.record_doc_key(rec, strategy, rec_idx)
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
                truncation=True,
                max_length=max_length,
                stride=self.cfg.tokenizer_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=False,
            )
            sample_mapping = [int(x) for x in tokenized["overflow_to_sample_mapping"]]
            offsets = tokenized["offset_mapping"]
            aligned_labels: list[list[int]] = []
            seen_word_ids: dict[int, set[int]] = defaultdict(set)
            overflow_counts: dict[int, int] = defaultdict(int)
            metadata_rows: list[dict[str, Any]] = []
            keep_indices: list[int] = []
            for output_index, source_index in enumerate(sample_mapping):
                word_label_ids = batch["word_label_ids"][source_index]
                word_ids = tokenized.word_ids(batch_index=output_index)
                token_offsets = offsets[output_index]
                label_ids: list[int] = []
                for word_idx, token_offset in zip(word_ids, token_offsets):
                    if word_idx is None:
                        label_ids.append(self.cfg.ignore_label)
                        continue
                    seen_word_ids[source_index].add(int(word_idx))
                    original_label_id = word_label_ids[word_idx]
                    if original_label_id == self.cfg.ignore_label:
                        label_ids.append(self.cfg.ignore_label)
                        continue
                    is_first_subtoken = int(token_offset[0]) == 0
                    if is_first_subtoken:
                        label_ids.append(original_label_id)
                    else:
                        original_label = self.state.id2label[original_label_id]
                        if original_label.startswith("B-"):
                            inside_label = "I-" + original_label[2:]
                            label_ids.append(
                                self.state.label2id.get(inside_label, original_label_id)
                            )
                        else:
                            label_ids.append(original_label_id)
                if len(label_ids) != len(tokenized["input_ids"][output_index]):
                    raise AssertionError("Token and label lengths differ after alignment.")
                overflow_index = overflow_counts[source_index]
                overflow_counts[source_index] += 1
                should_keep = not self.cfg.drop_chunks_with_no_trainable_tokens or any(
                    (label != self.cfg.ignore_label for label in label_ids)
                )
                if not should_keep:
                    continue
                keep_indices.append(output_index)
                aligned_labels.append(label_ids)
                metadata_rows.append(
                    {
                        "dataset_name": batch["dataset_name"][source_index],
                        "strategy": batch["strategy"][source_index],
                        "doc_key": batch["doc_key"][source_index],
                        "record_index": batch["record_index"][source_index],
                        "chunk_index": batch["chunk_index"][source_index],
                        "source_example_id": batch["source_example_id"][source_index],
                        "word_start": batch["word_start"][source_index],
                        "word_end": batch["word_end"][source_index],
                        "overflow_index": overflow_index,
                        "n_source_words": len(batch["tokens"][source_index]),
                        "n_subtokens": len(tokenized["input_ids"][output_index]),
                    }
                )

            result = {
                key: [values[i] for i in keep_indices]
                for key, values in tokenized.items()
                if key not in {"overflow_to_sample_mapping", "offset_mapping"}
            }
            result["labels"] = aligned_labels
            for key in metadata_rows[0].keys() if metadata_rows else []:
                result[key] = [row[key] for row in metadata_rows]
            return result

        return (tokenize_and_align_labels, max_length)

    def tokenize_dataset_strategy_once(
        self, dataset_name: str, strategy: str, model_spec: ModelSpec, tokenizer
    ) -> TokenizedCorpus:
        examples = self.state.examples_by_dataset_strategy[dataset_name][strategy]
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
            "n_overflow_chunks": max(0, len(tokenized_dataset) - len(examples)),
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
