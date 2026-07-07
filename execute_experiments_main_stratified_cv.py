from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
import logging
import os
import random
import shutil
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as transformers_logging

try:
    from datasets import disable_progress_bar as disable_datasets_progress_bar
    from datasets import enable_progress_bar as enable_datasets_progress_bar
except ImportError:
    disable_datasets_progress_bar = None
    enable_datasets_progress_bar = None

try:
    from datasets.utils import logging as datasets_logging
except ImportError:
    datasets_logging = None

try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    init_from_pretrained: bool = True
    revision: Optional[str] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    project_root: Path = field(default_factory=Path.cwd)
    processed_datasets_root: Optional[Path] = None
    runs_root: Optional[Path] = None
    results_root: Optional[Path] = None

    split_seed: int = 42
    n_folds: int = 5
    validation_fraction_of_train: float = 0.10
    sampling: Optional[int] = None

    datasets_to_run: str | list[str] = "all"
    strategies_to_run: str | list[str] = field(default_factory=lambda: ["column_aware"])
    model_registry: list[ModelSpec] = field(default_factory=lambda: [ModelSpec("bert-mlsm", "SzegedAI/bert-medium-mlsm")])

    max_length: int = 512
    word_window_size: int = 384
    word_window_stride: int = 0
    drop_chunks_with_no_trainable_tokens: bool = True

    num_train_epochs: float = 20
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    optimizer_name: str = "adamw_torch"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0

    best_model_metric: str = "macro_f1"
    best_model_greater_is_better: bool = True
    early_stopping_patience: Optional[int] = 3
    early_stopping_threshold: float = 0.0

    mixed_precision: str = "auto"
    force_trainable_params_fp32: bool = True
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = False

    quiet_training: bool = True
    suppress_model_load_warnings: bool = True
    suppress_train_stdout_stderr: bool = True
    show_run_progress: bool = False
    transformers_verbosity: str = "error"
    datasets_verbosity: str = "error"
    trainer_logging_strategy: str = "no"

    overwrite_output_dir: bool = True
    dry_run: bool = False

    mlflow_experiment_name: str = field(default_factory=lambda: f"serialization_strategies_{int(time.time())}")
    mlflow_db_path: Optional[Path] = None
    mlflow_artifact_root: Optional[Path] = None
    mlflow_tracking_uri: Optional[str] = None

    ignore_label: int = -100

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.processed_datasets_root = Path(self.processed_datasets_root or self.project_root / "data" / "processed").resolve()
        self.runs_root = Path(self.runs_root or self.project_root / "runs").resolve()
        # self.results_root = Path(self.results_root or self.project_root / "results" + "_".join(self.datasets_to_run)).resolve()
        dataset_suffix = (
            "all"
            if self.datasets_to_run == "all"
            else "_".join(str(x) for x in self.datasets_to_run)
        )

        self.results_root = Path(
            self.results_root or self.project_root / f"results_{dataset_suffix}"
        ).resolve()


        self.mlflow_db_path = Path(self.mlflow_db_path or self.project_root / "mlflow.db").resolve()
        self.mlflow_artifact_root = Path(self.mlflow_artifact_root or self.project_root / "mlartifacts").resolve()
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{self.mlflow_db_path.as_posix()}")

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data["model_registry"] = [asdict(model) for model in self.model_registry]
        return data


@dataclass
class ExperimentState:
    records_by_dataset_strategy: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    selected_strategy_files_by_dataset: dict[str, dict[str, Path]] = field(default_factory=dict)
    selected_datasets: list[str] = field(default_factory=list)
    selected_strategies_by_dataset: dict[str, list[str]] = field(default_factory=dict)

    label_list: list[str] = field(default_factory=list)
    label2id: dict[str, int] = field(default_factory=dict)
    id2label: dict[int, str] = field(default_factory=dict)
    non_o_label_ids: list[int] = field(default_factory=list)

    cv_split_assignments_by_dataset: dict[str, dict[int, dict[str, str]]] = field(default_factory=dict)
    doc_strata_by_dataset: dict[str, dict[str, str]] = field(default_factory=dict)
    cv_assignment_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)
    split_plan_paths: dict[str, Path] = field(default_factory=dict)
    examples_by_dataset_strategy: dict[str, dict[str, list[dict]]] = field(default_factory=dict)

    fold_results: list[dict[str, Any]] = field(default_factory=list)
    per_label_report_paths: list[Path] = field(default_factory=list)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.state = ExperimentState()

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

    def configure_quiet_output(self) -> None:
        if not self.cfg.quiet_training:
            transformers_logging.set_verbosity_info()
            if datasets_logging is not None:
                datasets_logging.set_verbosity_info()
            if enable_datasets_progress_bar is not None:
                enable_datasets_progress_bar()
            return

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
        warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
        warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")

        logging.getLogger().setLevel(logging.ERROR)
        for logger_name in [
            "transformers",
            "transformers.modeling_utils",
            "transformers.configuration_utils",
            "transformers.tokenization_utils_base",
            "datasets",
            "evaluate",
            "accelerate",
            "mlflow",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        if self.cfg.transformers_verbosity == "error":
            transformers_logging.set_verbosity_error()
        elif self.cfg.transformers_verbosity == "warning":
            transformers_logging.set_verbosity_warning()
        else:
            transformers_logging.set_verbosity_info()

        if datasets_logging is not None:
            if self.cfg.datasets_verbosity == "error":
                datasets_logging.set_verbosity_error()
            elif self.cfg.datasets_verbosity == "warning":
                datasets_logging.set_verbosity_warning()
            else:
                datasets_logging.set_verbosity_info()

        if disable_datasets_progress_bar is not None:
            disable_datasets_progress_bar()

    def setup_paths(self) -> None:
        self.cfg.runs_root.mkdir(parents=True, exist_ok=True)
        self.cfg.results_root.mkdir(parents=True, exist_ok=True)
        self.cfg.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)

    def setup_mlflow(self) -> int:
        while mlflow.active_run() is not None:
            mlflow.end_run()

        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        client = MlflowClient()
        existing = client.get_experiment_by_name(self.cfg.mlflow_experiment_name)
        if existing is None:
            experiment_id = client.create_experiment(
                name=self.cfg.mlflow_experiment_name,
                artifact_location=self.cfg.mlflow_artifact_root.as_uri(),
            )
        else:
            experiment_id = existing.experiment_id
        mlflow.set_experiment(self.cfg.mlflow_experiment_name)
        return experiment_id

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
            raise FileNotFoundError(f"Processed datasets root does not exist: {self.cfg.processed_datasets_root}")

        dataset_roots: dict[str, Path] = {}
        for candidate_root in sorted(self.cfg.processed_datasets_root.iterdir()):
            if candidate_root.is_dir() and self.discover_strategy_files(candidate_root):
                dataset_roots.setdefault(candidate_root.name, candidate_root)

        if self.cfg.datasets_to_run != "all":
            requested = set(self.cfg.datasets_to_run)
            missing = sorted(requested.difference(dataset_roots))
            if missing:
                raise AssertionError(f"Requested datasets not found: {missing}")
            dataset_roots = {name: root for name, root in dataset_roots.items() if name in requested}

        if not dataset_roots:
            raise AssertionError("No processed datasets found. Expected folders under data/processed containing */all.jsonl.")
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
                read_summary_rows.append({
                    "dataset_name": dataset_name,
                    "strategy": strategy,
                    "n_records": len(records),
                    "path": str(path),
                })

        if not self.state.records_by_dataset_strategy:
            raise AssertionError("No records loaded after dataset/strategy filtering.")

        self.state.selected_datasets = sorted(self.state.records_by_dataset_strategy.keys())
        self.state.selected_strategies_by_dataset = {
            dataset_name: sorted(records_by_strategy.keys())
            for dataset_name, records_by_strategy in self.state.records_by_dataset_strategy.items()
        }
        return pd.DataFrame(read_summary_rows)

    def is_real_label(self, x: Any) -> bool:
        return isinstance(x, str) and x != "" and x != str(self.cfg.ignore_label)

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
        return pd.DataFrame({"label_id": list(range(len(self.state.label_list))), "label": self.state.label_list})

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

        all_doc_keys = sorted({
            self.record_doc_key(rec, strategy, rec_idx)
            for strategy, records in records_by_strategy.items()
            for rec_idx, rec in enumerate(records)
        })

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
        self,
        trainval_keys: list[str],
        key_to_stratum: dict[str, str],
        seed: int,
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
                "validation_fraction_of_train must be between 0 and 1, "
                f"got {self.cfg.validation_fraction_of_train}."
            )
        if len(trainval_keys) < 2:
            raise AssertionError(
                "Each outer training partition must contain at least two documents "
                "to create a validation split."
            )

        y = [key_to_stratum[k] for k in trainval_keys]
        n_validation = int(round(len(trainval_keys) * self.cfg.validation_fraction_of_train))
        n_validation = max(1, min(n_validation, len(trainval_keys) - 1))

        n_classes = len(set(y))
        can_stratify = (
            self.can_use_stratification(y, min_count=2)
            and n_validation >= n_classes
            and (len(trainval_keys) - n_validation) >= n_classes
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

        return set(train_keys), set(validation_keys), inner_splitter_name

    def build_cv_split_assignments(self, records_by_strategy: dict[str, list[dict]], n_folds: int, seed: int):
        key_to_stratum = self.build_doc_strata(records_by_strategy)
        doc_keys = np.array(sorted(key_to_stratum.keys()))

        if len(doc_keys) < n_folds:
            raise ValueError(f"Not enough documents for {n_folds}-fold CV: found {len(doc_keys)} documents.")

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
                trainval_keys=trainval_keys,
                key_to_stratum=key_to_stratum,
                seed=seed + fold_index,
            )

            assignment = {}
            assignment.update({k: "train" for k in train_keys})
            assignment.update({k: "validation" for k in validation_keys})
            assignment.update({k: "test" for k in test_keys})

            assert set(assignment) == set(doc_keys.tolist())
            assert not (validation_keys & test_keys)
            assert not (train_keys & validation_keys)
            assert not (train_keys & test_keys)

            assignments_by_fold[fold_index] = assignment
            for doc_key, split in assignment.items():
                rows.append({
                    "fold": fold_index,
                    "doc_key": doc_key,
                    "split": split,
                    "stratum": key_to_stratum[doc_key],
                    "outer_splitter": outer_splitter_name,
                    "inner_validation_splitter": inner_splitter_name,
                    "validation_fraction_of_train": self.cfg.validation_fraction_of_train,
                })

        assignment_df = pd.DataFrame(rows).sort_values(["fold", "split", "doc_key"])
        return assignments_by_fold, key_to_stratum, assignment_df

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

            split_plan_path = self.cfg.results_root / f"{self.safe_name(dataset_name)}_cv_doc_split_assignments.csv"
            assignment_df.to_csv(split_plan_path, index=False)
            self.state.split_plan_paths[dataset_name] = split_plan_path
            print("Saved CV split plan to:", split_plan_path)
            print("Outer splitter used:", assignment_df["outer_splitter"].iloc[0])
            print("Inner validation splitter(s):", sorted(assignment_df["inner_validation_splitter"].unique().tolist()))

        strategy_split_rows = []
        for dataset_name, assignments_by_fold in self.state.cv_split_assignments_by_dataset.items():
            records_by_strategy = self.state.records_by_dataset_strategy[dataset_name]
            for fold_index, assignment in assignments_by_fold.items():
                for strategy, records in records_by_strategy.items():
                    strategy_doc_keys = sorted({self.record_doc_key(rec, strategy, i) for i, rec in enumerate(records)})
                    counts = Counter(assignment[k] for k in strategy_doc_keys)
                    n_docs = len(strategy_doc_keys)
                    strategy_split_rows.append({
                        "dataset_name": dataset_name,
                        "fold": fold_index,
                        "strategy": strategy,
                        "n_docs": n_docs,
                        "train_docs": counts.get("train", 0),
                        "validation_docs": counts.get("validation", 0),
                        "test_docs": counts.get("test", 0),
                        "observed_train_fraction": counts.get("train", 0) / n_docs if n_docs else 0,
                        "observed_validation_fraction": counts.get("validation", 0) / n_docs if n_docs else 0,
                        "observed_test_fraction": counts.get("test", 0) / n_docs if n_docs else 0,
                    })
        return pd.DataFrame(strategy_split_rows)

    def normalize_record_labels(self, raw_labels: list) -> list[int]:
        label_ids = []
        for label in raw_labels:
            if label == self.cfg.ignore_label or label == str(self.cfg.ignore_label) or label is None:
                label_ids.append(self.cfg.ignore_label)
            elif isinstance(label, str):
                label_ids.append(self.state.label2id[label])
            else:
                label_ids.append(self.cfg.ignore_label)
        return label_ids

    def iter_word_windows(self, n_tokens: int):
        assert self.cfg.word_window_size > 0
        assert self.cfg.word_window_stride >= 0
        step = self.cfg.word_window_size - self.cfg.word_window_stride
        assert step > 0, "WORD_WINDOW_STRIDE must be smaller than WORD_WINDOW_SIZE."

        start = 0
        while start < n_tokens:
            end = min(n_tokens, start + self.cfg.word_window_size)
            yield start, end
            if end >= n_tokens:
                break
            start += step

    def make_base_examples_for_strategy(self, dataset_name: str, strategy: str, records: list[dict]) -> list[dict]:
        examples = []
        for rec_idx, rec in enumerate(records):
            tokens = rec.get("tokens", [])
            labels = rec.get("labels", [])
            if len(tokens) != len(labels):
                raise ValueError(
                    f"{dataset_name}/{strategy} record {rec_idx}: len(tokens)={len(tokens)} but len(labels)={len(labels)}"
                )

            doc_key = self.record_doc_key(rec, strategy, rec_idx)
            word_label_ids = self.normalize_record_labels(labels)

            for chunk_idx, (start, end) in enumerate(self.iter_word_windows(len(tokens))):
                chunk_tokens = [str(t) for t in tokens[start:end]]
                chunk_label_ids = word_label_ids[start:end]
                if self.cfg.drop_chunks_with_no_trainable_tokens and all(x == self.cfg.ignore_label for x in chunk_label_ids):
                    continue
                examples.append({
                    "dataset_name": dataset_name,
                    "strategy": strategy,
                    "doc_key": doc_key,
                    "record_index": rec_idx,
                    "chunk_index": chunk_idx,
                    "word_start": start,
                    "word_end": end,
                    "tokens": chunk_tokens,
                    "word_label_ids": chunk_label_ids,
                })
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
                chunk_summary_rows.append({
                    "dataset_name": dataset_name,
                    "strategy": strategy,
                    "n_chunks": len(examples),
                    "n_docs": len({ex["doc_key"] for ex in examples}),
                    "mean_chunk_words": float(np.mean([len(ex["tokens"]) for ex in examples])) if examples else 0,
                    "max_chunk_words": max([len(ex["tokens"]) for ex in examples], default=0),
                })

        if not all(
            len(examples) > 0
            for examples_by_strategy in self.state.examples_by_dataset_strategy.values()
            for examples in examples_by_strategy.values()
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
                f"Tokenizer for {model_spec.name} is not fast. Fast tokenizers are required because this pipeline uses tokenizer.word_ids()."
            )
        return tokenizer

    def effective_max_length(self, tokenizer) -> int:
        tokenizer_limit = getattr(tokenizer, "model_max_length", self.cfg.max_length)
        if tokenizer_limit is None or tokenizer_limit > 1_000_000:
            return self.cfg.max_length
        return min(self.cfg.max_length, int(tokenizer_limit))

    def make_tokenize_and_align_labels_fn(self, tokenizer):
        max_length = self.effective_max_length(tokenizer)

        def tokenize_and_align_labels(batch):
            tokenized = tokenizer(
                batch["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
            )
            aligned_labels = []

            for i, word_label_ids in enumerate(batch["word_label_ids"]):
                word_ids = tokenized.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(self.cfg.ignore_label)
                    elif word_idx != previous_word_idx:
                        label_ids.append(word_label_ids[word_idx])
                    else:
                        original_label_id = word_label_ids[word_idx]
                        if original_label_id == self.cfg.ignore_label:
                            label_ids.append(self.cfg.ignore_label)
                        else:
                            original_label = self.state.id2label[original_label_id]
                            if original_label.startswith("B-"):
                                inside_label = "I-" + original_label[2:]
                                label_ids.append(self.state.label2id.get(inside_label, original_label_id))
                            else:
                                label_ids.append(original_label_id)
                    previous_word_idx = word_idx
                aligned_labels.append(label_ids)

            tokenized["labels"] = aligned_labels
            return tokenized

        return tokenize_and_align_labels

    def examples_for_dataset_strategy_fold(self, dataset_name: str, strategy: str, fold_index: int) -> list[dict]:
        assignment = self.state.cv_split_assignments_by_dataset[dataset_name][fold_index]
        fold_examples = []
        for ex in self.state.examples_by_dataset_strategy[dataset_name][strategy]:
            split = assignment[ex["doc_key"]]
            fold_examples.append({**ex, "fold": fold_index, "experiment_split": split})
        return fold_examples

    def make_hf_datasets(self, dataset_name: str, strategy: str, fold_index: int, tokenizer) -> dict[str, Dataset]:
        examples = self.examples_for_dataset_strategy_fold(dataset_name, strategy, fold_index)
        split_to_examples = {
            "train": [ex for ex in examples if ex["experiment_split"] == "train"],
            "validation": [ex for ex in examples if ex["experiment_split"] == "validation"],
            "test": [ex for ex in examples if ex["experiment_split"] == "test"],
        }

        datasets = {}
        tokenize_and_align_labels = self.make_tokenize_and_align_labels_fn(tokenizer)
        for split_name, split_examples in split_to_examples.items():
            assert split_examples, f"{dataset_name}/{strategy} fold {fold_index}: no {split_name} examples."
            ds = Dataset.from_list(split_examples)
            with self.quiet_section(self.cfg.quiet_training):
                tokenized_ds = ds.map(
                    tokenize_and_align_labels,
                    batched=True,
                    remove_columns=ds.column_names,
                    desc=None if self.cfg.quiet_training else f"Tokenizing {dataset_name}/{strategy}/fold_{fold_index}/{split_name}",
                )
            datasets[split_name] = tokenized_ds
        return datasets

    def flatten_predictions_and_labels(self, logits_or_predictions, labels):
        if logits_or_predictions.ndim == 3:
            predictions = np.argmax(logits_or_predictions, axis=-1)
        else:
            predictions = logits_or_predictions

        y_true = []
        y_pred = []
        for pred_row, label_row in zip(predictions, labels):
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == self.cfg.ignore_label:
                    continue
                y_true.append(int(label_id))
                y_pred.append(int(pred_id))
        return np.array(y_true), np.array(y_pred)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        y_true, y_pred = self.flatten_predictions_and_labels(logits, labels)
        if len(y_true) == 0:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "non_o_micro_f1": 0.0,
                "n_eval_tokens": 0,
            }

        accuracy = accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        if self.state.non_o_label_ids:
            non_o_p, non_o_r, non_o_f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=self.state.non_o_label_ids,
                average="micro",
                zero_division=0,
            )
        else:
            non_o_p = non_o_r = non_o_f1 = 0.0

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "non_o_micro_precision": float(non_o_p),
            "non_o_micro_recall": float(non_o_r),
            "non_o_micro_f1": float(non_o_f1),
            "n_eval_tokens": int(len(y_true)),
        }

    def per_label_report_dataframe(self, pred_output) -> pd.DataFrame:
        y_true, y_pred = self.flatten_predictions_and_labels(pred_output.predictions, pred_output.label_ids)
        labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        target_names = [self.state.id2label[i] for i in labels_present]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels_present,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        rows = []
        for label_name, values in report.items():
            if isinstance(values, dict):
                row = {"label": label_name}
                row.update(values)
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def safe_name(value: Any) -> str:
        text = str(value)
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)

    def select_mixed_precision_flags(self) -> dict[str, bool]:
        if not torch.cuda.is_available() or self.cfg.mixed_precision == "fp32":
            return {"fp16": False, "bf16": False}

        requested = str(self.cfg.mixed_precision).lower()
        if requested == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("mixed_precision='bf16' was requested, but this CUDA device does not support BF16.")
            return {"fp16": False, "bf16": True}
        if requested == "fp16":
            return {"fp16": True, "bf16": False}
        if requested == "auto":
            if torch.cuda.is_bf16_supported():
                return {"fp16": False, "bf16": True}
            return {"fp16": True, "bf16": False}
        raise ValueError("mixed_precision must be one of: 'auto', 'bf16', 'fp16', 'fp32'.")

    @staticmethod
    def model_initialization_mode(model_spec: ModelSpec) -> str:
        return "fine_tuning_from_pretrained" if model_spec.init_from_pretrained else "training_from_scratch"

    def force_trainable_parameters_fp32(self, model) -> None:
        if not self.cfg.force_trainable_params_fp32:
            return
        converted = 0
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16}:
                parameter.data = parameter.data.float()
                converted += parameter.numel()
        if converted:
            self.qprint(f"Converted {converted:,} trainable parameters to FP32 for AMP-compatible training.")

    @staticmethod
    def summarize_trainable_parameter_dtypes(model) -> dict[str, int]:
        counts: dict[str, int] = {}
        for parameter in model.parameters():
            if parameter.requires_grad:
                key = str(parameter.dtype)
                counts[key] = counts.get(key, 0) + parameter.numel()
        return counts

    def build_model(self, model_spec: ModelSpec):
        common_kwargs = {
            "num_labels": len(self.state.label_list),
            "id2label": self.state.id2label,
            "label2id": self.state.label2id,
            **dict(model_spec.model_kwargs),
        }
        if model_spec.revision is not None:
            common_kwargs["revision"] = model_spec.revision

        with self.quiet_section(self.cfg.quiet_training and self.cfg.suppress_model_load_warnings):
            if model_spec.init_from_pretrained:
                model = AutoModelForTokenClassification.from_pretrained(model_spec.model_name_or_path, **common_kwargs)
            else:
                config = AutoConfig.from_pretrained(model_spec.model_name_or_path, **common_kwargs)
                model = AutoModelForTokenClassification.from_config(config)

        if self.cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        self.force_trainable_parameters_fp32(model)
        self.qprint("Trainable parameter dtypes:", self.summarize_trainable_parameter_dtypes(model))
        return model

    @staticmethod
    def count_total_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def count_trainable_parameters(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def make_training_args(self, run_id: str, seed: int) -> TrainingArguments:
        output_dir = self.cfg.runs_root / run_id
        if self.cfg.overwrite_output_dir and output_dir.exists():
            shutil.rmtree(output_dir)

        precision_flags = self.select_mixed_precision_flags()
        candidate_kwargs = {
            "output_dir": str(output_dir),
            "learning_rate": self.cfg.learning_rate,
            "optim": self.cfg.optimizer_name,
            "per_device_train_batch_size": self.cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": self.cfg.per_device_eval_batch_size,
            "num_train_epochs": self.cfg.num_train_epochs,
            "weight_decay": self.cfg.weight_decay,
            "warmup_ratio": self.cfg.warmup_ratio,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "lr_scheduler_type": self.cfg.lr_scheduler_type,
            "max_grad_norm": self.cfg.max_grad_norm,
            "save_strategy": "epoch",
            "logging_strategy": self.cfg.trainer_logging_strategy,
            "logging_steps": 50,
            "logging_first_step": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": self.cfg.best_model_metric,
            "greater_is_better": self.cfg.best_model_greater_is_better,
            "save_total_limit": self.cfg.save_total_limit,
            "report_to": "none",
            "run_name": run_id,
            "seed": seed,
            "data_seed": seed,
            "fp16": precision_flags["fp16"],
            "bf16": precision_flags["bf16"],
            "dataloader_num_workers": self.cfg.dataloader_num_workers,
            "disable_tqdm": self.cfg.quiet_training,
            "log_level": self.cfg.transformers_verbosity,
            "log_level_replica": self.cfg.transformers_verbosity,
            "gradient_checkpointing": self.cfg.gradient_checkpointing,
        }

        sig = inspect.signature(TrainingArguments.__init__)
        params = set(sig.parameters)
        if "eval_strategy" in params:
            candidate_kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in params:
            candidate_kwargs["evaluation_strategy"] = "epoch"
        if "warmup_ratio" not in params and "warmup_steps" in params:
            candidate_kwargs["warmup_steps"] = 0

        filtered_kwargs = {key: value for key, value in candidate_kwargs.items() if key in params}
        ignored_kwargs = sorted(set(candidate_kwargs).difference(filtered_kwargs))
        if ignored_kwargs:
            self.qprint(f"{run_id}: ignored unsupported TrainingArguments kwargs: {ignored_kwargs}")
        return TrainingArguments(**filtered_kwargs)

    def build_trainer(self, model, training_args, datasets: dict[str, Dataset], tokenizer, data_collator):
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": datasets["train"],
            "eval_dataset": datasets["validation"],
            "data_collator": data_collator,
            "compute_metrics": self.compute_metrics,
        }

        callbacks = []
        if self.cfg.early_stopping_patience is not None:
            if EarlyStoppingCallback is None:
                self.qprint("EarlyStoppingCallback unavailable in this Transformers version; continuing without early stopping.")
            else:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=self.cfg.early_stopping_patience,
                        early_stopping_threshold=self.cfg.early_stopping_threshold,
                    )
                )
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer_sig = inspect.signature(Trainer.__init__)
        trainer_params = set(trainer_sig.parameters)
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)
        return self.remove_notebook_progress_callbacks(trainer)

    def remove_notebook_progress_callbacks(self, trainer: Trainer) -> Trainer:
        callbacks = list(getattr(trainer.callback_handler, "callbacks", []))
        kept_callbacks = [cb for cb in callbacks if cb.__class__.__name__ != "NotebookProgressCallback"]
        removed = len(callbacks) - len(kept_callbacks)
        if removed:
            self.qprint(f"Removed {removed} NotebookProgressCallback instance(s).")
        trainer.callback_handler.callbacks = kept_callbacks
        return trainer

    def run_trainer_quietly(self, fn, *args, **kwargs):
        with self.quiet_section(self.cfg.quiet_training and self.cfg.suppress_train_stdout_stderr):
            return fn(*args, **kwargs)

    @staticmethod
    def cleanup_after_fold() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def maybe_mlflow_run(self, run_name: str, nested: bool = False, tags: Optional[dict[str, Any]] = None):
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            if tags:
                mlflow.set_tags({k: str(v) for k, v in tags.items()})
            yield run

    @staticmethod
    def mlflow_param_value(value: Any) -> str:
        if isinstance(value, Path):
            text = str(value)
        elif isinstance(value, ModelSpec):
            text = json.dumps(asdict(value), default=str, sort_keys=True)
        elif isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, default=str, sort_keys=True)
        else:
            text = str(value)
        return text[:500]

    def log_params_to_mlflow(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            mlflow.log_param(str(key), self.mlflow_param_value(value))

    @staticmethod
    def log_metrics_to_mlflow(metrics: dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                numeric_metrics[f"{prefix}{key}"] = float(value)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)

    def log_trainer_history_to_mlflow(self, trainer: Trainer, prefix: str = "history_") -> None:
        for row in getattr(trainer.state, "log_history", []):
            step = row.get("step")
            metrics = {
                k: v
                for k, v in row.items()
                if k not in {"step"} and isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)
            }
            self.log_metrics_to_mlflow(metrics, prefix=prefix, step=int(step) if step is not None else None)

    @staticmethod
    def summarize_cv_results(fold_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if fold_df.empty:
            return pd.DataFrame()
        candidate_cols = [
            c
            for c in fold_df.columns
            if (
                c.startswith("test_")
                or c.startswith("val_")
                or c.startswith("train_")
                or c in {"best_validation_metric", "trainable_parameters", "total_parameters", "trainable_parameter_ratio"}
            )
        ]
        metric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(fold_df[c])]
        summary = fold_df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
        summary.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).rstrip("_") if isinstance(col, tuple) else str(col)
            for col in summary.columns
        ]
        fold_counts = fold_df.groupby(group_cols, dropna=False).size().rename("n_folds_completed").reset_index()
        return summary.merge(fold_counts, on=group_cols, how="left")

    def experiment_params(self) -> dict[str, Any]:
        return {
            "split_seed": self.cfg.split_seed,
            "n_folds": self.cfg.n_folds,
            "outer_test_fraction_per_fold": 1.0 / self.cfg.n_folds,
            "validation_fraction_of_train": self.cfg.validation_fraction_of_train,
            "word_window_size": self.cfg.word_window_size,
            "word_window_stride": self.cfg.word_window_stride,
            "max_length": self.cfg.max_length,
            "num_train_epochs": self.cfg.num_train_epochs,
            "learning_rate": self.cfg.learning_rate,
            "optimizer": self.cfg.optimizer_name,
            "weight_decay": self.cfg.weight_decay,
            "warmup_ratio": self.cfg.warmup_ratio,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "gradient_checkpointing": self.cfg.gradient_checkpointing,
            "mixed_precision": self.cfg.mixed_precision,
            "force_trainable_params_fp32": self.cfg.force_trainable_params_fp32,
            "best_model_metric": self.cfg.best_model_metric,
            "best_model_greater_is_better": self.cfg.best_model_greater_is_better,
            "early_stopping_patience": self.cfg.early_stopping_patience,
            "early_stopping_threshold": self.cfg.early_stopping_threshold,
            "quiet_training": self.cfg.quiet_training,
            "trainer_logging_strategy": self.cfg.trainer_logging_strategy,
        }

    def prepare(self) -> None:
        self.setup_paths()
        self.configure_quiet_output()
        print("torch:", torch.__version__)
        print(json.dumps(self.cfg.to_serializable_dict(), indent=2, default=str))

        read_summary = self.load_records()
        read_summary.to_csv(self.cfg.results_root / "dataset_read_summary.csv", index=False)
        label_df = self.build_label_maps()
        label_df.to_csv(self.cfg.results_root / "label_map.csv", index=False)
        split_summary = self.build_all_cv_assignments()
        split_summary.to_csv(self.cfg.results_root / "strategy_split_summary.csv", index=False)
        chunk_summary = self.build_examples()
        chunk_summary.to_csv(self.cfg.results_root / "chunk_summary.csv", index=False)

    def run(self) -> pd.DataFrame:
        self.prepare()
        if self.cfg.dry_run:
            print("Dry run completed. No training executed.")
            return pd.DataFrame()

        experiment_id = self.setup_mlflow()
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        print("MLflow experiment:", self.cfg.mlflow_experiment_name, "id=", experiment_id)

        experiment_params = self.experiment_params()
        config_path = self.cfg.results_root / "resolved_experiment_config.json"
        config_path.write_text(json.dumps(self.cfg.to_serializable_dict(), indent=2, default=str), encoding="utf-8")

        with self.maybe_mlflow_run(
            run_name="token_classification_cv_experiment",
            nested=False,
            tags={"run_level": "experiment", "task": "token_classification"},
        ):
            self.log_params_to_mlflow(experiment_params)
            mlflow.log_artifact(str(config_path), artifact_path="config")

            for dataset_name in self.state.selected_datasets:
                with self.maybe_mlflow_run(
                    run_name=f"dataset__{self.safe_name(dataset_name)}",
                    nested=True,
                    tags={"run_level": "dataset", "dataset_name": dataset_name},
                ):
                    self.log_params_to_mlflow({"dataset_name": dataset_name, **experiment_params})
                    mlflow.log_artifact(str(self.state.split_plan_paths[dataset_name]), artifact_path="split_plan")

                    strategies_for_dataset = self.state.selected_strategies_by_dataset[dataset_name]
                    for strategy_idx, strategy in enumerate(strategies_for_dataset, start=1):
                        self.qprint("=" * 110)
                        self.qprint(f"Dataset: {dataset_name} | Strategy [{strategy_idx}/{len(strategies_for_dataset)}]: {strategy}")
                        self.qprint("=" * 110)

                        with self.maybe_mlflow_run(
                            run_name=f"strategy__{self.safe_name(strategy)}",
                            nested=True,
                            tags={"run_level": "strategy", "dataset_name": dataset_name, "strategy": strategy},
                        ):
                            self.log_params_to_mlflow({"dataset_name": dataset_name, "strategy": strategy, **experiment_params})

                            for model_idx, model_spec in enumerate(self.cfg.model_registry, start=1):
                                initialization_mode = self.model_initialization_mode(model_spec)
                                self.qprint("-" * 110)
                                self.qprint(
                                    f"Model [{model_idx}/{len(self.cfg.model_registry)}]: {model_spec.name} "
                                    f"({model_spec.model_name_or_path}) | {initialization_mode}"
                                )
                                self.qprint("-" * 110)

                                tokenizer = self.build_tokenizer(model_spec)
                                data_collator = DataCollatorForTokenClassification(
                                    tokenizer=tokenizer,
                                    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
                                )
                                model_run_rows = []

                                with self.maybe_mlflow_run(
                                    run_name=f"model__{self.safe_name(model_spec.name)}",
                                    nested=True,
                                    tags={
                                        "run_level": "model",
                                        "dataset_name": dataset_name,
                                        "strategy": strategy,
                                        "model_name": model_spec.name,
                                        "model_name_or_path": model_spec.model_name_or_path,
                                        "initialization_mode": initialization_mode,
                                    },
                                ):
                                    self.log_params_to_mlflow({
                                        "dataset_name": dataset_name,
                                        "strategy": strategy,
                                        "model_name": model_spec.name,
                                        "model_name_or_path": model_spec.model_name_or_path,
                                        "tokenizer_name_or_path": model_spec.tokenizer_name_or_path or model_spec.model_name_or_path,
                                        "init_from_pretrained": model_spec.init_from_pretrained,
                                        "initialization_mode": initialization_mode,
                                        **experiment_params,
                                    })

                                    for fold_index in range(self.cfg.n_folds):
                                        fold_seed = self.cfg.split_seed + fold_index
                                        set_seed(fold_seed)
                                        run_id = (
                                            f"{self.safe_name(dataset_name)}__"
                                            f"{self.safe_name(strategy)}__"
                                            f"{self.safe_name(model_spec.name)}__"
                                            f"fold_{fold_index}"
                                        )
                                        self.qprint(f"\nFold {fold_index + 1}/{self.cfg.n_folds}: {run_id}")

                                        datasets = self.make_hf_datasets(dataset_name, strategy, fold_index, tokenizer)
                                        model = self.build_model(model_spec)
                                        n_total_params = self.count_total_parameters(model)
                                        n_trainable_params = self.count_trainable_parameters(model)
                                        trainable_ratio = n_trainable_params / n_total_params if n_total_params else 0.0

                                        training_args = self.make_training_args(run_id=run_id, seed=fold_seed)
                                        trainer = self.build_trainer(model, training_args, datasets, tokenizer, data_collator)

                                        with self.maybe_mlflow_run(
                                            run_name=f"fold_{fold_index}",
                                            nested=True,
                                            tags={
                                                "run_level": "fold",
                                                "dataset_name": dataset_name,
                                                "strategy": strategy,
                                                "model_name": model_spec.name,
                                                "fold": fold_index,
                                                "initialization_mode": initialization_mode,
                                            },
                                        ):
                                            fold_params = {
                                                "dataset_name": dataset_name,
                                                "strategy": strategy,
                                                "model_name": model_spec.name,
                                                "model_name_or_path": model_spec.model_name_or_path,
                                                "init_from_pretrained": model_spec.init_from_pretrained,
                                                "initialization_mode": initialization_mode,
                                                "fold": fold_index,
                                                "fold_seed": fold_seed,
                                                "total_parameters": n_total_params,
                                                "trainable_parameters": n_trainable_params,
                                                "trainable_parameter_ratio": trainable_ratio,
                                                "n_train_chunks": len(datasets["train"]),
                                                "n_validation_chunks": len(datasets["validation"]),
                                                "n_test_chunks": len(datasets["test"]),
                                            }
                                            self.log_params_to_mlflow({**experiment_params, **fold_params})

                                            train_result = self.run_trainer_quietly(trainer.train)
                                            self.log_trainer_history_to_mlflow(trainer)

                                            val_metrics = self.run_trainer_quietly(
                                                trainer.evaluate,
                                                eval_dataset=datasets["validation"],
                                                metric_key_prefix="val",
                                            )
                                            test_output = self.run_trainer_quietly(
                                                trainer.predict,
                                                test_dataset=datasets["test"],
                                                metric_key_prefix="test",
                                            )
                                            test_metrics = dict(test_output.metrics)

                                            report_df = self.per_label_report_dataframe(test_output)
                                            report_path = self.cfg.results_root / f"{run_id}_per_label_test_report.csv"
                                            report_df.to_csv(report_path, index=False)
                                            self.state.per_label_report_paths.append(report_path)

                                            best_model_dir = self.cfg.runs_root / run_id / "best_model"
                                            trainer.save_model(best_model_dir)
                                            tokenizer.save_pretrained(best_model_dir)

                                            train_metrics = dict(train_result.metrics)
                                            best_metric = getattr(trainer.state, "best_metric", None)
                                            best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)

                                            result_row = {
                                                "dataset_name": dataset_name,
                                                "strategy": strategy,
                                                "model_name": model_spec.name,
                                                "model_name_or_path": model_spec.model_name_or_path,
                                                "init_from_pretrained": model_spec.init_from_pretrained,
                                                "initialization_mode": initialization_mode,
                                                "fold": fold_index,
                                                "fold_seed": fold_seed,
                                                "total_parameters": n_total_params,
                                                "trainable_parameters": n_trainable_params,
                                                "trainable_parameter_ratio": trainable_ratio,
                                                "n_train_chunks": len(datasets["train"]),
                                                "n_validation_chunks": len(datasets["validation"]),
                                                "n_test_chunks": len(datasets["test"]),
                                                "word_window_size": self.cfg.word_window_size,
                                                "word_window_stride": self.cfg.word_window_stride,
                                                "max_length": self.cfg.max_length,
                                                "best_validation_metric": best_metric,
                                                "best_model_checkpoint": best_checkpoint,
                                                "best_model_dir": str(best_model_dir),
                                                "per_label_report_path": str(report_path),
                                            }
                                            result_row.update(train_metrics)
                                            result_row.update(val_metrics)
                                            result_row.update(test_metrics)

                                            self.state.fold_results.append(result_row)
                                            model_run_rows.append(result_row)

                                            self.log_metrics_to_mlflow(train_metrics)
                                            self.log_metrics_to_mlflow(val_metrics)
                                            self.log_metrics_to_mlflow(test_metrics)
                                            if best_metric is not None and isinstance(best_metric, (int, float, np.integer, np.floating)):
                                                self.log_metrics_to_mlflow({"best_validation_metric": best_metric})

                                            mlflow.log_artifact(str(report_path), artifact_path="reports")
                                            pd.DataFrame(self.state.fold_results).to_csv(
                                                self.cfg.results_root / "strategy_model_fold_results.csv",
                                                index=False,
                                            )

                                        del trainer
                                        del model
                                        self.cleanup_after_fold()

                                    model_fold_df = pd.DataFrame(model_run_rows)
                                    model_summary_df = self.summarize_cv_results(
                                        model_fold_df,
                                        group_cols=["dataset_name", "strategy", "model_name"],
                                    )
                                    model_summary_path = (
                                        self.cfg.results_root
                                        / f"{self.safe_name(dataset_name)}__{self.safe_name(strategy)}__{self.safe_name(model_spec.name)}_cv_summary.csv"
                                    )
                                    model_summary_df.to_csv(model_summary_path, index=False)
                                    mlflow.log_artifact(str(model_summary_path), artifact_path="summaries")
                                    if not model_summary_df.empty:
                                        summary_row = model_summary_df.iloc[0].to_dict()
                                        summary_metrics = {
                                            f"cv_{k}": v
                                            for k, v in summary_row.items()
                                            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)
                                        }
                                        self.log_metrics_to_mlflow(summary_metrics)

                                del tokenizer
                                self.cleanup_after_fold()

        return self.finalize_results()

    def finalize_results(self) -> pd.DataFrame:
        print("Done.")
        fold_results_df = pd.DataFrame(self.state.fold_results)
        if fold_results_df.empty:
            print("No fold results produced.")
            return fold_results_df

        fold_results_path = self.cfg.results_root / "strategy_model_fold_results.csv"
        fold_results_df.to_csv(fold_results_path, index=False)

        cv_summary_df = self.summarize_cv_results(
            fold_results_df,
            group_cols=["dataset_name", "strategy", "model_name", "model_name_or_path"],
        )
        cv_summary_path = self.cfg.results_root / "strategy_model_cv_summary.csv"
        cv_summary_df.to_csv(cv_summary_path, index=False)

        sort_col = "test_non_o_micro_f1_mean" if "test_non_o_micro_f1_mean" in cv_summary_df.columns else None
        if sort_col:
            cv_summary_df = cv_summary_df.sort_values(sort_col, ascending=False)

        print("Saved fold-level results to:", fold_results_path)
        print("Saved CV summary to:", cv_summary_path)
        return cv_summary_df


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_scalar(value: str) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "all":
        return "all"
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def parse_selection(value: Optional[str]) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() == "all":
        return "all"
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def load_json_object(value: str) -> Any:
    candidate = Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(value)


def default_model_presets() -> dict[str, ModelSpec]:
    specs = [
        ModelSpec("bert-mlsm", "SzegedAI/bert-medium-mlsm"),
        ModelSpec("bert_small", "google/bert_uncased_L-4_H-512_A-8"),

        ModelSpec("deberta_v3_xsmall", "microsoft/deberta-v3-xsmall"),
        ModelSpec("deberta_v3_small", "microsoft/deberta-v3-small"),
        ModelSpec("deberta_v3_large", "microsoft/deberta-v3-large"),

        ModelSpec("distilbert_base", "distilbert/distilbert-base-uncased"),
        ModelSpec("minilm_l12_h384", "microsoft/MiniLM-L12-H384-uncased"),

        ModelSpec("modernbert_base", "answerdotai/ModernBERT-base"),
        ModelSpec("modernbert_large", "answerdotai/ModernBERT-large"),

        ModelSpec("eurobert_210m", "EuroBERT/EuroBERT-210m", model_kwargs={"trust_remote_code": True}, tokenizer_kwargs={"trust_remote_code": True}),

        ModelSpec("nomic_bert_2048", "nomic-ai/nomic-bert-2048", model_kwargs={"trust_remote_code": True}, tokenizer_kwargs={"trust_remote_code": True}),
        ModelSpec("xlm_roberta_large", "FacebookAI/xlm-roberta-base"),
    ]
    return {spec.name: spec for spec in specs}


def copy_model_spec(spec: ModelSpec, trust_remote_code: bool = False, init_from_pretrained: Optional[bool] = None) -> ModelSpec:
    model_kwargs = dict(spec.model_kwargs)
    tokenizer_kwargs = dict(spec.tokenizer_kwargs)
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        tokenizer_kwargs["trust_remote_code"] = True
    return ModelSpec(
        name=spec.name,
        model_name_or_path=spec.model_name_or_path,
        tokenizer_name_or_path=spec.tokenizer_name_or_path,
        init_from_pretrained=spec.init_from_pretrained if init_from_pretrained is None else init_from_pretrained,
        revision=spec.revision,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def model_name_from_path(identifier: str) -> str:
    if "=" in identifier:
        identifier = identifier.split("=", 1)[0]
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in identifier).strip("_")


def model_spec_from_identifier(identifier: str, trust_remote_code: bool = False, init_from_pretrained: Optional[bool] = None) -> ModelSpec:
    presets = default_model_presets()
    identifier = identifier.strip()
    if identifier in presets:
        return copy_model_spec(presets[identifier], trust_remote_code=trust_remote_code, init_from_pretrained=init_from_pretrained)
    if "=" in identifier:
        name, path = identifier.split("=", 1)
        name = name.strip()
        path = path.strip()
    else:
        name = model_name_from_path(identifier)
        path = identifier

    model_kwargs = {"trust_remote_code": True} if trust_remote_code else {}
    tokenizer_kwargs = {"trust_remote_code": True} if trust_remote_code else {}
    return ModelSpec(
        name=name,
        model_name_or_path=path,
        init_from_pretrained=True if init_from_pretrained is None else init_from_pretrained,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def model_specs_from_json(value: str, trust_remote_code: bool = False, init_from_pretrained: Optional[bool] = None) -> list[ModelSpec]:
    raw = load_json_object(value)
    if not isinstance(raw, list):
        raise ValueError("Model config JSON must be a list of model-spec objects.")
    specs = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each model config entry must be an object.")
        spec = ModelSpec(**item)
        specs.append(copy_model_spec(spec, trust_remote_code=trust_remote_code, init_from_pretrained=init_from_pretrained))
    return specs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run serialized-OCR token-classification experiments with configurable datasets, strategies, models, and MLflow logging."
    )

    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)

    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--strategies", type=str, default=None, help="Comma-separated strategy names or 'all'.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated preset names or Hugging Face model ids.")
    parser.add_argument("--model", action="append", default=None, help="Additional model, either name=hf_id or hf_id. Can be repeated.")
    parser.add_argument("--model-config", type=str, default=None, help="JSON string or path to a JSON list of ModelSpec objects.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train-from-scratch", action="store_true", help="Initialize selected model architectures from config instead of pretrained weights.")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument(
        "--validation-fraction-of-train",
        type=float,
        default=None,
        help="Fraction of the outer training folds reserved for validation. Default: 0.10.",
    )
    parser.add_argument("--sampling", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--word-window-size", type=int, default=None)
    parser.add_argument("--word-window-stride", type=int, default=None)

    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr-scheduler-type", type=str, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", type=str2bool, nargs="?", const=True, default=None)

    parser.add_argument("--best-model-metric", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--mixed-precision", choices=["auto", "bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)

    parser.add_argument("--quiet", dest="quiet_training", action="store_true", default=None)
    parser.add_argument("--no-quiet", dest="quiet_training", action="store_false")
    parser.add_argument("--show-run-progress", action="store_true")
    parser.add_argument("--trainer-logging-strategy", choices=["no", "steps", "epoch"], default=None)

    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-artifact-root", type=Path, default=None)
    parser.add_argument("--mlflow-db-path", type=Path, default=None)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Generic override, e.g. --set max_length=2048 or --set MAX_LENGTH=2048.")
    return parser


def apply_if_not_none(kwargs: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        kwargs[key] = value


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    kwargs: dict[str, Any] = {}
    apply_if_not_none(kwargs, "project_root", args.project_root)
    apply_if_not_none(kwargs, "processed_datasets_root", args.processed_root)
    apply_if_not_none(kwargs, "runs_root", args.runs_root)
    apply_if_not_none(kwargs, "results_root", args.results_root)
    apply_if_not_none(kwargs, "split_seed", args.seed)
    apply_if_not_none(kwargs, "n_folds", args.n_folds)
    apply_if_not_none(kwargs, "validation_fraction_of_train", args.validation_fraction_of_train)
    apply_if_not_none(kwargs, "sampling", args.sampling)
    apply_if_not_none(kwargs, "max_length", args.max_length)
    apply_if_not_none(kwargs, "word_window_size", args.word_window_size)
    apply_if_not_none(kwargs, "word_window_stride", args.word_window_stride)
    apply_if_not_none(kwargs, "num_train_epochs", args.epochs)
    apply_if_not_none(kwargs, "per_device_train_batch_size", args.train_batch_size)
    apply_if_not_none(kwargs, "per_device_eval_batch_size", args.eval_batch_size)
    apply_if_not_none(kwargs, "learning_rate", args.learning_rate)
    apply_if_not_none(kwargs, "optimizer_name", args.optimizer)
    apply_if_not_none(kwargs, "weight_decay", args.weight_decay)
    apply_if_not_none(kwargs, "warmup_ratio", args.warmup_ratio)
    apply_if_not_none(kwargs, "gradient_accumulation_steps", args.grad_accum)
    apply_if_not_none(kwargs, "lr_scheduler_type", args.lr_scheduler_type)
    apply_if_not_none(kwargs, "max_grad_norm", args.max_grad_norm)
    apply_if_not_none(kwargs, "gradient_checkpointing", args.gradient_checkpointing)
    apply_if_not_none(kwargs, "best_model_metric", args.best_model_metric)
    apply_if_not_none(kwargs, "early_stopping_patience", args.early_stopping_patience)
    apply_if_not_none(kwargs, "mixed_precision", args.mixed_precision)
    apply_if_not_none(kwargs, "save_total_limit", args.save_total_limit)
    apply_if_not_none(kwargs, "dataloader_num_workers", args.dataloader_num_workers)
    apply_if_not_none(kwargs, "quiet_training", args.quiet_training)
    apply_if_not_none(kwargs, "trainer_logging_strategy", args.trainer_logging_strategy)
    apply_if_not_none(kwargs, "mlflow_experiment_name", args.mlflow_experiment_name)
    apply_if_not_none(kwargs, "mlflow_tracking_uri", args.mlflow_tracking_uri)
    apply_if_not_none(kwargs, "mlflow_artifact_root", args.mlflow_artifact_root)
    apply_if_not_none(kwargs, "mlflow_db_path", args.mlflow_db_path)

    if args.datasets is not None:
        kwargs["datasets_to_run"] = parse_selection(args.datasets)
    if args.strategies is not None:
        kwargs["strategies_to_run"] = parse_selection(args.strategies)
    if args.show_run_progress:
        kwargs["show_run_progress"] = True
    if args.no_early_stopping:
        kwargs["early_stopping_patience"] = None
    if args.dry_run:
        kwargs["dry_run"] = True

    init_from_pretrained = False if args.train_from_scratch else None
    selected_models: list[ModelSpec] | None = None
    if args.model_config is not None:
        selected_models = model_specs_from_json(args.model_config, trust_remote_code=args.trust_remote_code, init_from_pretrained=init_from_pretrained)
    elif args.models is not None:
        selected_models = [
            model_spec_from_identifier(item, trust_remote_code=args.trust_remote_code, init_from_pretrained=init_from_pretrained)
            for item in parse_selection(args.models)
        ]

    if args.model:
        if selected_models is None:
            selected_models = []
        selected_models.extend(
            model_spec_from_identifier(item, trust_remote_code=args.trust_remote_code, init_from_pretrained=init_from_pretrained)
            for item in args.model
        )

    if selected_models is not None:
        kwargs["model_registry"] = selected_models

    config = ExperimentConfig(**kwargs)
    apply_generic_overrides(config, args.overrides)
    return config


def apply_generic_overrides(config: ExperimentConfig, overrides: list[str]) -> None:
    if not overrides:
        return

    valid_fields = {field.name for field in fields(config)}
    legacy_mapping = {field.name.upper(): field.name for field in fields(config)}
    legacy_mapping.update({
        "SPLIT_SEED": "split_seed",
        "N_FOLDS": "n_folds",
        "VAL_RATIO": "validation_fraction_of_train",
        "VALIDATION_FRACTION_OF_TRAIN": "validation_fraction_of_train",
        "SAMPLING": "sampling",
        "DATASETS_TO_RUN": "datasets_to_run",
        "STRATEGIES_TO_RUN": "strategies_to_run",
        "MAX_LENGTH": "max_length",
        "WORD_WINDOW_SIZE": "word_window_size",
        "WORD_WINDOW_STRIDE": "word_window_stride",
        "NUM_TRAIN_EPOCHS": "num_train_epochs",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "per_device_train_batch_size",
        "PER_DEVICE_EVAL_BATCH_SIZE": "per_device_eval_batch_size",
        "LEARNING_RATE": "learning_rate",
        "OPTIMIZER_NAME": "optimizer_name",
        "WEIGHT_DECAY": "weight_decay",
        "WARMUP_RATIO": "warmup_ratio",
        "GRADIENT_ACCUMULATION_STEPS": "gradient_accumulation_steps",
        "LR_SCHEDULER_TYPE": "lr_scheduler_type",
        "MAX_GRAD_NORM": "max_grad_norm",
        "BEST_MODEL_METRIC": "best_model_metric",
        "EARLY_STOPPING_PATIENCE": "early_stopping_patience",
        "MIXED_PRECISION": "mixed_precision",
        "SAVE_TOTAL_LIMIT": "save_total_limit",
        "QUIET_TRAINING": "quiet_training",
        "MLFLOW_EXPERIMENT_NAME": "mlflow_experiment_name",
        "MLFLOW_TRACKING_URI": "mlflow_tracking_uri",
    })

    path_fields = {"project_root", "processed_datasets_root", "runs_root", "results_root", "mlflow_db_path", "mlflow_artifact_root"}

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override {item!r}. Use name=value.")
        raw_key, raw_value = item.split("=", 1)
        raw_key = raw_key.strip()
        key = legacy_mapping.get(raw_key, raw_key)
        if key not in valid_fields:
            raise ValueError(f"Unknown config field {raw_key!r}. Valid fields include: {sorted(valid_fields)}")
        value = parse_scalar(raw_value)
        if key in path_fields and value is not None:
            value = Path(value).resolve()
        setattr(config, key, value)

    config.__post_init__()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = build_config_from_args(args)
    runner = ExperimentRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
