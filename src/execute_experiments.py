from pathlib import Path
import os
import json
import random
import gc
import inspect
import shutil
import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import mlflow
from mlflow.tracking import MlflowClient

from datasets import Dataset
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

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as transformers_logging

try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None

import time
import torch
print(torch.__version__)

PROJECT_ROOT = Path.cwd()

PROCESSED_DATASETS_ROOT = PROJECT_ROOT / "data" / "processed"
RUNS_ROOT = PROJECT_ROOT / "runs"
RESULTS_ROOT = PROJECT_ROOT / "results"

RUNS_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("PROCESSED_DATASETS_ROOT:", PROCESSED_DATASETS_ROOT)
print("RUNS_ROOT:", RUNS_ROOT)
print("RESULTS_ROOT:", RESULTS_ROOT)
print("CUDA available:", torch.cuda.is_available())

SPLIT_SEED = 42

N_FOLDS = 2
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SAMPLING = 20


AUTO_DISCOVER_DATASETS = True
DATASETS_TO_RUN = "all"
# DATASETS_TO_RUN = ["multi_docs",]
# STRATEGIES_TO_RUN = "all"
STRATEGIES_TO_RUN = ["column_aware",]

# Model registry. Add future models here. Each entry must be compatible with
# AutoModelForTokenClassification and a fast tokenizer that supports word_ids().
@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    init_from_pretrained: bool = True
    revision: Optional[str] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY = [
    ModelSpec("bert-mlsm", "SzegedAI/bert-medium-mlsm"),
    # ModelSpec("bert_small", "google/bert_uncased_L-4_H-512_A-8"),
    # ModelSpec("distilbert_base", "distilbert/distilbert-base-uncased"),
    # ModelSpec("minilm_l12_h384", "microsoft/MiniLM-L12-H384-uncased"),
    # ModelSpec("electra_small", "google/electra-small-discriminator"),
    # ModelSpec("deberta_v3_xsmall", "microsoft/deberta-v3-xsmall"),
    # ModelSpec("deberta_v3_small", "microsoft/deberta-v3-small"),
    # larger encoders
    # ModelSpec("modernbert_base", "answerdotai/ModernBERT-base"),
    # ModelSpec("roberta_base", "FacebookAI/roberta-base"),
    # ModelSpec("eurobert_210m", "EuroBERT/EuroBERT-210m"),
]

MAX_LENGTH = 512
WORD_WINDOW_SIZE = 384
WORD_WINDOW_STRIDE = 0

# Chunk filtering.
DROP_CHUNKS_WITH_NO_TRAINABLE_TOKENS = True

NUM_TRAIN_EPOCHS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
OPTIMIZER_NAME = "adamw_torch"  # PyTorch AdamW via Transformers TrainingArguments.optim
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRADIENT_ACCUMULATION_STEPS = 1
LR_SCHEDULER_TYPE = "linear"
MAX_GRAD_NORM = 1.0

BEST_MODEL_METRIC = "macro_f1"
BEST_MODEL_GREATER_IS_BETTER = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.0

MIXED_PRECISION = "auto"
FORCE_TRAINABLE_PARAMS_FP32 = True
SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 0

# Quiet-output controls.
# Quiet mode suppresses model-loading warnings, HF/datasets progress bars, Trainer logs,
# and the notebook's per-fold display calls. Metrics are still written to CSV and MLflow.
QUIET_TRAINING = True
SUPPRESS_MODEL_LOAD_WARNINGS = True
SUPPRESS_TRAIN_STDOUT_STDERR = True
SHOW_RUN_PROGRESS = False
TRANSFORMERS_VERBOSITY = "error"  # "error", "warning", "info"
DATASETS_VERBOSITY = "error"
TRAINER_LOGGING_STRATEGY = "no"  # "no", "steps", or "epoch"

OVERWRITE_OUTPUT_DIR = True

MLFLOW_EXPERIMENT_NAME = "serialization_strategies_" + str(int(time.time()))
MLFLOW_DB_PATH = (PROJECT_ROOT / "mlflow.db").resolve()
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlartifacts").resolve()
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)

print({
    "SPLIT_SEED": SPLIT_SEED,
    "N_FOLDS": N_FOLDS,
    "ratios": (TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
    "datasets": DATASETS_TO_RUN,
    "strategies": STRATEGIES_TO_RUN,
    "models": [asdict(m) for m in MODEL_REGISTRY],
    "MAX_LENGTH": MAX_LENGTH,
    "WORD_WINDOW_SIZE": WORD_WINDOW_SIZE,
    "WORD_WINDOW_STRIDE": WORD_WINDOW_STRIDE,
    "NUM_TRAIN_EPOCHS": NUM_TRAIN_EPOCHS,
    "BEST_MODEL_METRIC": BEST_MODEL_METRIC,
    "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "QUIET_TRAINING": QUIET_TRAINING,
})

def configure_quiet_output() -> None:
    """Centralized suppression for noisy notebook/training output."""
    if not QUIET_TRAINING:
        transformers_logging.set_verbosity_info()
        if datasets_logging is not None:
            datasets_logging.set_verbosity_info()
        if enable_datasets_progress_bar is not None:
            enable_datasets_progress_bar()
        return

    # Python warnings.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
    warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
    warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")

    # Standard logging.
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

    # Hugging Face loggers.
    if TRANSFORMERS_VERBOSITY == "error":
        transformers_logging.set_verbosity_error()
    elif TRANSFORMERS_VERBOSITY == "warning":
        transformers_logging.set_verbosity_warning()
    else:
        transformers_logging.set_verbosity_info()

    if datasets_logging is not None:
        if DATASETS_VERBOSITY == "error":
            datasets_logging.set_verbosity_error()
        elif DATASETS_VERBOSITY == "warning":
            datasets_logging.set_verbosity_warning()
        else:
            datasets_logging.set_verbosity_info()

    if disable_datasets_progress_bar is not None:
        disable_datasets_progress_bar()


def qprint(*args, **kwargs) -> None:
    if SHOW_RUN_PROGRESS:
        print(*args, **kwargs)


@contextmanager
def quiet_section(enabled: bool = True):
    """Suppress stdout/stderr only inside noisy calls, while preserving exceptions."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


configure_quiet_output()

while mlflow.active_run() is not None:
    mlflow.end_run()

MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
existing_experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if existing_experiment is None:
    experiment_id = client.create_experiment(
        name=MLFLOW_EXPERIMENT_NAME,
        artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri(),
    )
else:
    experiment_id = existing_experiment.experiment_id

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("MLflow experiment:", MLFLOW_EXPERIMENT_NAME, "id=", experiment_id)

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


def discover_strategy_files(processed_root: Path) -> dict[str, Path]:
    strategy_files = {}
    for path in sorted(processed_root.glob("*/all.jsonl")):
        strategy_files[path.parent.name] = path
    return strategy_files


def discover_dataset_roots() -> dict[str, Path]:
    dataset_roots: dict[str, Path] = {}

    if AUTO_DISCOVER_DATASETS:
        for candidate_root in sorted(PROCESSED_DATASETS_ROOT.iterdir()):
            if not candidate_root.is_dir():
                continue
            if discover_strategy_files(candidate_root):
                dataset_roots.setdefault(candidate_root.name, candidate_root)

    if DATASETS_TO_RUN != "all":
        requested = set(DATASETS_TO_RUN)
        missing = sorted(requested.difference(dataset_roots))
        assert not missing, f"Requested datasets not found: {missing}"
        dataset_roots = {name: root for name, root in dataset_roots.items() if name in requested}

    existing_roots = {}
    for name, root in dataset_roots.items():
        if root.exists():
            existing_roots[name] = root
        else:
            print(f"Skipping dataset {name!r}; missing folder: {root}")

    assert existing_roots, "No processed datasets found. Expected folders under data/processed containing */all.jsonl."
    return existing_roots


available_dataset_roots = discover_dataset_roots()

selected_strategy_files_by_dataset: dict[str, dict[str, Path]] = {}
records_by_dataset_strategy: dict[str, dict[str, list[dict]]] = {}

for dataset_name, dataset_root in available_dataset_roots.items():
    available_strategy_files = discover_strategy_files(dataset_root)

    print(f"\nDataset: {dataset_name}")
    print("Available strategies:")
    for name, path in available_strategy_files.items():
        print(f"  {name:24s} -> {path}")

    if STRATEGIES_TO_RUN == "all":
        selected_strategy_files = available_strategy_files
    else:
        missing = sorted(set(STRATEGIES_TO_RUN).difference(available_strategy_files))

        selected_strategy_files = {
            name: available_strategy_files[name]
            for name in STRATEGIES_TO_RUN
            if name in available_strategy_files
        }

    if not selected_strategy_files:
        print(f"Skipping dataset {dataset_name!r}; no selected strategies found.")
        continue

    selected_strategy_files_by_dataset[dataset_name] = selected_strategy_files
    records_by_dataset_strategy[dataset_name] = {}

    for strategy, path in selected_strategy_files.items():
        records = read_jsonl(path)

        if SAMPLING is not None:
            rng = random.Random(SPLIT_SEED)
            records = records.copy()
            rng.shuffle(records)
            records = records[:SAMPLING]

        records_by_dataset_strategy[dataset_name][strategy] = records

read_summary_rows = []
for dataset_name, records_by_strategy in records_by_dataset_strategy.items():
    for strategy, records in records_by_strategy.items():
        read_summary_rows.append({
            "dataset_name": dataset_name,
            "strategy": strategy,
            "n_records": len(records),
            "path": str(selected_strategy_files_by_dataset[dataset_name][strategy]),
        })

dataset_read_summary = pd.DataFrame(read_summary_rows)

selected_datasets = sorted(records_by_dataset_strategy.keys())
selected_strategies_by_dataset = {
    dataset_name: sorted(records_by_strategy.keys())
    for dataset_name, records_by_strategy in records_by_dataset_strategy.items()
}

IGNORE_LABEL = -100


def is_real_label(x) -> bool:
    return isinstance(x, str) and x != "" and x != str(IGNORE_LABEL)


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


label_set = set()

for dataset_name, records_by_strategy in records_by_dataset_strategy.items():
    for strategy, records in records_by_strategy.items():
        for rec in records:
            for label in rec.get("labels", []):
                if is_real_label(label):
                    label_set.add(label)

label_list = sort_bio_labels(label_set)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
non_o_label_ids = [i for i, label in id2label.items() if label != "O"]

print("n_labels:", len(label_list))
print(label_list)

label_df = pd.DataFrame({
    "label_id": list(range(len(label_list))),
    "label": label_list,
})

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


def normalize_label_name(label: Any) -> Optional[str]:
    if not is_real_label(label):
        return None
    if label == "O":
        return "O"
    if isinstance(label, str) and "-" in label:
        return label.split("-", 1)[1]
    return str(label)


def build_doc_strata(records_by_strategy: dict[str, list[dict]]) -> dict[str, str]:
    # Build a coarse document-level stratification key.
    # Exact BIO-label combinations are often too sparse for StratifiedKFold.
    # If any stratum is too rare for N_FOLDS, the splitter falls back to KFold.
    entity_counts_by_doc = defaultdict(Counter)

    for strategy, records in records_by_strategy.items():
        for rec_idx, rec in enumerate(records):
            doc_key = record_doc_key(rec, strategy, rec_idx)
            labels = rec.get("labels", [])
            for label in labels:
                normalized = normalize_label_name(label)
                if normalized is None or normalized == "O":
                    continue
                entity_counts_by_doc[doc_key][normalized] += 1

    all_doc_keys = sorted({
        record_doc_key(rec, strategy, rec_idx)
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


def can_use_stratification(labels: list[str], min_count: int) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= min_count


def split_validation_test(heldout_keys: list[str], key_to_stratum: dict[str, str], seed: int) -> tuple[set[str], set[str]]:
    assert len(heldout_keys) >= 2, "Each held-out fold must contain at least two documents for validation/test splitting."
    y = [key_to_stratum[k] for k in heldout_keys]
    stratify = y if can_use_stratification(y, min_count=2) else None

    val_keys, test_keys = train_test_split(
        heldout_keys,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return set(val_keys), set(test_keys)


def build_cv_split_assignments(records_by_strategy: dict[str, list[dict]], n_folds: int, seed: int):
    key_to_stratum = build_doc_strata(records_by_strategy)
    doc_keys = np.array(sorted(key_to_stratum.keys()))

    if len(doc_keys) < n_folds:
        raise ValueError(
            f"Not enough documents for {n_folds}-fold CV: found {len(doc_keys)} documents."
        )

    y = np.array([key_to_stratum[k] for k in doc_keys])

    if can_use_stratification(y.tolist(), min_count=n_folds):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(doc_keys, y)
        splitter_name = "StratifiedKFold"
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(doc_keys)
        splitter_name = "KFold"

    assignments_by_fold = {}
    rows = []

    for fold_index, (train_idx, heldout_idx) in enumerate(split_iter):
        train_keys = set(doc_keys[train_idx].tolist())
        heldout_keys = sorted(doc_keys[heldout_idx].tolist())
        validation_keys, test_keys = split_validation_test(
            heldout_keys=heldout_keys,
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
                "splitter": splitter_name,
            })

    assignment_df = pd.DataFrame(rows).sort_values(["fold", "split", "doc_key"])
    return assignments_by_fold, key_to_stratum, assignment_df


cv_split_assignments_by_dataset = {}
doc_strata_by_dataset = {}
cv_assignment_dfs = {}
split_plan_paths = {}

for dataset_name, records_by_strategy in records_by_dataset_strategy.items():
    assignments, doc_strata, assignment_df = build_cv_split_assignments(
        records_by_strategy=records_by_strategy,
        n_folds=N_FOLDS,
        seed=SPLIT_SEED,
    )

    cv_split_assignments_by_dataset[dataset_name] = assignments
    doc_strata_by_dataset[dataset_name] = doc_strata
    cv_assignment_dfs[dataset_name] = assignment_df

    split_plan_path = RESULTS_ROOT / f"{dataset_name}_cv_doc_split_assignments.csv" if "safe_name" in globals() else RESULTS_ROOT / f"{dataset_name}_cv_doc_split_assignments.csv"
    assignment_df.to_csv(split_plan_path, index=False)
    split_plan_paths[dataset_name] = split_plan_path

    cv_split_summary_df = (
        assignment_df
        .groupby(["fold", "split"])
        .size()
        .rename("n_docs")
        .reset_index()
    )
    cv_split_summary_df["dataset_name"] = dataset_name
    cv_split_summary_df["ratio"] = cv_split_summary_df["n_docs"] / assignment_df["doc_key"].nunique()

    print("Saved CV split plan to:", split_plan_path)
    print("Splitter used:", assignment_df["splitter"].iloc[0])

# Per-dataset/per-strategy check. If some strategies have missing documents, ratios may be approximate.
strategy_split_rows = []
for dataset_name, assignments_by_fold in cv_split_assignments_by_dataset.items():
    records_by_strategy = records_by_dataset_strategy[dataset_name]
    for fold_index, assignment in assignments_by_fold.items():
        for strategy, records in records_by_strategy.items():
            strategy_doc_keys = sorted({record_doc_key(rec, strategy, i) for i, rec in enumerate(records)})
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
                "train_ratio": counts.get("train", 0) / n_docs if n_docs else 0,
                "validation_ratio": counts.get("validation", 0) / n_docs if n_docs else 0,
                "test_ratio": counts.get("test", 0) / n_docs if n_docs else 0,
            })

strategy_split_summary_df = pd.DataFrame(strategy_split_rows)

def normalize_record_labels(raw_labels: list) -> list[int]:
    label_ids = []

    for label in raw_labels:
        if label == IGNORE_LABEL or label == str(IGNORE_LABEL) or label is None:
            label_ids.append(IGNORE_LABEL)
        elif isinstance(label, str):
            label_ids.append(label2id[label])
        else:
            # Defensive fallback. Unknown non-string labels are ignored.
            label_ids.append(IGNORE_LABEL)

    return label_ids


def iter_word_windows(n_tokens: int, window_size: int, stride: int):
    assert window_size > 0
    assert stride >= 0
    step = window_size - stride
    assert step > 0, "WORD_WINDOW_STRIDE must be smaller than WORD_WINDOW_SIZE."

    start = 0
    while start < n_tokens:
        end = min(n_tokens, start + window_size)
        yield start, end

        if end >= n_tokens:
            break

        start += step


def make_base_examples_for_strategy(dataset_name: str, strategy: str, records: list[dict]) -> list[dict]:
    examples = []

    for rec_idx, rec in enumerate(records):
        tokens = rec.get("tokens", [])
        labels = rec.get("labels", [])

        if len(tokens) != len(labels):
            raise ValueError(
                f"{dataset_name}/{strategy} record {rec_idx}: len(tokens)={len(tokens)} "
                f"but len(labels)={len(labels)}"
            )

        doc_key = record_doc_key(rec, strategy, rec_idx)
        word_label_ids = normalize_record_labels(labels)

        for chunk_idx, (start, end) in enumerate(
            iter_word_windows(len(tokens), WORD_WINDOW_SIZE, WORD_WINDOW_STRIDE)
        ):
            chunk_tokens = [str(t) for t in tokens[start:end]]
            chunk_label_ids = word_label_ids[start:end]

            if DROP_CHUNKS_WITH_NO_TRAINABLE_TOKENS and all(x == IGNORE_LABEL for x in chunk_label_ids):
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


examples_by_dataset_strategy = {
    dataset_name: {
        strategy: make_base_examples_for_strategy(dataset_name, strategy, records)
        for strategy, records in records_by_strategy.items()
    }
    for dataset_name, records_by_strategy in records_by_dataset_strategy.items()
}

chunk_summary_rows = []
for dataset_name, examples_by_strategy in examples_by_dataset_strategy.items():
    for strategy, examples in examples_by_strategy.items():
        chunk_summary_rows.append({
            "dataset_name": dataset_name,
            "strategy": strategy,
            "n_chunks": len(examples),
            "n_docs": len({ex["doc_key"] for ex in examples}),
            "mean_chunk_words": float(np.mean([len(ex["tokens"]) for ex in examples])) if examples else 0,
            "max_chunk_words": max([len(ex["tokens"]) for ex in examples], default=0),
        })

chunk_summary_df = pd.DataFrame(chunk_summary_rows).sort_values(["dataset_name", "strategy"])

assert all(
    len(examples) > 0
    for examples_by_strategy in examples_by_dataset_strategy.values()
    for examples in examples_by_strategy.values()
), "At least one dataset/strategy has no examples."


def build_tokenizer(model_spec: ModelSpec):
    tokenizer_name = model_spec.tokenizer_name_or_path or model_spec.model_name_or_path
    tokenizer_kwargs = dict(model_spec.tokenizer_kwargs)
    tokenizer_kwargs.setdefault("use_fast", True)

    with quiet_section(QUIET_TRAINING and SUPPRESS_MODEL_LOAD_WARNINGS):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    if not tokenizer.is_fast:
        raise ValueError(
            f"Tokenizer for {model_spec.name} is not fast. Fast tokenizers are required "
            "because this notebook uses tokenizer.word_ids() for token-label alignment."
        )

    return tokenizer


def effective_max_length(tokenizer) -> int:
    tokenizer_limit = getattr(tokenizer, "model_max_length", MAX_LENGTH)
    if tokenizer_limit is None or tokenizer_limit > 1_000_000:
        return MAX_LENGTH
    return min(MAX_LENGTH, int(tokenizer_limit))


def make_tokenize_and_align_labels_fn(tokenizer):
    max_length = effective_max_length(tokenizer)

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
                    label_ids.append(IGNORE_LABEL)
                elif word_idx != previous_word_idx:
                    label_ids.append(word_label_ids[word_idx])
                else:
                    original_label_id = word_label_ids[word_idx]
                    if original_label_id == IGNORE_LABEL:
                        label_ids.append(IGNORE_LABEL)
                    else:
                        original_label = id2label[original_label_id]
                        if original_label.startswith("B-"):
                            inside_label = "I-" + original_label[2:]
                            label_ids.append(label2id.get(inside_label, original_label_id))
                        else:
                            label_ids.append(original_label_id)

                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    return tokenize_and_align_labels


def has_only_o_or_ignored_labels(example: dict) -> bool:
    return all((x == IGNORE_LABEL or id2label[x] == "O") for x in example["word_label_ids"])


def examples_for_dataset_strategy_fold(dataset_name: str, strategy: str, fold_index: int) -> list[dict]:
    assignment = cv_split_assignments_by_dataset[dataset_name][fold_index]
    fold_examples = []

    for ex in examples_by_dataset_strategy[dataset_name][strategy]:
        split = assignment[ex["doc_key"]]

        fold_examples.append({
            **ex,
            "fold": fold_index,
            "experiment_split": split,
        })

    return fold_examples


def make_hf_datasets(dataset_name: str, strategy: str, fold_index: int, tokenizer) -> dict[str, Dataset]:
    examples = examples_for_dataset_strategy_fold(dataset_name, strategy, fold_index)

    split_to_examples = {
        "train": [ex for ex in examples if ex["experiment_split"] == "train"],
        "validation": [ex for ex in examples if ex["experiment_split"] == "validation"],
        "test": [ex for ex in examples if ex["experiment_split"] == "test"],
    }

    datasets = {}
    tokenize_and_align_labels = make_tokenize_and_align_labels_fn(tokenizer)

    for split_name, split_examples in split_to_examples.items():
        assert split_examples, f"{dataset_name}/{strategy} fold {fold_index}: no {split_name} examples."

        ds = Dataset.from_list(split_examples)
        with quiet_section(QUIET_TRAINING):
            tokenized_ds = ds.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=ds.column_names,
                desc=None if QUIET_TRAINING else f"Tokenizing {dataset_name}/{strategy}/fold_{fold_index}/{split_name}",
            )
        datasets[split_name] = tokenized_ds

    return datasets


# Smoke-test one dataset/strategy/model/fold.
first_dataset = selected_datasets[0]
first_strategy = selected_strategies_by_dataset[first_dataset][0]
first_model_spec = MODEL_REGISTRY[0]
_tmp_tokenizer = build_tokenizer(first_model_spec)
_tmp_datasets = make_hf_datasets(first_dataset, first_strategy, fold_index=0, tokenizer=_tmp_tokenizer)

print(first_dataset, first_strategy, first_model_spec.name)
for split_name, ds in _tmp_datasets.items():
    print(split_name, ds)

del _tmp_tokenizer, _tmp_datasets

def flatten_predictions_and_labels(logits_or_predictions, labels):
    if logits_or_predictions.ndim == 3:
        predictions = np.argmax(logits_or_predictions, axis=-1)
    else:
        predictions = logits_or_predictions

    y_true = []
    y_pred = []

    for pred_row, label_row in zip(predictions, labels):
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == IGNORE_LABEL:
                continue
            y_true.append(int(label_id))
            y_pred.append(int(pred_id))

    return np.array(y_true), np.array(y_pred)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    y_true, y_pred = flatten_predictions_and_labels(logits, labels)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "non_o_micro_f1": 0.0,
            "n_eval_tokens": 0,
        }

    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    if non_o_label_ids:
        non_o_p, non_o_r, non_o_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=non_o_label_ids,
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


def per_label_report_dataframe(pred_output) -> pd.DataFrame:
    y_true, y_pred = flatten_predictions_and_labels(pred_output.predictions, pred_output.label_ids)

    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [id2label[i] for i in labels_present]

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

def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def select_mixed_precision_flags() -> dict[str, bool]:
    """Return TrainingArguments precision flags.

    Important: do not load trainable models directly in torch.float16 when using
    fp16=True. AMP expects trainable parameters to remain FP32 while operations
    are autocast to lower precision.
    """
    if not torch.cuda.is_available() or MIXED_PRECISION == "fp32":
        return {"fp16": False, "bf16": False}

    requested = str(MIXED_PRECISION).lower()

    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("MIXED_PRECISION='bf16' was requested, but this CUDA device does not support BF16.")
        return {"fp16": False, "bf16": True}

    if requested == "fp16":
        return {"fp16": True, "bf16": False}

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return {"fp16": False, "bf16": True}
        return {"fp16": True, "bf16": False}

    raise ValueError("MIXED_PRECISION must be one of: 'auto', 'bf16', 'fp16', 'fp32'.")


def model_initialization_mode(model_spec: ModelSpec) -> str:
    return "fine_tuning_from_pretrained" if model_spec.init_from_pretrained else "training_from_scratch"


def force_trainable_parameters_fp32(model) -> None:
    """Keep trainable parameters in FP32 for AMP-compatible fine-tuning."""
    if not FORCE_TRAINABLE_PARAMS_FP32:
        return

    converted = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16}:
            parameter.data = parameter.data.float()
            converted += parameter.numel()

    if converted:
        qprint(f"Converted {converted:,} trainable parameters to FP32 for AMP-compatible training.")


def summarize_trainable_parameter_dtypes(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            key = str(parameter.dtype)
            counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def build_model(model_spec: ModelSpec):
    common_kwargs = {
        "num_labels": len(label_list),
        "id2label": id2label,
        "label2id": label2id,
        **dict(model_spec.model_kwargs),
    }
    if model_spec.revision is not None:
        common_kwargs["revision"] = model_spec.revision

    with quiet_section(QUIET_TRAINING and SUPPRESS_MODEL_LOAD_WARNINGS):
        if model_spec.init_from_pretrained:
            model = AutoModelForTokenClassification.from_pretrained(
                model_spec.model_name_or_path,
                **common_kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_spec.model_name_or_path,
                **common_kwargs,
            )
            model = AutoModelForTokenClassification.from_config(config)

    force_trainable_parameters_fp32(model)
    qprint("Trainable parameter dtypes:", summarize_trainable_parameter_dtypes(model))
    return model


def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_training_args(run_id: str, seed: int) -> TrainingArguments:
    # Create TrainingArguments robustly across transformers versions.
    output_dir = RUNS_ROOT / run_id

    if OVERWRITE_OUTPUT_DIR and output_dir.exists():
        shutil.rmtree(output_dir)

    precision_flags = select_mixed_precision_flags()

    candidate_kwargs = {
        "output_dir": str(output_dir),

        # Optimization.
        "learning_rate": LEARNING_RATE,
        "optim": OPTIMIZER_NAME,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "max_grad_norm": MAX_GRAD_NORM,

        # Saving/logging/eval.
        "save_strategy": "epoch",
        "logging_strategy": TRAINER_LOGGING_STRATEGY,
        "logging_steps": 50,
        "logging_first_step": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": BEST_MODEL_METRIC,
        "greater_is_better": BEST_MODEL_GREATER_IS_BETTER,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "report_to": "none",
        "run_name": run_id,

        # Reproducibility/runtime.
        "seed": seed,
        "data_seed": seed,
        "fp16": precision_flags["fp16"],
        "bf16": precision_flags["bf16"],
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,

        # Quiet notebook output.
        "disable_tqdm": QUIET_TRAINING,
        "log_level": TRANSFORMERS_VERBOSITY,
        "log_level_replica": TRANSFORMERS_VERBOSITY,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters)

    # transformers renamed evaluation_strategy -> eval_strategy in newer versions.
    if "eval_strategy" in params:
        candidate_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        candidate_kwargs["evaluation_strategy"] = "epoch"

    # Fallbacks for older versions that do not support warmup_ratio.
    if "warmup_ratio" not in params and "warmup_steps" in params:
        candidate_kwargs["warmup_steps"] = 0

    filtered_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in params
    }

    ignored_kwargs = sorted(set(candidate_kwargs).difference(filtered_kwargs))
    if ignored_kwargs:
        qprint(f"{run_id}: ignored unsupported TrainingArguments kwargs: {ignored_kwargs}")

    return TrainingArguments(**filtered_kwargs)


def build_trainer(model, training_args, datasets: dict[str, Dataset], tokenizer, data_collator):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    callbacks = []
    if EARLY_STOPPING_PATIENCE is not None:
        if EarlyStoppingCallback is None:
            qprint("EarlyStoppingCallback unavailable in this Transformers version; continuing without early stopping.")
        else:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
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
    return remove_notebook_progress_callbacks(trainer)


def remove_notebook_progress_callbacks(trainer: Trainer) -> Trainer:
    # Remove NotebookProgressCallback, which can fail on manual evaluate/predict.
    callbacks = list(getattr(trainer.callback_handler, "callbacks", []))
    kept_callbacks = [
        cb for cb in callbacks
        if cb.__class__.__name__ != "NotebookProgressCallback"
    ]

    removed = len(callbacks) - len(kept_callbacks)
    if removed:
        qprint(f"Removed {removed} NotebookProgressCallback instance(s).")

    trainer.callback_handler.callbacks = kept_callbacks
    return trainer


def run_trainer_quietly(fn, *args, **kwargs):
    with quiet_section(QUIET_TRAINING and SUPPRESS_TRAIN_STDOUT_STDERR):
        return fn(*args, **kwargs)


def cleanup_after_fold():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



@contextmanager
def maybe_mlflow_run(run_name: str, nested: bool = False, tags: Optional[dict[str, Any]] = None):

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        yield run


def mlflow_param_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, default=str, sort_keys=True)
    else:
        text = str(value)
    return text[:500]


def log_params_to_mlflow(params: dict[str, Any]):
    for key, value in params.items():
        mlflow.log_param(str(key), mlflow_param_value(value))


def log_metrics_to_mlflow(metrics: dict[str, Any], prefix: str = "", step: Optional[int] = None):
    numeric_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            numeric_metrics[f"{prefix}{key}"] = float(value)

    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics, step=step)


def log_trainer_history_to_mlflow(trainer: Trainer, prefix: str = "history_"):

    for row in getattr(trainer.state, "log_history", []):
        step = row.get("step")
        metrics = {
            k: v for k, v in row.items()
            if k not in {"step"} and isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)
        }
        log_metrics_to_mlflow(metrics, prefix=prefix, step=int(step) if step is not None else None)


def summarize_cv_results(fold_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    candidate_cols = [
        c for c in fold_df.columns
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
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]

    fold_counts = fold_df.groupby(group_cols, dropna=False).size().rename("n_folds_completed").reset_index()
    summary = summary.merge(fold_counts, on=group_cols, how="left")
    return summary

fold_results = []
per_label_report_paths = []

experiment_params = {
    "split_seed": SPLIT_SEED,
    "n_folds": N_FOLDS,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "word_window_size": WORD_WINDOW_SIZE,
    "word_window_stride": WORD_WINDOW_STRIDE,
    "max_length": MAX_LENGTH,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": OPTIMIZER_NAME,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "mixed_precision": MIXED_PRECISION,
    "force_trainable_params_fp32": FORCE_TRAINABLE_PARAMS_FP32,
    "best_model_metric": BEST_MODEL_METRIC,
    "best_model_greater_is_better": BEST_MODEL_GREATER_IS_BETTER,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
    "quiet_training": QUIET_TRAINING,
    "trainer_logging_strategy": TRAINER_LOGGING_STRATEGY,
}

with maybe_mlflow_run(
    run_name="token_classification_cv_experiment",
    nested=False,
    tags={"run_level": "experiment", "task": "token_classification"},
):
    log_params_to_mlflow(experiment_params)

    for dataset_name in selected_datasets:
        with maybe_mlflow_run(
            run_name=f"dataset__{safe_name(dataset_name)}",
            nested=True,
            tags={"run_level": "dataset", "dataset_name": dataset_name},
        ):
            log_params_to_mlflow({"dataset_name": dataset_name, **experiment_params})
            mlflow.log_artifact(str(split_plan_paths[dataset_name]), artifact_path="split_plan")

            strategies_for_dataset = selected_strategies_by_dataset[dataset_name]

            for strategy_idx, strategy in enumerate(strategies_for_dataset, start=1):
                qprint("=" * 110)
                qprint(f"Dataset: {dataset_name} | Strategy [{strategy_idx}/{len(strategies_for_dataset)}]: {strategy}")
                qprint("=" * 110)

                with maybe_mlflow_run(
                    run_name=f"strategy__{safe_name(strategy)}",
                    nested=True,
                    tags={
                        "run_level": "strategy",
                        "dataset_name": dataset_name,
                        "strategy": strategy,
                    },
                ):
                    log_params_to_mlflow({
                        "dataset_name": dataset_name,
                        "strategy": strategy,
                        **experiment_params,
                    })

                    for model_idx, model_spec in enumerate(MODEL_REGISTRY, start=1):
                        initialization_mode = model_initialization_mode(model_spec)

                        qprint("-" * 110)
                        qprint(
                            f"Model [{model_idx}/{len(MODEL_REGISTRY)}]: "
                            f"{model_spec.name} ({model_spec.model_name_or_path}) | {initialization_mode}"
                        )
                        qprint("-" * 110)

                        tokenizer = build_tokenizer(model_spec)
                        data_collator = DataCollatorForTokenClassification(
                            tokenizer=tokenizer,
                            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
                        )

                        model_run_rows = []

                        with maybe_mlflow_run(
                            run_name=f"model__{safe_name(model_spec.name)}",
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
                            log_params_to_mlflow({
                                "dataset_name": dataset_name,
                                "strategy": strategy,
                                "model_name": model_spec.name,
                                "model_name_or_path": model_spec.model_name_or_path,
                                "tokenizer_name_or_path": model_spec.tokenizer_name_or_path or model_spec.model_name_or_path,
                                "init_from_pretrained": model_spec.init_from_pretrained,
                                "initialization_mode": initialization_mode,
                                **experiment_params,
                            })

                            for fold_index in range(N_FOLDS):
                                fold_seed = SPLIT_SEED + fold_index
                                set_seed(fold_seed)

                                run_id = (
                                    f"{safe_name(dataset_name)}__"
                                    f"{safe_name(strategy)}__"
                                    f"{safe_name(model_spec.name)}__"
                                    f"fold_{fold_index}"
                                )
                                qprint(f"\nFold {fold_index + 1}/{N_FOLDS}: {run_id}")

                                datasets = make_hf_datasets(dataset_name, strategy, fold_index, tokenizer)

                                model = build_model(model_spec)
                                n_total_params = count_total_parameters(model)
                                n_trainable_params = count_trainable_parameters(model)
                                trainable_ratio = n_trainable_params / n_total_params if n_total_params else 0.0

                                qprint(f"Total parameters: {n_total_params:,}")
                                qprint(f"Trainable parameters: {n_trainable_params:,}")
                                qprint({split_name: len(ds) for split_name, ds in datasets.items()})

                                training_args = make_training_args(run_id=run_id, seed=fold_seed)
                                trainer = build_trainer(
                                    model=model,
                                    training_args=training_args,
                                    datasets=datasets,
                                    tokenizer=tokenizer,
                                    data_collator=data_collator,
                                )

                                with maybe_mlflow_run(
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
                                    log_params_to_mlflow({**experiment_params, **fold_params})

                                    train_result = run_trainer_quietly(trainer.train)
                                    log_trainer_history_to_mlflow(trainer)

                                    # The trainer has loaded the best validation checkpoint at this point
                                    # because load_best_model_at_end=True.
                                    val_metrics = run_trainer_quietly(
                                        trainer.evaluate,
                                        eval_dataset=datasets["validation"],
                                        metric_key_prefix="val",
                                    )
                                    test_output = run_trainer_quietly(
                                        trainer.predict,
                                        test_dataset=datasets["test"],
                                        metric_key_prefix="test",
                                    )
                                    test_metrics = dict(test_output.metrics)

                                    report_df = per_label_report_dataframe(test_output)
                                    report_path = RESULTS_ROOT / f"{run_id}_per_label_test_report.csv"
                                    report_df.to_csv(report_path, index=False)
                                    per_label_report_paths.append(report_path)

                                    best_model_dir = RUNS_ROOT / run_id / "best_model"
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
                                        "word_window_size": WORD_WINDOW_SIZE,
                                        "word_window_stride": WORD_WINDOW_STRIDE,
                                        "max_length": MAX_LENGTH,
                                        "best_validation_metric": best_metric,
                                        "best_model_checkpoint": best_checkpoint,
                                        "best_model_dir": str(best_model_dir),
                                        "per_label_report_path": str(report_path),
                                    }

                                    result_row.update(train_metrics)
                                    result_row.update(val_metrics)
                                    result_row.update(test_metrics)

                                    fold_results.append(result_row)
                                    model_run_rows.append(result_row)

                                    log_metrics_to_mlflow(train_metrics)
                                    log_metrics_to_mlflow(val_metrics)
                                    log_metrics_to_mlflow(test_metrics)
                                    if best_metric is not None and isinstance(best_metric, (int, float, np.integer, np.floating)):
                                        log_metrics_to_mlflow({"best_validation_metric": best_metric})

                                    mlflow.log_artifact(str(report_path), artifact_path="reports")

                                    fold_results_so_far_df = pd.DataFrame(fold_results)
                                    interim_path = RESULTS_ROOT / "strategy_model_fold_results.csv"
                                    fold_results_so_far_df.to_csv(interim_path, index=False)

                                del trainer
                                del model
                                cleanup_after_fold()

                            model_fold_df = pd.DataFrame(model_run_rows)
                            model_summary_df = summarize_cv_results(
                                model_fold_df,
                                group_cols=["dataset_name", "strategy", "model_name"],
                            )
                            model_summary_path = (
                                RESULTS_ROOT
                                / f"{safe_name(dataset_name)}__{safe_name(strategy)}__{safe_name(model_spec.name)}_cv_summary.csv"
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
                                log_metrics_to_mlflow(summary_metrics)

                        del tokenizer
                        cleanup_after_fold()

print("Done.")

fold_results_df = pd.DataFrame(fold_results)

if fold_results_df.empty:
    print("No fold results. Run the training cell first.")
else:
    fold_results_path = RESULTS_ROOT / "strategy_model_fold_results.csv"
    fold_results_df.to_csv(fold_results_path, index=False)

    cv_summary_df = summarize_cv_results(
        fold_results_df,
        group_cols=["dataset_name", "strategy", "model_name", "model_name_or_path"],
    )
    cv_summary_path = RESULTS_ROOT / "strategy_model_cv_summary.csv"
    cv_summary_df.to_csv(cv_summary_path, index=False)

    sort_col = "test_non_o_micro_f1_mean" if "test_non_o_micro_f1_mean" in cv_summary_df.columns else None
    if sort_col:
        cv_summary_df = cv_summary_df.sort_values(sort_col, ascending=False)

    print("Saved fold-level results to:", fold_results_path)
    print("Saved CV summary to:", cv_summary_path)