from __future__ import annotations
import argparse
import ast
import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


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
    n_folds: int = 10
    validation_fraction_of_train: float = 0.1
    sampling: Optional[int] = None
    datasets_to_run: str | list[str] = "all"
    strategies_to_run: str | list[str] = field(default_factory=lambda: ["column_aware"])
    model_registry: list[ModelSpec] = field(
        default_factory=lambda: [ModelSpec("bert-mlsm", "SzegedAI/bert-medium-mlsm")]
    )
    max_length: int = 512
    word_window_size: int = 0
    word_window_stride: int = 0
    tokenizer_stride: int = 0
    tokenization_batch_size: int = 64
    tokenization_num_proc: Optional[int] = None
    drop_chunks_with_no_trainable_tokens: bool = True
    num_train_epochs: float = 20
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-05
    optimizer_name: str = "adamw_torch"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    loss_function: str = "cross_entropy"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    best_model_metric: str = "macro_f1"
    best_model_greater_is_better: bool = True
    early_stopping_patience: Optional[int] = 3
    early_stopping_threshold: float = 0.0
    mixed_precision: str = "auto"
    force_trainable_params_fp32: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    eval_accumulation_steps: Optional[int] = 16
    restore_best_model_in_memory: bool = True
    remove_fold_output_dir: bool = True
    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: Optional[int] = 2
    dataloader_pin_memory: bool = True
    minimum_torch_version: str = "2.6.0"
    minimum_transformers_version: str = "5.1.0"
    minimum_cuda_version: str = "12.4"
    enforce_minimum_versions: bool = True
    quiet_training: bool = True
    suppress_model_load_warnings: bool = True
    suppress_train_stdout_stderr: bool = True
    show_run_progress: bool = True
    transformers_verbosity: str = "error"
    datasets_verbosity: str = "error"
    trainer_logging_strategy: str = "no"
    overwrite_output_dir: bool = True
    dry_run: bool = False
    mlflow_experiment_name: str = field(
        default_factory=lambda: f"serialization_strategies_{int(time.time())}"
    )
    mlflow_db_path: Optional[Path] = None
    mlflow_artifact_root: Optional[Path] = None
    mlflow_tracking_uri: Optional[str] = None
    ignore_label: int = -100

    def __post_init__(self) -> None:
        self.loss_function = str(self.loss_function).lower()
        if self.loss_function not in {"cross_entropy", "focal"}:
            raise ValueError(
                f"loss_function must be 'cross_entropy' or 'focal', got {self.loss_function!r}."
            )
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}.")
        if self.focal_alpha is not None and (not 0.0 <= float(self.focal_alpha) <= 1.0):
            raise ValueError(
                f"focal_alpha must be between 0 and 1 when provided, got {self.focal_alpha}."
            )
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}.")
        if self.word_window_size < 0:
            raise ValueError(
                "word_window_size must be >= 0. Use 0 for automatic shared source-token windows."
            )
        if self.word_window_stride < 0:
            raise ValueError(f"word_window_stride must be >= 0, got {self.word_window_stride}.")
        if self.word_window_size > 0 and self.word_window_stride >= self.word_window_size:
            raise ValueError("word_window_stride must be smaller than word_window_size.")
        if self.tokenizer_stride < 0:
            raise ValueError(f"tokenizer_stride must be >= 0, got {self.tokenizer_stride}.")
        if self.tokenization_batch_size <= 0:
            raise ValueError(
                f"tokenization_batch_size must be > 0, got {self.tokenization_batch_size}."
            )
        if self.dataloader_num_workers < 0:
            raise ValueError(
                f"dataloader_num_workers must be >= 0, got {self.dataloader_num_workers}."
            )
        if self.dataloader_num_workers == 0:
            self.dataloader_persistent_workers = False
            self.dataloader_prefetch_factor = None
        self.project_root = Path(self.project_root).resolve()
        self.processed_datasets_root = Path(
            self.processed_datasets_root or self.project_root / "data" / "processed"
        ).resolve()
        self.runs_root = Path(self.runs_root or self.project_root / "runs").resolve()
        dataset_suffix = (
            "all"
            if self.datasets_to_run == "all"
            else "_".join((str(x) for x in self.datasets_to_run))
        )
        self.results_root = Path(
            self.results_root
            or self.project_root / f"results_{dataset_suffix}_{self.loss_function}"
        ).resolve()
        self.mlflow_db_path = Path(self.mlflow_db_path or self.project_root / "mlflow.db").resolve()
        self.mlflow_artifact_root = Path(
            self.mlflow_artifact_root or self.project_root / "mlartifacts"
        ).resolve()
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = os.environ.get(
                "MLFLOW_TRACKING_URI", f"sqlite:///{self.mlflow_db_path.as_posix()}"
            )

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        data["model_registry"] = [asdict(model) for model in self.model_registry]
        return data


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
        ModelSpec(
            "eurobert_210m",
            "EuroBERT/EuroBERT-210m",
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"trust_remote_code": True},
        ),
        ModelSpec(
            "nomic_bert_2048",
            "nomic-ai/nomic-bert-2048",
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"trust_remote_code": True},
        ),
        ModelSpec("xlm_roberta_large", "FacebookAI/xlm-roberta-base"),
    ]
    return {spec.name: spec for spec in specs}


def copy_model_spec(
    spec: ModelSpec, trust_remote_code: bool = False, init_from_pretrained: Optional[bool] = None
) -> ModelSpec:
    model_kwargs = dict(spec.model_kwargs)
    tokenizer_kwargs = dict(spec.tokenizer_kwargs)
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        tokenizer_kwargs["trust_remote_code"] = True
    return ModelSpec(
        name=spec.name,
        model_name_or_path=spec.model_name_or_path,
        tokenizer_name_or_path=spec.tokenizer_name_or_path,
        init_from_pretrained=(
            spec.init_from_pretrained if init_from_pretrained is None else init_from_pretrained
        ),
        revision=spec.revision,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def model_name_from_path(identifier: str) -> str:
    if "=" in identifier:
        identifier = identifier.split("=", 1)[0]
    return "".join(
        (ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in identifier)
    ).strip("_")


def model_spec_from_identifier(
    identifier: str, trust_remote_code: bool = False, init_from_pretrained: Optional[bool] = None
) -> ModelSpec:
    presets = default_model_presets()
    identifier = identifier.strip()
    if identifier in presets:
        return copy_model_spec(
            presets[identifier],
            trust_remote_code=trust_remote_code,
            init_from_pretrained=init_from_pretrained,
        )
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


def model_specs_from_json(
    value: str, trust_remote_code: bool = False, init_from_pretrained: Optional[bool] = None
) -> list[ModelSpec]:
    raw = load_json_object(value)
    if not isinstance(raw, list):
        raise ValueError("Model config JSON must be a list of model-spec objects.")
    specs = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each model config entry must be an object.")
        spec = ModelSpec(**item)
        specs.append(
            copy_model_spec(
                spec, trust_remote_code=trust_remote_code, init_from_pretrained=init_from_pretrained
            )
        )
    return specs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run serialized-OCR token-classification experiments with configurable datasets, strategies, models, and MLflow logging."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--datasets", type=str, default=None, help="Comma-separated dataset names or 'all'."
    )
    parser.add_argument(
        "--strategies", type=str, default=None, help="Comma-separated strategy names or 'all'."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated preset names or Hugging Face model ids.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Additional model, either name=hf_id or hf_id. Can be repeated.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="JSON string or path to a JSON list of ModelSpec objects.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Initialize selected model architectures from config instead of pretrained weights.",
    )
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
    parser.add_argument("--tokenizer-stride", type=int, default=None)
    parser.add_argument("--tokenization-batch-size", type=int, default=None)
    parser.add_argument("--tokenization-num-proc", type=int, default=None)
    parser.add_argument(
        "--validate-token-coverage", type=str2bool, nargs="?", const=True, default=None
    )
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
    parser.add_argument("--loss-function", choices=["cross_entropy", "focal"], default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Optional focal alpha. If set, non-O labels receive alpha and O receives 1-alpha. Use >0.5 to upweight entity tokens.",
    )
    parser.add_argument(
        "--gradient-checkpointing", type=str2bool, nargs="?", const=True, default=None
    )
    parser.add_argument("--best-model-metric", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--mixed-precision", choices=["auto", "bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--tf32", type=str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--eval-accumulation-steps", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument(
        "--dataloader-persistent-workers", type=str2bool, nargs="?", const=True, default=None
    )
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--dataloader-pin-memory", type=str2bool, nargs="?", const=True, default=None
    )
    parser.add_argument(
        "--enforce-minimum-versions", type=str2bool, nargs="?", const=True, default=None
    )
    parser.add_argument("--quiet", dest="quiet_training", action="store_true", default=None)
    parser.add_argument("--no-quiet", dest="quiet_training", action="store_false")
    parser.add_argument("--show-run-progress", action="store_true")
    parser.add_argument(
        "--trainer-logging-strategy", choices=["no", "steps", "epoch"], default=None
    )
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-artifact-root", type=Path, default=None)
    parser.add_argument("--mlflow-db-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Generic override, e.g. --set max_length=2048 or --set MAX_LENGTH=2048.",
    )
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
    apply_if_not_none(kwargs, "tokenizer_stride", args.tokenizer_stride)
    apply_if_not_none(kwargs, "tokenization_batch_size", args.tokenization_batch_size)
    apply_if_not_none(kwargs, "tokenization_num_proc", args.tokenization_num_proc)
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
    apply_if_not_none(kwargs, "loss_function", args.loss_function)
    apply_if_not_none(kwargs, "focal_gamma", args.focal_gamma)
    apply_if_not_none(kwargs, "focal_alpha", args.focal_alpha)
    apply_if_not_none(kwargs, "gradient_checkpointing", args.gradient_checkpointing)
    apply_if_not_none(kwargs, "best_model_metric", args.best_model_metric)
    apply_if_not_none(kwargs, "early_stopping_patience", args.early_stopping_patience)
    apply_if_not_none(kwargs, "mixed_precision", args.mixed_precision)
    apply_if_not_none(kwargs, "tf32", args.tf32)
    apply_if_not_none(kwargs, "eval_accumulation_steps", args.eval_accumulation_steps)
    apply_if_not_none(kwargs, "dataloader_num_workers", args.dataloader_num_workers)
    apply_if_not_none(kwargs, "dataloader_persistent_workers", args.dataloader_persistent_workers)
    apply_if_not_none(kwargs, "dataloader_prefetch_factor", args.dataloader_prefetch_factor)
    apply_if_not_none(kwargs, "dataloader_pin_memory", args.dataloader_pin_memory)
    apply_if_not_none(kwargs, "enforce_minimum_versions", args.enforce_minimum_versions)
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
        selected_models = model_specs_from_json(
            args.model_config,
            trust_remote_code=args.trust_remote_code,
            init_from_pretrained=init_from_pretrained,
        )
    elif args.models is not None:
        selected_models = [
            model_spec_from_identifier(
                item,
                trust_remote_code=args.trust_remote_code,
                init_from_pretrained=init_from_pretrained,
            )
            for item in parse_selection(args.models)
        ]
    if args.model:
        if selected_models is None:
            selected_models = []
        selected_models.extend(
            (
                model_spec_from_identifier(
                    item,
                    trust_remote_code=args.trust_remote_code,
                    init_from_pretrained=init_from_pretrained,
                )
                for item in args.model
            )
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
    legacy_mapping.update(
        {
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
            "LOSS_FUNCTION": "loss_function",
            "FOCAL_GAMMA": "focal_gamma",
            "FOCAL_ALPHA": "focal_alpha",
            "BEST_MODEL_METRIC": "best_model_metric",
            "EARLY_STOPPING_PATIENCE": "early_stopping_patience",
            "MIXED_PRECISION": "mixed_precision",
            "QUIET_TRAINING": "quiet_training",
            "MLFLOW_EXPERIMENT_NAME": "mlflow_experiment_name",
            "MLFLOW_TRACKING_URI": "mlflow_tracking_uri",
        }
    )
    path_fields = {
        "project_root",
        "processed_datasets_root",
        "runs_root",
        "results_root",
        "mlflow_db_path",
        "mlflow_artifact_root",
    }
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set override {item!r}. Use name=value.")
        raw_key, raw_value = item.split("=", 1)
        raw_key = raw_key.strip()
        key = legacy_mapping.get(raw_key, raw_key)
        if key not in valid_fields:
            raise ValueError(
                f"Unknown config field {raw_key!r}. Valid fields include: {sorted(valid_fields)}"
            )
        value = parse_scalar(raw_value)
        if key in path_fields and value is not None:
            value = Path(value).resolve()
        setattr(config, key, value)
    config.__post_init__()
