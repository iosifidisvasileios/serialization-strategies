from __future__ import annotations
import gc
import inspect
import json
import logging
import os
import shutil
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from mlflow.tracking import MlflowClient
from packaging.version import InvalidVersion, Version
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as transformers_logging
from data_pipeline import DataPipeline, ExperimentData, TokenizedCorpus
from experiment_config import ExperimentConfig, ModelSpec

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


def focal_loss_for_token_classification(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
    alpha: Optional[float],
    ignore_index: int,
    o_label_id: Optional[int],
) -> torch.Tensor:
    """Focal loss for token classification with ignored labels.

    alpha is optional. When provided, it is applied in a binary-style way:
    non-O labels receive alpha and O labels receive 1-alpha. For imbalanced
    BIO tagging where O dominates, use alpha > 0.5 to upweight entity tokens.
    """
    num_labels = logits.shape[-1]
    flat_logits = logits.reshape(-1, num_labels)
    flat_labels = labels.reshape(-1)
    active_mask = flat_labels != ignore_index
    if not torch.any(active_mask):
        return flat_logits.sum() * 0.0
    active_logits = flat_logits[active_mask]
    active_labels = flat_labels[active_mask]
    ce_loss = F.cross_entropy(active_logits, active_labels, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (1.0 - pt) ** gamma * ce_loss
    if alpha is not None:
        alpha_value = float(alpha)
        if o_label_id is None:
            alpha_t = torch.full_like(loss, alpha_value)
        else:
            alpha_t = torch.where(
                active_labels == int(o_label_id),
                torch.full_like(loss, 1.0 - alpha_value),
                torch.full_like(loss, alpha_value),
            )
        loss = alpha_t * loss
    return loss.mean()


class FocalLossTrainer(Trainer):
    """Trainer variant that replaces model cross-entropy with focal loss."""

    def __init__(
        self,
        *args,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        ignore_index: int = -100,
        o_label_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ignore_index = ignore_index
        self.o_label_id = o_label_id
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None, **kwargs
    ):
        del num_items_in_batch, kwargs
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        if labels is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        else:
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = focal_loss_for_token_classification(
                logits,
                labels,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
                ignore_index=self.ignore_index,
                o_label_id=self.o_label_id,
            )
        return (loss, outputs) if return_outputs else loss


class InMemoryBestModelCallback(TrainerCallback):
    """Track and restore the best validation model without writing checkpoints.

    Only model weights and buffers are copied to CPU memory. Optimizer and
    scheduler state are never stored. The retained state is released as soon as
    training ends and the best weights have been restored to the live model.
    """

    def __init__(
        self,
        *,
        metric_name: str,
        greater_is_better: bool,
        patience: Optional[int],
        threshold: float,
        restore_best_model: bool = True,
    ) -> None:
        self.metric_name = str(metric_name)
        self.greater_is_better = bool(greater_is_better)
        self.patience = patience
        self.threshold = float(threshold)
        self.restore_best_model = bool(restore_best_model)
        self.best_metric: Optional[float] = None
        self.best_global_step: Optional[int] = None
        self.best_epoch: Optional[float] = None
        self.bad_evaluation_count = 0
        self.best_state_dict: Optional[dict[str, torch.Tensor]] = None
        self.training_active = False
        self.restored_best_model = False

    def _metric_key(self, metrics: dict[str, Any]) -> Optional[str]:
        candidates = [self.metric_name]
        if not self.metric_name.startswith("eval_"):
            candidates.insert(0, f"eval_{self.metric_name}")
        for candidate in candidates:
            if candidate in metrics:
                return candidate
        return None

    def _is_improvement(self, metric_value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return metric_value > self.best_metric + self.threshold
        return metric_value < self.best_metric - self.threshold

    @staticmethod
    def _copy_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().to(device="cpu", copy=True)
            for name, tensor in model.state_dict().items()
        }

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.best_metric = None
        self.best_global_step = None
        self.best_epoch = None
        self.bad_evaluation_count = 0
        self.best_state_dict = None
        self.training_active = True
        self.restored_best_model = False
        return control

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        del args, kwargs
        if not self.training_active:
            return control
        if metrics is None or model is None:
            return control
        metric_key = self._metric_key(metrics)
        if metric_key is None:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"Best-model metric {self.metric_name!r} was not found after evaluation. Available metrics: {available}"
            )
        metric_value = float(metrics[metric_key])
        if self._is_improvement(metric_value):
            self.best_state_dict = self._copy_state_dict_to_cpu(model)
            self.best_metric = metric_value
            self.best_global_step = int(state.global_step)
            self.best_epoch = float(state.epoch) if state.epoch is not None else None
            self.bad_evaluation_count = 0
            state.best_metric = metric_value
            state.best_global_step = int(state.global_step)
            state.best_model_checkpoint = None
        else:
            self.bad_evaluation_count += 1
            if self.patience is not None and self.bad_evaluation_count >= self.patience:
                control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        del args, kwargs
        if self.restore_best_model and model is not None and (self.best_state_dict is not None):
            model.load_state_dict(self.best_state_dict, strict=True)
            self.restored_best_model = True
        state.best_metric = self.best_metric
        if self.best_global_step is not None:
            state.best_global_step = self.best_global_step
        state.best_model_checkpoint = None
        self.best_state_dict = None
        self.training_active = False
        gc.collect()
        return control


@dataclass
class TrainingState:
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    per_label_report_paths: list[Path] = field(default_factory=list)


class ExperimentRunner:
    """Orchestrate model training, evaluation, tracking, and result aggregation."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.pipeline = DataPipeline(config)
        self.state = TrainingState()
        self.runtime_info: dict[str, Any] = {}

    @property
    def data(self) -> ExperimentData:
        return self.pipeline.data

    def qprint(self, *args, **kwargs) -> None:
        if self.cfg.show_run_progress:
            print(*args, **kwargs)

    @staticmethod
    def _version_at_least(actual: str, minimum: str) -> bool:
        try:
            return Version(str(actual).split("+")[0]) >= Version(minimum)
        except InvalidVersion as exc:
            raise RuntimeError(f"Could not parse version {actual!r}.") from exc

    def validate_runtime(self) -> dict[str, Any]:
        runtime = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": None,
            "gpu_total_memory_gib": None,
            "gpu_compute_capability": None,
        }
        problems = []
        if not self._version_at_least(torch.__version__, self.cfg.minimum_torch_version):
            problems.append(
                f"torch>={self.cfg.minimum_torch_version} is required; found {torch.__version__}."
            )
        if not self._version_at_least(
            transformers.__version__, self.cfg.minimum_transformers_version
        ):
            problems.append(
                f"transformers>={self.cfg.minimum_transformers_version} is required; found {transformers.__version__}."
            )
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            runtime["gpu_name"] = properties.name
            runtime["gpu_total_memory_gib"] = properties.total_memory / 1024**3
            runtime["gpu_compute_capability"] = ".".join(
                (str(x) for x in torch.cuda.get_device_capability(device))
            )
            if torch.version.cuda is None:
                problems.append("CUDA is available but torch.version.cuda is missing.")
            elif not self._version_at_least(torch.version.cuda, self.cfg.minimum_cuda_version):
                problems.append(
                    f"A CUDA {self.cfg.minimum_cuda_version}+ PyTorch build is required; found CUDA {torch.version.cuda}."
                )
        else:
            self.qprint("CUDA is not available; the run will execute on CPU.")
        if problems:
            message = "Runtime compatibility check failed:\n- " + "\n- ".join(problems)
            if self.cfg.enforce_minimum_versions:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)
        return runtime

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
        return (np.array(y_true), np.array(y_pred))

    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        del labels
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

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
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        if self.data.non_o_label_ids:
            non_o_p, non_o_r, non_o_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=self.data.non_o_label_ids, average="micro", zero_division=0
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
        y_true, y_pred = self.flatten_predictions_and_labels(
            pred_output.predictions, pred_output.label_ids
        )
        labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        target_names = [self.data.id2label[i] for i in labels_present]
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
        return "".join((ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text))

    def select_mixed_precision_flags(self) -> dict[str, bool]:
        if not torch.cuda.is_available() or self.cfg.mixed_precision == "fp32":
            return {"fp16": False, "bf16": False}
        requested = str(self.cfg.mixed_precision).lower()
        if requested == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "mixed_precision='bf16' was requested, but this CUDA device does not support BF16."
                )
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
        return (
            "fine_tuning_from_pretrained"
            if model_spec.init_from_pretrained
            else "training_from_scratch"
        )

    def force_trainable_parameters_fp32(self, model) -> None:
        if not self.cfg.force_trainable_params_fp32:
            return
        converted = 0
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16}:
                parameter.data = parameter.data.float()
                converted += parameter.numel()
        if converted:
            self.qprint(
                f"Converted {converted:,} trainable parameters to FP32 for AMP-compatible training."
            )

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
            "num_labels": len(self.data.label_list),
            "id2label": self.data.id2label,
            "label2id": self.data.label2id,
            **dict(model_spec.model_kwargs),
        }
        if model_spec.revision is not None:
            common_kwargs["revision"] = model_spec.revision
        with self.quiet_section(self.cfg.quiet_training and self.cfg.suppress_model_load_warnings):
            if model_spec.init_from_pretrained:
                model = AutoModelForTokenClassification.from_pretrained(
                    model_spec.model_name_or_path, **common_kwargs
                )
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
        return sum((p.numel() for p in model.parameters()))

    @staticmethod
    def count_trainable_parameters(model) -> int:
        return sum((p.numel() for p in model.parameters() if p.requires_grad))

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
            "save_strategy": "no",
            "logging_strategy": self.cfg.trainer_logging_strategy,
            "logging_steps": 50,
            "logging_first_step": False,
            "load_best_model_at_end": False,
            "metric_for_best_model": self.cfg.best_model_metric,
            "greater_is_better": self.cfg.best_model_greater_is_better,
            "report_to": "none",
            "run_name": run_id,
            "seed": seed,
            "data_seed": seed,
            "fp16": precision_flags["fp16"],
            "bf16": precision_flags["bf16"],
            "tf32": self.cfg.tf32
            and torch.cuda.is_available()
            and (torch.cuda.get_device_capability()[0] >= 8),
            "gradient_checkpointing": self.cfg.gradient_checkpointing,
            "eval_accumulation_steps": self.cfg.eval_accumulation_steps,
            "dataloader_num_workers": self.cfg.dataloader_num_workers,
            "dataloader_persistent_workers": self.cfg.dataloader_persistent_workers,
            "dataloader_prefetch_factor": self.cfg.dataloader_prefetch_factor,
            "dataloader_pin_memory": self.cfg.dataloader_pin_memory,
            "remove_unused_columns": True,
            "include_num_input_tokens_seen": True,
            "disable_tqdm": self.cfg.quiet_training,
            "log_level": self.cfg.transformers_verbosity,
            "log_level_replica": self.cfg.transformers_verbosity,
        }
        sig = inspect.signature(TrainingArguments.__init__)
        params = set(sig.parameters)
        if "eval_strategy" in params:
            candidate_kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in params:
            candidate_kwargs["evaluation_strategy"] = "epoch"
        if "warmup_ratio" not in params and "warmup_steps" in params:
            candidate_kwargs["warmup_steps"] = 0
        filtered_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in params and value is not None
        }
        ignored_kwargs = sorted(set(candidate_kwargs).difference(filtered_kwargs))
        if ignored_kwargs:
            self.qprint(
                f"{run_id}: ignored unsupported or unset TrainingArguments kwargs: {ignored_kwargs}"
            )
        return TrainingArguments(**filtered_kwargs)

    def build_trainer(
        self, model, training_args, datasets: dict[str, Dataset], tokenizer, data_collator
    ):
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": datasets["train"],
            "eval_dataset": datasets["validation"],
            "data_collator": data_collator,
            "compute_metrics": self.compute_metrics,
            "preprocess_logits_for_metrics": self.preprocess_logits_for_metrics,
        }
        callbacks = [
            InMemoryBestModelCallback(
                metric_name=self.cfg.best_model_metric,
                greater_is_better=self.cfg.best_model_greater_is_better,
                patience=self.cfg.early_stopping_patience,
                threshold=self.cfg.early_stopping_threshold,
                restore_best_model=self.cfg.restore_best_model_in_memory,
            )
        ]
        trainer_kwargs["callbacks"] = callbacks
        trainer_sig = inspect.signature(Trainer.__init__)
        trainer_params = set(trainer_sig.parameters)
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        if self.cfg.loss_function == "focal":
            trainer = FocalLossTrainer(
                **trainer_kwargs,
                focal_gamma=self.cfg.focal_gamma,
                focal_alpha=self.cfg.focal_alpha,
                ignore_index=self.cfg.ignore_label,
                o_label_id=self.data.label2id.get("O"),
            )
        else:
            trainer = Trainer(**trainer_kwargs)
        return self.remove_notebook_progress_callbacks(trainer)

    def remove_notebook_progress_callbacks(self, trainer: Trainer) -> Trainer:
        callbacks = list(getattr(trainer.callback_handler, "callbacks", []))
        kept_callbacks = [
            cb for cb in callbacks if cb.__class__.__name__ != "NotebookProgressCallback"
        ]
        removed = len(callbacks) - len(kept_callbacks)
        if removed:
            self.qprint(f"Removed {removed} NotebookProgressCallback instance(s).")
        trainer.callback_handler.callbacks = kept_callbacks
        return trainer

    def run_trainer_quietly(self, fn, *args, **kwargs):
        with self.quiet_section(self.cfg.quiet_training and self.cfg.suppress_train_stdout_stderr):
            return fn(*args, **kwargs)

    @staticmethod
    def reset_cuda_peak_memory() -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def cuda_memory_metrics(prefix: str) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize()
        return {
            f"{prefix}_peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            f"{prefix}_peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            f"{prefix}_current_gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            f"{prefix}_current_gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        }

    @staticmethod
    def remove_trainer_output_dir(output_dir: Path) -> bool:
        if not output_dir.exists():
            return False
        shutil.rmtree(output_dir)
        return True

    @staticmethod
    def cleanup_after_fold() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def maybe_mlflow_run(
        self, run_name: str, nested: bool = False, tags: Optional[dict[str, Any]] = None
    ):
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
    def log_metrics_to_mlflow(
        metrics: dict[str, Any], prefix: str = "", step: Optional[int] = None
    ) -> None:
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
                if k not in {"step"}
                and isinstance(v, (int, float, np.integer, np.floating))
                and np.isfinite(v)
            }
            self.log_metrics_to_mlflow(
                metrics, prefix=prefix, step=int(step) if step is not None else None
            )

    @staticmethod
    def summarize_cv_results(fold_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if fold_df.empty:
            return pd.DataFrame()
        candidate_cols = [
            c
            for c in fold_df.columns
            if c.startswith("test_")
            or c.startswith("val_")
            or c.startswith("train_")
            or (
                c
                in {
                    "best_validation_metric",
                    "trainable_parameters",
                    "total_parameters",
                    "trainable_parameter_ratio",
                }
            )
        ]
        metric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(fold_df[c])]
        summary = (
            fold_df.groupby(group_cols, dropna=False)[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            (
                "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
                if isinstance(col, tuple)
                else str(col)
            )
            for col in summary.columns
        ]
        fold_counts = (
            fold_df.groupby(group_cols, dropna=False)
            .size()
            .rename("n_folds_completed")
            .reset_index()
        )
        return summary.merge(fold_counts, on=group_cols, how="left")

    def experiment_params(self) -> dict[str, Any]:
        return {
            "split_seed": self.cfg.split_seed,
            "n_folds": self.cfg.n_folds,
            "outer_test_fraction_per_fold": 1.0 / self.cfg.n_folds,
            "validation_fraction_of_train": self.cfg.validation_fraction_of_train,
            "word_window_size": self.cfg.word_window_size,
            "word_window_stride": self.cfg.word_window_stride,
            "tokenizer_stride": self.cfg.tokenizer_stride,
            "tokenization_batch_size": self.cfg.tokenization_batch_size,
            "tokenization_num_proc": self.cfg.tokenization_num_proc,
            "max_length": self.cfg.max_length,
            "num_train_epochs": self.cfg.num_train_epochs,
            "per_device_train_batch_size": self.cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": self.cfg.per_device_eval_batch_size,
            "effective_train_batch_size_per_process": self.cfg.per_device_train_batch_size
            * self.cfg.gradient_accumulation_steps,
            "loss_function": self.cfg.loss_function,
            "focal_gamma": self.cfg.focal_gamma,
            "focal_alpha": self.cfg.focal_alpha,
            "learning_rate": self.cfg.learning_rate,
            "optimizer": self.cfg.optimizer_name,
            "weight_decay": self.cfg.weight_decay,
            "warmup_ratio": self.cfg.warmup_ratio,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "gradient_checkpointing": self.cfg.gradient_checkpointing,
            "mixed_precision": self.cfg.mixed_precision,
            "force_trainable_params_fp32": self.cfg.force_trainable_params_fp32,
            "tf32": self.cfg.tf32,
            "eval_accumulation_steps": self.cfg.eval_accumulation_steps,
            "checkpoint_storage_enabled": False,
            "final_model_storage_enabled": False,
            "restore_best_model_in_memory": self.cfg.restore_best_model_in_memory,
            "remove_fold_output_dir": self.cfg.remove_fold_output_dir,
            "dataloader_num_workers": self.cfg.dataloader_num_workers,
            "dataloader_persistent_workers": self.cfg.dataloader_persistent_workers,
            "dataloader_prefetch_factor": self.cfg.dataloader_prefetch_factor,
            "best_model_metric": self.cfg.best_model_metric,
            "best_model_greater_is_better": self.cfg.best_model_greater_is_better,
            "early_stopping_patience": self.cfg.early_stopping_patience,
            "early_stopping_threshold": self.cfg.early_stopping_threshold,
            "quiet_training": self.cfg.quiet_training,
            "trainer_logging_strategy": self.cfg.trainer_logging_strategy,
        }

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
        sort_col = (
            "test_non_o_micro_f1_mean"
            if "test_non_o_micro_f1_mean" in cv_summary_df.columns
            else None
        )
        if sort_col:
            cv_summary_df = cv_summary_df.sort_values(sort_col, ascending=False)
        print("Saved fold-level results to:", fold_results_path)
        print("Saved CV summary to:", cv_summary_path)
        return cv_summary_df

    def prepare(self) -> None:
        self.setup_paths()
        self.configure_quiet_output()
        self.runtime_info = self.validate_runtime()
        print(json.dumps(self.runtime_info, indent=2, default=str))
        print(json.dumps(self.cfg.to_serializable_dict(), indent=2, default=str))
        runtime_path = self.cfg.results_root / "runtime_environment.json"
        runtime_path.write_text(
            json.dumps(self.runtime_info, indent=2, default=str), encoding="utf-8"
        )
        self.pipeline.prepare()

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
        config_path.write_text(
            json.dumps(self.cfg.to_serializable_dict(), indent=2, default=str), encoding="utf-8"
        )
        runtime_path = self.cfg.results_root / "runtime_environment.json"
        with self.maybe_mlflow_run(
            run_name="token_classification_cv_experiment",
            nested=False,
            tags={"run_level": "experiment", "task": "token_classification"},
        ):
            self.log_params_to_mlflow({**experiment_params, **self.runtime_info})
            mlflow.log_artifact(str(config_path), artifact_path="config")
            mlflow.log_artifact(str(runtime_path), artifact_path="config")
            for dataset_name in self.data.selected_datasets:
                self._run_dataset(dataset_name, experiment_params)
        return self.finalize_results()

    def _run_dataset(self, dataset_name: str, experiment_params: dict[str, Any]) -> None:
        with self.maybe_mlflow_run(
            run_name=f"dataset__{self.safe_name(dataset_name)}",
            nested=True,
            tags={"run_level": "dataset", "dataset_name": dataset_name},
        ):
            self.log_params_to_mlflow({"dataset_name": dataset_name, **experiment_params})
            mlflow.log_artifact(
                str(self.data.split_plan_paths[dataset_name]), artifact_path="split_plan"
            )
            strategies = self.data.selected_strategies_by_dataset[dataset_name]
            for strategy_index, strategy in enumerate(strategies, start=1):
                self._run_strategy(
                    dataset_name=dataset_name,
                    strategy=strategy,
                    strategy_index=strategy_index,
                    strategy_count=len(strategies),
                    experiment_params=experiment_params,
                )

    def _run_strategy(
        self,
        *,
        dataset_name: str,
        strategy: str,
        strategy_index: int,
        strategy_count: int,
        experiment_params: dict[str, Any],
    ) -> None:
        self.qprint("=" * 110)
        self.qprint(
            f"Dataset: {dataset_name} | Strategy [{strategy_index}/{strategy_count}]: {strategy}"
        )
        self.qprint("=" * 110)
        with self.maybe_mlflow_run(
            run_name=f"strategy__{self.safe_name(strategy)}",
            nested=True,
            tags={"run_level": "strategy", "dataset_name": dataset_name, "strategy": strategy},
        ):
            self.log_params_to_mlflow(
                {"dataset_name": dataset_name, "strategy": strategy, **experiment_params}
            )
            for model_index, model_spec in enumerate(self.cfg.model_registry, start=1):
                self._run_model(
                    dataset_name=dataset_name,
                    strategy=strategy,
                    model_spec=model_spec,
                    model_index=model_index,
                    model_count=len(self.cfg.model_registry),
                    experiment_params=experiment_params,
                )

    def _run_model(
        self,
        *,
        dataset_name: str,
        strategy: str,
        model_spec: ModelSpec,
        model_index: int,
        model_count: int,
        experiment_params: dict[str, Any],
    ) -> None:
        initialization_mode = self.model_initialization_mode(model_spec)
        self.qprint("-" * 110)
        self.qprint(
            f"Model [{model_index}/{model_count}]: {model_spec.name} ({model_spec.model_name_or_path}) | {initialization_mode}"
        )
        self.qprint("-" * 110)
        tokenizer = self.pipeline.build_tokenizer(model_spec)
        tokenized_corpus = self.pipeline.tokenize_dataset_strategy_once(
            dataset_name=dataset_name, strategy=strategy, model_spec=model_spec, tokenizer=tokenizer
        )
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        model_run_rows: list[dict[str, Any]] = []
        try:
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
                self.log_params_to_mlflow(
                    {
                        "dataset_name": dataset_name,
                        "strategy": strategy,
                        "model_name": model_spec.name,
                        "model_name_or_path": model_spec.model_name_or_path,
                        "tokenizer_name_or_path": model_spec.tokenizer_name_or_path
                        or model_spec.model_name_or_path,
                        "init_from_pretrained": model_spec.init_from_pretrained,
                        "initialization_mode": initialization_mode,
                        **tokenized_corpus.summary,
                        **experiment_params,
                    }
                )
                tokenization_summary_path = self.cfg.results_root / "tokenization_summary.csv"
                if tokenization_summary_path.exists():
                    mlflow.log_artifact(
                        str(tokenization_summary_path), artifact_path="tokenization"
                    )
                for fold_index in range(self.cfg.n_folds):
                    result_row = self._run_fold(
                        dataset_name=dataset_name,
                        strategy=strategy,
                        model_spec=model_spec,
                        initialization_mode=initialization_mode,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        tokenized_corpus=tokenized_corpus,
                        fold_index=fold_index,
                        experiment_params=experiment_params,
                    )
                    self.state.fold_results.append(result_row)
                    model_run_rows.append(result_row)
                    pd.DataFrame(self.state.fold_results).to_csv(
                        self.cfg.results_root / "strategy_model_fold_results.csv", index=False
                    )
                self._save_model_summary(
                    dataset_name=dataset_name,
                    strategy=strategy,
                    model_spec=model_spec,
                    model_run_rows=model_run_rows,
                )
        finally:
            del tokenized_corpus
            del tokenizer
            del data_collator
            self.cleanup_after_fold()

    def _run_fold(
        self,
        *,
        dataset_name: str,
        strategy: str,
        model_spec: ModelSpec,
        initialization_mode: str,
        tokenizer,
        data_collator,
        tokenized_corpus: TokenizedCorpus,
        fold_index: int,
        experiment_params: dict[str, Any],
    ) -> dict[str, Any]:
        fold_seed = self.cfg.split_seed + fold_index
        set_seed(fold_seed)
        run_id = f"{self.safe_name(dataset_name)}__{self.safe_name(strategy)}__{self.safe_name(model_spec.name)}__fold_{fold_index}"
        self.qprint(f"\nFold {fold_index + 1}/{self.cfg.n_folds}: {run_id}")
        fold_datasets = self.pipeline.make_fold_datasets_from_tokenized(
            dataset_name=dataset_name,
            strategy=strategy,
            fold_index=fold_index,
            corpus=tokenized_corpus,
        )
        model = self.build_model(model_spec)
        total_parameters = self.count_total_parameters(model)
        trainable_parameters = self.count_trainable_parameters(model)
        trainable_ratio = trainable_parameters / total_parameters if total_parameters else 0.0
        training_args = self.make_training_args(run_id=run_id, seed=fold_seed)
        trainer = self.build_trainer(model, training_args, fold_datasets, tokenizer, data_collator)
        try:
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
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "trainable_parameter_ratio": trainable_ratio,
                    "n_train_chunks": len(fold_datasets["train"]),
                    "n_validation_chunks": len(fold_datasets["validation"]),
                    "n_test_chunks": len(fold_datasets["test"]),
                    "effective_max_length": tokenized_corpus.summary["effective_max_length"],
                }
                self.log_params_to_mlflow({**experiment_params, **fold_params})
                train_result, train_gpu_metrics = self._train_fold(trainer)
                val_metrics, val_gpu_metrics = self._evaluate_validation(
                    trainer, fold_datasets["validation"]
                )
                test_output, test_metrics, test_gpu_metrics = self._evaluate_test(
                    trainer, fold_datasets["test"]
                )
                report_path = self._save_per_label_report(run_id, test_output)
                best_model_info = self._best_model_info(trainer)
                gpu_metrics = {**train_gpu_metrics, **val_gpu_metrics, **test_gpu_metrics}
                result_row = {
                    **fold_params,
                    "word_window_size": self.cfg.word_window_size,
                    "word_window_stride": self.cfg.word_window_stride,
                    "tokenizer_stride": self.cfg.tokenizer_stride,
                    "max_length": self.cfg.max_length,
                    **best_model_info,
                    "checkpoint_storage_enabled": False,
                    "final_model_storage_enabled": False,
                    "best_model_dir": None,
                    "per_label_report_path": str(report_path),
                    **gpu_metrics,
                    **dict(train_result.metrics),
                    **val_metrics,
                    **test_metrics,
                }
                self._log_fold_outputs(
                    train_metrics=dict(train_result.metrics),
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    gpu_metrics=gpu_metrics,
                    best_metric=best_model_info["best_validation_metric"],
                    report_path=report_path,
                )
                return result_row
        finally:
            fold_output_dir = Path(training_args.output_dir)
            del trainer
            del model
            del fold_datasets
            del training_args
            self.cleanup_after_fold()
            if self.cfg.remove_fold_output_dir:
                self.remove_trainer_output_dir(fold_output_dir)

    def _train_fold(self, trainer: Trainer):
        self.reset_cuda_peak_memory()
        train_result = self.run_trainer_quietly(trainer.train)
        gpu_metrics = self.cuda_memory_metrics("train")
        self.log_trainer_history_to_mlflow(trainer)
        return (train_result, gpu_metrics)

    def _evaluate_validation(self, trainer: Trainer, validation_dataset: Dataset):
        self.reset_cuda_peak_memory()
        metrics = self.run_trainer_quietly(
            trainer.evaluate, eval_dataset=validation_dataset, metric_key_prefix="val"
        )
        return (metrics, self.cuda_memory_metrics("validation"))

    def _evaluate_test(self, trainer: Trainer, test_dataset: Dataset):
        self.reset_cuda_peak_memory()
        output = self.run_trainer_quietly(
            trainer.predict, test_dataset=test_dataset, metric_key_prefix="test"
        )
        return (output, dict(output.metrics), self.cuda_memory_metrics("test"))

    def _save_per_label_report(self, run_id: str, test_output) -> Path:
        report_df = self.per_label_report_dataframe(test_output)
        report_path = self.cfg.results_root / f"{run_id}_per_label_test_report.csv"
        report_df.to_csv(report_path, index=False)
        self.state.per_label_report_paths.append(report_path)
        return report_path

    @staticmethod
    def _best_model_info(trainer: Trainer) -> dict[str, Any]:
        best_callback = next(
            (
                callback
                for callback in trainer.callback_handler.callbacks
                if isinstance(callback, InMemoryBestModelCallback)
            ),
            None,
        )
        return {
            "best_validation_metric": getattr(trainer.state, "best_metric", None),
            "best_model_checkpoint_source": None,
            "best_model_global_step": getattr(trainer.state, "best_global_step", None),
            "best_model_epoch": best_callback.best_epoch if best_callback is not None else None,
            "best_model_restored_from_memory": bool(
                best_callback is not None and best_callback.restored_best_model
            ),
        }

    def _log_fold_outputs(
        self,
        *,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        gpu_metrics: dict[str, Any],
        best_metric: Any,
        report_path: Path,
    ) -> None:
        self.log_metrics_to_mlflow(train_metrics)
        self.log_metrics_to_mlflow(val_metrics)
        self.log_metrics_to_mlflow(test_metrics)
        self.log_metrics_to_mlflow(gpu_metrics)
        if isinstance(best_metric, (int, float, np.integer, np.floating)):
            self.log_metrics_to_mlflow({"best_validation_metric": best_metric})
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    def _save_model_summary(
        self,
        *,
        dataset_name: str,
        strategy: str,
        model_spec: ModelSpec,
        model_run_rows: list[dict[str, Any]],
    ) -> None:
        model_fold_df = pd.DataFrame(model_run_rows)
        model_summary_df = self.summarize_cv_results(
            model_fold_df, group_cols=["dataset_name", "strategy", "model_name"]
        )
        model_summary_path = (
            self.cfg.results_root
            / f"{self.safe_name(dataset_name)}__{self.safe_name(strategy)}__{self.safe_name(model_spec.name)}_cv_summary.csv"
        )
        model_summary_df.to_csv(model_summary_path, index=False)
        mlflow.log_artifact(str(model_summary_path), artifact_path="summaries")
        if model_summary_df.empty:
            return
        summary_row = model_summary_df.iloc[0].to_dict()
        summary_metrics = {
            f"cv_{key}": value
            for key, value in summary_row.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)
        }
        self.log_metrics_to_mlflow(summary_metrics)
