# Serialization Strategies for Document Understanding

This project evaluates different serialization strategies for token classification tasks on document understanding datasets. It provides a systematic framework for comparing how different ways of serializing OCR tokens and layout information affect model performance.

## Overview

The main experiment runner (`src/training/execute_experiment_new.py`) evaluates each serialization method on each dataset using cross-validation with transformer models. The framework supports:

- **Multiple serialization strategies** for converting structured document OCR data into token sequences
- **Cross-validation evaluation** with train/validation/test splits
- **MLflow tracking** for experiment reproducibility and analysis
- **Multiple document datasets** including charity reports, FCC invoices, NDAs, ad buy forms, multi-documents, and registration forms
- **Configurable transformer models** from HuggingFace

## Project Structure

```
serialization-strategies/
├── run_eval.sh                                    # Evaluation script
├── dataset_serialization_RealKIE.ipynb           # RealKIE dataset notebook
├── dataset_serialization_VRDU.ipynb              # VRDU dataset notebook
├── dataset_serialization_multitype.ipynb         # Multi-type dataset notebook
├── LICENSE                                        # Project license
├── pyproject.toml                                # Project dependencies
├── src/
│   ├── datasets/                  # Dataset loaders
│   │   ├── base_loader.py        # Base dataset loader class
│   │   ├── charity_loader.py     # Charity report dataset
│   │   ├── fcc_invoice_loader.py # FCC invoice dataset
│   │   ├── nda_loader.py         # NDA dataset
│   │   ├── resource_contract_loader.py
│   │   └── sec_s1_loader.py      # SEC S1 filing dataset
│   ├── preprocessing/             # Preprocessing utilities
│   │   ├── alignment.py          # Annotation-token alignment
│   │   ├── label_utils.py        # Label processing utilities
│   │   ├── ocr_utils.py          # OCR processing functions
│   │   ├── quality.py            # Data quality checks
│   │   ├── schema.py             # Data schemas
│   │   └── value_normalization.py
│   ├── serialization/            # Serialization strategies
│   │   ├── base.py               # Base serializer class
│   │   ├── plain_text.py         # Plain text serialization
│   │   ├── page_aware.py         # Page-aware serialization
│   │   ├── block_aware.py        # Block-aware serialization
│   │   ├── line_aware.py         # Line-aware serialization
│   │   ├── column_aware.py       # Column-aware serialization
│   │   ├── bbox_token.py         # Bounding box token serialization
│   │   ├── xycut_aware.py        # XYCut-aware serialization
│   │   ├── rowcol_bucket.py      # Row/column bucket serialization
│   │   ├── lmdx_coord_suffix.py  # LayoutMDX coordinate suffix
│   │   ├── compact_bbox_token.py # Compact bbox token
│   │   └── t5_json.py            # T5 JSON serialization
│   └── training/                  # Training and experiment orchestration
│       ├── execute_experiment_new.py  # Main experiment runner
│       ├── experiment_config.py       # Configuration management
│       ├── training_engine.py         # Training engine with MLflow integration
│       └── data_pipeline.py           # Data loading and preprocessing pipeline
└── .gitignore                    # Git ignore rules
```

## Main Components

### Experiment Runner (`src/training/execute_experiment_new.py`)

The experiment runner orchestrates the evaluation of serialization strategies:

- **Dataset discovery**: Automatically discovers processed datasets from `data/processed/`
- **Cross-validation**: Performs N-fold cross-validation with stratified splits
- **Model training**: Trains transformer models using HuggingFace Trainer
- **Evaluation**: Computes comprehensive metrics including accuracy, F1 scores, and per-label reports
- **MLflow logging**: Tracks all experiments, parameters, and results

Key configuration options (via command-line arguments or `ExperimentConfig`):

```python
# Dataset and strategy selection
datasets_to_run = "all"  # or specific dataset names
strategies_to_run = ["column_aware"]  # or "all"

# Cross-validation settings
n_folds = 5
split_seed = 42
validation_fraction_of_train = 0.10

# Model configuration
model_registry = [
    ModelSpec("bert-mlsm", "SzegedAI/bert-medium-mlsm"),
    # Add more models here
]

# Training hyperparameters
max_length = 512
word_window_size = 384
num_train_epochs = 20
learning_rate = 5e-5
per_device_train_batch_size = 8
```

### Serialization Strategies

The project implements multiple serialization strategies in `src/serialization/`:

- **plain_text**: Simple token sequence without layout information
- **page_aware**: Adds page markers (`[PAGE_0]`, `[PAGE_1]`, etc.)
- **block_aware**: Groups tokens by detected blocks/regions
- **line_aware**: Preserves line structure within serialization
- **column_aware**: Detects and serializes by visual columns (deterministic reading order)
- **bbox_token**: Embeds bounding box coordinates as special tokens
- **xycut_aware**: Uses XYCut algorithm for layout-aware serialization
- **rowcol_bucket**: Serializes using row/column bucket encoding
- **lmdx_coord_suffix**: LayoutMDX-style coordinate suffixes
- **compact_bbox_token**: Compact representation of bounding box information
- **t5_json**: JSON-based serialization for seq2seq models

Each serializer extends `BaseSerializer` and implements:
- `make_items(doc)`: Converts a canonical document into serialized items
- `serialize_train(doc)`: Creates training data with labels
- `serialize_inference(doc)`: Creates inference data without labels

### Datasets

The framework includes loaders for various document datasets:

- **Charity reports**: Annual charity report documents
- **FCC invoices**: Federal Communications Commission invoices
- **NDAs**: Non-disclosure agreements
- **Ad buy forms**: Advertisement purchase forms
- **Multi-documents**: Multi-page document collections
- **Registration forms**: Various registration form documents
- **Resource contracts**: Resource extraction contracts
- **SEC S1 filings**: Securities and Exchange Commission S1 registration statements

Each dataset loader extends `BaseDatasetLoader` and provides:
- Document loading from raw data
- OCR processing
- Annotation alignment
- Label extraction

### Preprocessing Pipeline

The preprocessing modules provide utilities for:

- **OCR processing**: Token extraction, bounding box normalization, page size extraction
- **Alignment**: Mapping annotations to OCR tokens using spatial overlap
- **Label processing**: BIO label encoding, label filtering, frequency analysis
- **Quality checks**: Data validation and quality metrics
- **Schema definitions**: Canonical document representation

## Running Experiments

### Prerequisites

Install dependencies using the project's lock file:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt  # if available
```

### Basic Usage

1. **Prepare datasets**: Ensure processed datasets are in `data/processed/<dataset_name>/<strategy>/all.jsonl`

2. **Configure experiment**: Use command-line arguments or modify `src/training/experiment_config.py`:
   - `--datasets`: Which datasets to evaluate
   - `--strategies`: Which serialization strategies to test
   - `--models`: Which models to use
   - Training hyperparameters

3. **Run experiments**:

```bash
python src/training/execute_experiment_new.py --datasets <dataset_names> --strategies <strategy_names> --models <model_names>
```

### Command Line Arguments

The experiment runner supports command-line arguments:

```bash
python src/training/execute_experiment_new.py --datasets <dataset_names> --strategies <strategy_names> --models <model_names> [additional options]
```

Common options:
- `--datasets`: Comma-separated list of dataset names (e.g., "charity_reports,fcc_invoices")
- `--strategies`: Comma-separated list of strategy names (e.g., "column_aware,plain_text")
- `--models`: Comma-separated list of model names (e.g., "bert-mlsm,deberta_v3_small")
- `--max-length`: Maximum sequence length (default: 512)
- `--train-batch-size`: Training batch size (default: 32)
- `--eval-batch-size`: Evaluation batch size (default: 8)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--num-train-epochs`: Number of training epochs (default: 20)
- `--loss-function`: Loss function type (e.g., "focal")
- `--focal-gamma`: Focal loss gamma parameter
- `--focal-alpha`: Focal loss alpha parameter
- `--mixed-precision`: Precision mode (auto, fp16, bf16, fp32)
- `--gradient-checkpointing`: Enable gradient checkpointing
- `--grad-accum`: Gradient accumulation steps

### Batch Evaluation

Use the provided `run_eval.sh` script to run experiments on multiple datasets:

```bash
bash run_eval.sh
```

The script contains pre-configured commands for all datasets with multiple models.

### Expected Output

The experiment runner generates:

- **Results CSVs** in `results_<dataset_name>/` (per dataset):
  - `strategy_split_summary.csv`: Summary of strategy and split information
  - `<dataset>_cv_doc_split_assignments.csv`: Cross-validation split assignments
  - `chunk_summary.csv`: Summary of chunked documents
  - `dataset_read_summary.csv`: Dataset loading statistics
  - `label_map.csv`: Label mapping information

- **MLflow tracking**:
  - All parameters, metrics, and artifacts logged to MLflow
  - Access via MLflow UI: `mlflow ui`

## Data Format

### Input Data

Processed datasets should be in JSONL format with the following schema:

```json
{
  "id": "document_id",
  "tokens": ["token1", "token2", ...],
  "labels": ["B-ENTITY", "I-ENTITY", "O", ...],
  "bboxes": [[x0, y0, x1, y1], ...],
  "pages": [0, 0, 1, ...],
  "metadata": {
    "doc_key": "original_document_id",
    "dataset": "dataset_name"
  }
}
```

### Serialization Output

Each serializer produces records with:

```python
{
  "id": str,
  "dataset": str,
  "serializer": str,
  "tokens": list[str],
  "labels": list[str],  # training only
  "source_token_indices": list[int|None],
  "loss_mask": list[bool],
  "layout_roles": list[str|None],
  "pages": list[int|None],
  "bboxes": list[list[int]|None],
  "normalized_bboxes": list[list[int]|None],
  "metadata": dict
}
```

## Evaluation Metrics

The framework computes comprehensive metrics:

- **Accuracy**: Overall token-level accuracy
- **Macro F1**: F1 score averaged across all labels
- **Weighted F1**: F1 score weighted by label frequency
- **Non-O Micro F1**: F1 score for entity labels only (excluding "O")
- **Per-label metrics**: Precision, recall, F1 for each label

## Configuration Options

### Training Configuration

Key training parameters in `src/training/experiment_config.py` or via command-line:

- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `per_device_train_batch_size`: Batch size per device
- `max_length`: Maximum sequence length
- `word_window_size`: Window size for chunking documents
- `mixed_precision`: Precision mode ("auto", "fp16", "bf16", "fp32")
- `early_stopping_patience`: Patience for early stopping

### Quiet Mode

The framework supports quiet training mode:

```python
QUIET_TRAINING = True  # Suppresses most output
SUPPRESS_MODEL_LOAD_WARNINGS = True
TRAINER_LOGGING_STRATEGY = "no"
```

Metrics are still written to CSV and MLflow even in quiet mode.

## Extending the Framework

### Adding a New Serialization Strategy

1. Create a new file in `src/serialization/`
2. Extend `BaseSerializer` and implement `make_items()`
3. Add to `TOKEN_CLASSIFICATION_SERIALIZERS` in `src/serialization/__init__.py`

Example:

```python
from .base import BaseSerializer, SerializedItem

class MySerializer(BaseSerializer):
    name = "my_strategy"
    
    def make_items(self, doc):
        items = []
        # Your serialization logic here
        return items
```

### Adding a New Dataset

1. Create a new loader in `src/datasets/`
2. Extend `BaseDatasetLoader`
3. Add import to `src/datasets/__init__.py`

### Adding a New Model

Add to `model_registry` in `src/training/experiment_config.py` or via command-line:

```python
model_registry = [
    ModelSpec(
        name="my-model",
        model_name_or_path="huggingface/model-name",
        tokenizer_name_or_path="huggingface/tokenizer-name",
    ),
]
```

## MLflow Integration

The framework uses MLflow for experiment tracking:

- **Experiment name**: `serialization_strategies_<timestamp>`
- **Tracking URI**: SQLite database at `mlflow.db`
- **Artifact root**: `mlartifacts/`

Run the MLflow UI to explore results:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

## Citation

If you use this framework in your research, please cite appropriately.

## License

Please refer to the project license file for usage terms.

## Contact

For questions or issues, please open an issue on the project repository.
