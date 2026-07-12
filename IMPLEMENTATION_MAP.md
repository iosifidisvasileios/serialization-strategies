# Implementation Map

This map describes what is present in the repository as of the reviewed `main`
checkout. “Implemented” means concrete code exists; it does not imply that the
component has been exercised here. Runtime verification was not possible because
no usable Python executable is installed in the review environment.

## End-to-end flow

1. Dataset-specific loaders convert source rows and OCR payloads into a shared
   `CanonicalDocument` model.
2. Preprocessing utilities parse OCR, normalize labels and values, align
   annotations to tokens, create BIO labels, and audit data quality.
3. Serializers convert canonical documents into token-classification records or
   T5-style text-to-JSON records.
4. `DataPipeline` discovers pre-generated JSONL records, builds label maps and
   document-level CV assignments, creates windows, tokenizes them, and aligns
   word labels with tokenizer subwords.
5. `ExperimentRunner` trains each dataset/strategy/model/fold combination,
   evaluates it, logs nested MLflow runs, and writes CSV/JSON artifacts.

Important boundary: the training runner consumes already serialized files under
`data/processed/<dataset>/<strategy>/all.jsonl`. It does not invoke the dataset
loaders or serializers itself. Dataset preparation is performed separately by
the notebooks.

## Serialization strategies

Registry: `src/serialization/__init__.py`

| Strategy | Task | Implemented behavior | Status |
|---|---|---|---|
| `plain_text` | Token classification | Emits canonical OCR tokens without added layout tokens. | Implemented and registered |
| `page_aware` | Token classification | Inserts `[PAGE_n]` whenever the page changes. | Implemented and registered |
| `block_aware` | Token classification | Adds page and block markers; uses token block IDs or infers them from character-span overlap with OCR blocks. | Implemented and registered |
| `line_aware` | Token classification | Groups tokens into visual lines, reorders within that layout, and emits page/line markers. | Implemented and registered |
| `rowcol_bucket` | Token classification | Prefixes every OCR token with quantized row/column tokens and optional page markers. | Implemented and registered |
| `bbox_token` | Token classification | Prefixes each OCR token with separate quantized page/x0/y0/x1/y1 tokens; optional width/height tokens. | Implemented and registered, but omitted from `run_eval.sh` |
| `column_aware` | Token classification | Detects up to four visual columns from horizontal whitespace and emits deterministic column reading order. | Implemented and registered |
| `xycut_aware` | Token classification | Recursively partitions pages along whitespace gaps, emits region markers, and orders leaf regions. | Implemented and registered |
| `lmdx_coord_suffix` | Token classification | Appends one center or full-bbox coordinate token after each OCR token, with page markers. | Implemented and registered |
| `compact_bbox_token` | Token classification | Emits one compact full-bbox token per OCR token, as a prefix or suffix. | Implemented and registered |
| `t5_json` | Seq2seq extraction | Builds prompted document text and normalized JSON targets in field-dictionary or occurrence-grouped line-item form. | Implemented and registered, but not integrated into the token-classification runner |

All token-classification serializers share `BaseSerializer`, which produces a
consistent record containing tokens, labels, source-token indices, loss masks,
layout roles, pages, raw/normalized boxes, and metadata. Synthetic layout tokens
are excluded from loss with label `-100`. Helpers also map serialized predictions
back to original OCR-token positions.

## Dataset ingestion

Base implementation: `src/datasets/base_loader.py`

| Loader | Dataset key | Specialization |
|---|---|---|
| `CharityReportLoader` | `charity_reports` | Name-only subclass of the generic loader |
| `FCCInvoiceLoader` | `fcc_invoices` | Name-only subclass of the generic loader |
| `NDALoader` | `ndas` | Name-only subclass of the generic loader |
| `ResourceContractLoader` | `resource_contracts` | Name-only subclass of the generic loader |
| `SECS1Loader` | `sec_s1` | Name-only subclass of the generic loader |

The generic loader implements flexible column aliases, inline/file OCR loading,
OCR token and block parsing, annotation parsing, optional BIO alignment, page-size
extraction, metadata preservation, strict/non-strict handling, and optional raw
OCR retention.

The README and `run_eval.sh` additionally name `multi_docs`,
`registration_form`, and `ad_buy_form`, but there are no dedicated loader classes
for them in `src/datasets`. They may only be prepared by the notebooks/generic
data transformations.

## Preprocessing

| Area | File | Implemented capabilities |
|---|---|---|
| Canonical schema | `src/preprocessing/schema.py` | Dataclasses for OCR tokens/blocks, annotations, labeled spans, alignment issues/results, and documents; serialization helpers and document updates. |
| OCR utilities | `src/preprocessing/ocr_utils.py` | JSON/gzip input, OCR-unit traversal, bbox conversion/normalization, intersection/IoU/union, token/block/page extraction, reading/layout sorting, coordinate bucketing, and visual-line grouping. |
| Annotation alignment | `src/preprocessing/alignment.py` | Character-overlap matching, diagnostics, BIO assignment, BIO-to-span conversion, field dictionaries, and geometry-based line-item grouping helpers. |
| Label utilities | `src/preprocessing/label_utils.py` | Label aliases and normalization, annotation parsing, frequency analysis/filtering, BIO vocabulary/maps, and summaries. |
| Value normalization | `src/preprocessing/value_normalization.py` | Generic whitespace cleanup plus label-sensitive normalization for dates, money/numbers, percentages, identifiers, emails, and phone-like values. |
| Quality checks | `src/preprocessing/quality.py` | Annotation alignment checks, document reports, corpus audits, filtering annotations for training, and gold-span records. |

## Training and evaluation

Entry point: `src/training/execute_experiment_new.py`

| Component | File | Implemented capabilities |
|---|---|---|
| Configuration | `experiment_config.py` | Dataclass configuration, model presets/custom model specs, CLI parsing, JSON inputs, scalar parsing, and generic `--set` overrides. |
| Data pipeline | `data_pipeline.py` | Dataset/strategy discovery; JSONL validation; BIO label maps; document strata; stratified CV with non-stratified fallbacks; train/validation split; word windows; tokenizer/subword label alignment; tokenize-once reuse across folds; Hugging Face Dataset construction. |
| Training engine | `training_engine.py` | Hugging Face token-classification model creation; pretrained or config-only initialization; cross-entropy or focal loss; mixed precision; gradient checkpointing; in-memory best-model restoration; early stopping; GPU memory metrics; quiet mode and cleanup. |
| Evaluation | `training_engine.py` | Accuracy, macro/weighted F1, non-`O` micro F1, per-label precision/recall/F1/support, fold results, and mean/std CV summaries. |
| Tracking | `training_engine.py` | Nested MLflow runs at experiment/dataset/strategy/model/fold levels; parameter, metric, configuration, split-plan, report, and summary artifacts. |
| Batch commands | `run_eval.sh` | Commands for six dataset keys, nine strategies, five model presets, and both cross-entropy and focal-loss runs. |

The default model preset registry contains BERT/DistilBERT/MiniLM/ModernBERT
variants; exact Hugging Face identifiers are defined in
`src/training/experiment_config.py`.

## Dataset preparation notebooks

Three notebooks are committed:

- `dataset_serialization_VRDU.ipynb`
- `dataset_serialization_RealKIE.ipynb`
- `dataset_serialization_multitype.ipynb`

They are the repository's preparation layer for producing the processed JSONL
layout expected by `DataPipeline`. Raw datasets and generated `data/processed`
files are not committed.

## Review findings and implementation gaps

1. **Package imports are likely broken.** `src/training/__init__.py` imports
   `data_pipeline`, `experiment_config`, and `training_engine` as top-level
   modules instead of using relative imports. Therefore `import src` is expected
   to fail in a normal package context.
2. **No automated tests are committed.** There is no test suite for serializers,
   alignment, splitting, tokenization, metrics, or CLI behavior.
3. **Runtime validation is absent.** Static source inspection succeeded, but this
   review environment has no usable Python executable, so imports, compilation,
   CLI help, and dry runs could not be executed.
4. **Serialization is offline, not end-to-end.** The experiment runner discovers
   serialized JSONL but does not construct it from raw documents or serializer
   classes. This makes notebook output a required external step.
5. **Seq2seq stops at serialization.** `t5_json` produces training/inference
   records, but there is no seq2seq training or evaluation engine.
6. **README overstates dedicated dataset support.** Several datasets mentioned
   in documentation and shell commands have no corresponding loader class.
7. **`run_eval.sh` is a command list rather than a defensive shell runner.** It
   has no shebang, strict error handling, argument forwarding, resumability, or
   orchestration logic, and repeats commands manually.
8. **Environment portability is narrow.** `pyproject.toml` requires Python 3.12,
   pins Linux-only CUDA 13.0 PyTorch wheels, and does not define an installable
   console-script entry point.

## Practical completeness summary

- **Strongest/most complete:** canonical data model, OCR/label preprocessing,
  token-classification serializer family, CV data pipeline, and experiment
  tracking/reporting.
- **Present but not connected end-to-end:** dataset loaders and serializers to
  training; T5 JSON serialization to seq2seq training.
- **Missing for production confidence:** tests, verified packaging/imports,
  reproducible raw-to-processed CLI pipeline, and portable environment setup.
