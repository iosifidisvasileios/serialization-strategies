# OCR-Output-Only Preprocessing and Serialization Gap Analysis

Reviewed against the current `main` checkout on 2026-07-11.

## Executive conclusion

This report assumes the original page images are permanently unavailable. The
system can use only OCR output: text, offsets, pages, boxes, blocks, style, and
any confidence or hierarchy metadata already present in that output.

The repository has a useful collection of **post-OCR serialization** methods.
It parses tokens and boxes, aligns annotations, reorders tokens, and emits text
markers. It does not yet validate or repair OCR geometry, use OCR confidence,
repair OCR tokenization, reconstruct robust reading order, or infer semantic
page structure from the OCR output.

Before adding more strategies, the experiment harness needs several P0 fixes.
Today, layout marker strings are not registered as atomic tokenizer tokens,
numeric boxes stored in JSON are discarded by the text-model pipeline,
different strategies can contain different documents, and evaluation counts
subwords rather than reconstructed OCR tokens or entity spans. Those issues can
change strategy rankings independently of layout quality.

The recommended order is:

1. make comparisons paired, tokenization-aware, and OCR-token-level;
2. add a canonical OCR schema, geometry validation, optional confidence, and
   reversible text/token repair;
3. implement robust reading order, OCR-derived regions, tables, and relations;
4. add true numeric-layout, text-layout, and graph baselines;
5. add OCR-output corruption experiments and robustness slices.

## Scope under the no-image constraint

These should be evaluated separately rather than mixed under one strategy name.

| Layer | Input → output | What exists now | Main additions |
|---|---|---|---|
| OCR normalization | provider OCR → canonical tokens/geometry | Basic narrow-schema parser | provider adapters, validation, optional confidence/polygons, Unicode repair, merge/split, provenance |
| Structure reconstruction | canonical OCR → ordered lines/regions/cells/edges | Greedy lines, columns, simplified XY-cut | robust reading order, paragraphs, headers/footers, tables, hierarchy, spatial graphs |
| Model serialization | structured OCR → model input | 10 token classifiers + T5 JSON | whitespace, relative geometry, stable region markers, relations, numeric-layout models |

Image-only proposals are out of scope: orientation correction, deskewing,
dewarping, contrast enhancement, binarization, denoising, ruling-line removal,
crop-based re-recognition, visual layout detectors, pixel-input LayoutLM variants,
and OCR-free models cannot be implemented from OCR output alone. Rotation can
only be corrected if the OCR output already includes reliable orientation or
polygon information.

## P0: limitations that affect every current comparison

### 1. Stored layout is not actually passed as numeric layout

`BaseSerializer` writes `bboxes`, `normalized_bboxes`, `item_attrs`, pages, and
source indices (`src/serialization/base.py:98-147`). `DataPipeline` then retains
only `tokens` and `labels` in examples (`src/training/data_pipeline.py:442-478`).
`ExperimentRunner` builds ordinary text `AutoModelForTokenClassification`
models (`src/training/training_engine.py:516-535`).

Consequence: all current layout experiments are tests of **layout written as
text**, not tests of 2D embeddings or visual features. The rich geometry in the
record is dead data during training.

### 2. Layout markers are not atomic tokens

`build_tokenizer()` only calls `AutoTokenizer.from_pretrained()`
(`src/training/data_pipeline.py:517-527`); no serializer vocabulary is added and
the model embedding table is never resized. Strings such as `[PAGE_3]`,
`[X0_12]`, and `[B_0_12_8_18_10]` may therefore split differently for every
model. The `compact_bbox_token` docstring itself says its representation is
low-overhead only *if* those strings are later added to the vocabulary
(`src/serialization/compact_bbox_token.py:8-19`).

This makes marker fragmentation and tokenizer choice a hidden experimental
variable. Hugging Face explicitly documents that added special tokens are not
split and that the model embeddings must be resized after adding them:
[tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer).

Fix: give every serializer a finite `layout_vocabulary()` contract, register
those tokens before tokenization, resize the model, and log how every emitted
item tokenizes. Do not create one special token for every full bbox combination;
factor coordinates into bounded vocabularies such as `[X0] [B12]`.

### 3. Strategies receive unequal context and compute

Default emitted items per OCR token differ substantially:

- plain text: 1;
- row/column buckets: about 3;
- compact bbox and LMDX: about 2;
- full bbox: about 6.

Tokenizer overflow is unaware of layout groups (`src/training/data_pipeline.py:
548-627`). Verbose methods produce more chunks and less neighboring OCR context.
Page/block/line/region markers are emitted once, so an overflow chunk beginning
mid-group may lose its marker entirely. Fixed epochs and batch sizes then give
strategies with more chunks more optimizer work (`src/training/training_engine.py:
547-591`).

Fix and report both:

- **equal original-OCR context**: windows are defined in source OCR tokens and
  all required group markers are re-emitted at each chunk boundary;
- **equal compute**: equal optimizer updates or equal non-padding input tokens.

Always log real-token coverage, marker subtokens, truncation, chunks per document,
and total training tokens.

### 4. Evaluation is subword-weighted, not OCR-token/entity-level

Every subtoken inherits a label and later subtokens turn `B-` into `I-`
(`src/training/data_pipeline.py:571-591`). Metrics flatten all nonignored
subtokens (`src/training/training_engine.py:381-439`). A word split into five
pieces therefore counts five times. Overflow overlap can duplicate evaluation
positions; `seen_word_ids` is populated but never validated
(`src/training/data_pipeline.py:562-575`).

The repository already provides a source-index collapse helper
(`src/serialization/base.py:220-281`), but the training examples discard source
indices and metrics never call it.

Fix: retain `source_token_indices`, word IDs, document IDs, and overflow ownership;
choose one prediction per original OCR token; then report:

- OCR-token accuracy and macro/weighted/non-`O` F1;
- strict entity-span F1 and normalized entity-value F1;
- document-level exact match or field-level F1;
- coverage and duplicate counts.

### 5. Reordering can invalidate BIO sequences

Line, column, and XY-cut serializers both reorder tokens and add markers. Labels
are copied by source index without rebuilding BIO (`src/serialization/base.py:
111-121`). A reordered multi-line entity can begin with `I-`, be separated from
its `B-`, or be split across regions.

Fix: add separate ablations for `reorder_only`, `markers_only`, and
`reorder_plus_markers`. Recompute BIO within each emitted sequence or evaluate a
token-label task whose semantics do not depend on serialization adjacency.

### 6. Strategy cohorts are not guaranteed to match

The notebooks catch serialization errors and continue, so one strategy can omit
documents another retains (`dataset_serialization_multitype.ipynb:2760-2800`,
`dataset_serialization_VRDU.ipynb:3719-3760`, and
`dataset_serialization_RealKIE.ipynb:805-868`). The trainer constructs a union of
document keys but never asserts an identical set per strategy
(`src/training/data_pipeline.py:227-252,358-410`). Sampling is performed on each
strategy independently (`src/training/data_pipeline.py:142-149`).

Fix: generate one canonical cohort manifest first; make every strategy return a
record or an explicit failure; fail comparison runs unless document keys,
original-token counts, source maps, and original labels match exactly.

### 7. Notebook outputs do not match default trainer discovery

The trainer expects `<root>/<dataset>/<strategy>/all.jsonl`
(`src/training/data_pipeline.py:91-118`) below the default `data/processed` root.
The notebooks use incompatible layouts:

- multitype: `data/processed/multi_docs/serialized/<strategy>/all.jsonl`;
- VRDU: `data/processed/vrdu/serialized/<corpus>/<strategy>/all.jsonl`;
- RealKIE: `data/raw/processed/<dataset>/<strategy>/all.jsonl`.

Fix: replace notebook-only export with one versioned preparation CLI that writes
the trainer layout, a cohort manifest, a preprocessing manifest, errors, checksums,
and full quality summaries.

### 8. Random document CV can leak templates

The split stratum is only the dominant entity label
(`src/training/data_pipeline.py:227-325`). There is no grouping by template,
vendor, issuer, original PDF, scan source, or near-duplicate family. Absolute
coordinate methods can therefore memorize common templates and appear more
general than they are.

Fix: add grouped CV by template/vendor/source and near-duplicate clustering;
retain the existing random-document CV only as an easier in-distribution result.

## Limitations of each existing strategy

| Strategy | What it currently tests | Main limitations |
|---|---|---|
| `plain_text` | OCR text in provider/canonical order | Drops all boundaries and spacing; “reading order” primarily trusts OCR character offsets `(page,start,y,x)`, so it is OCR-provider-dependent (`ocr_utils.py:96-103,127-131`). |
| `page_aware` | Absolute page marker on transitions | Page numbers have no stable semantics across providers; overflow chunks can lose the marker; conveys no within-page structure (`page_aware.py:25-40`). |
| `block_aware` | Page/block transition strings | Missing IDs are inferred only through character-span overlap, not geometry; ties favor larger block ID; existing IDs are trusted; `block_type` is ignored; arbitrary document-local IDs are encoded as though comparable (`block_aware.py:82-119`). |
| `line_aware` | Greedy visual lines plus markers and reorder | Center-Y-only grouping assigns the first matching line; raw-pixel threshold depends on DPI/font/skew; same-Y columns can merge; sorting is LTR only; line IDs continue across pages (`ocr_utils.py:149-158`, `line_aware.py:32-76`). |
| `rowcol_bucket` | Absolute center bucket per token | Loses box size, repeats two markers per OCR token, aliases positions at fixed boundaries, and becomes `UNK` without page size (`rowcol_bucket.py:44-100`). |
| `bbox_token` | Absolute quantized page and four corners | Roughly five layout items per word; page repeats unnecessarily; severe context cost; no polygon/rotation; omitted from `run_eval.sh` (`bbox_token.py:53-132`). |
| `compact_bbox_token` | Full absolute bbox in one emitted string | The string can still become many subtokens; theoretical page × bucket⁴ combinations cannot all be special tokens; crop/rotation/page-size sensitive (`compact_bbox_token.py:8-19,94-112`). |
| `lmdx_coord_suffix` | Absolute center or bbox suffix | Doubles emitted words; center conflates differently sized boxes; bbox mode has the same high-cardinality issue; one missing geometry field makes the whole suffix unknown (`lmdx_coord_suffix.py:47-132`). |
| `column_aware` | Flat left-to-right whitespace columns | Full-width headers, footers, or spanning cells can bridge all X intervals and suppress cuts; sparse pages default to one column; fixed thresholds are crop/font/provider-sensitive; flat column order fails row-major forms, tables, nested columns, sidebars, and RTL (`column_aware.py:128-234`). |
| `xycut_aware` | Simplified recursive whitespace ordering | Spanning/noisy boxes suppress cuts; thresholds use the tight region bbox; fixed depth/token/gap heuristics are uncalibrated; depth-first order is not semantic reading order; IDs are unstable (`xycut_aware.py:130-298`). |
| `t5_json` | OCR text → normalized JSON record generation | Schema prompt is inferred from fields present in each training document, leaking target presence and mismatching inference; input can truncate while target retains unseen fields; scalar/list type changes; line items pair by occurrence index rather than geometry; no seq2seq trainer/evaluator (`t5_json.py:67-171`). |

### Coordinate-family limitations shared by four serializers

- Page dimensions are mandatory for normalization; broad exception handling
  silently turns geometry into `UNK`.
- Invalid, inverted, zero-area, or out-of-page boxes are not rejected; clamping
  can hide bad OCR geometry (`src/preprocessing/schema.py:39-50`).
- Fixed 100-bin absolute coordinates are discontinuous under small OCR jitter and
  sensitive to cropping, margins, rotation, deskewing, and aspect ratio.
- They do not directly express relative alignment, containment, adjacency, or
  key/value relations. BROS is a relevant relative-position alternative:
  [BROS paper](https://arxiv.org/abs/2108.04539).

## Limitations of the shared OCR preprocessing

### Canonical schema and ingestion

- `OCRToken` has an axis-aligned integer bbox but no first-class recognition or
  detection confidence, polygon/baseline, rotation, language/script, OCR engine
  and version, alternative hypotheses, hierarchy IDs, or transform provenance
  (`src/preprocessing/schema.py:5-20`).
- The parser assumes `pages[].size`, `tokens`, `blocks`, `doc_offset`, and
  `position` (`src/preprocessing/ocr_utils.py:69-120`). Tokens without offsets or
  boxes are silently discarded (`ocr_utils.py:87-103`).
- There is no validation for negative, inverted, zero-area, out-of-page, or
  impossible boxes; zero page dimensions can cause division by zero.
- Parsing logic is duplicated in the base loader and notebooks, so providers can
  be normalized differently.

### Text and token repair

- OCR token text is not Unicode-normalized, dehyphenated, split, merged, or
  deduplicated before labeling.
- `compact_alnum()` deletes all non-ASCII letters (`value_normalization.py:20-25`).
- Money/date label classification uses substring matching: a label ending in
  `net` may be treated as money, and one containing `date` as a date
  (`value_normalization.py:10-35`).
- Money parsing is US-centric and keeps the first number; dates only lose spaces
  rather than being parsed (`value_normalization.py:38-102`).
- Loose matching accepts either normalized value as a substring of the other,
  which can accept short false positives (`value_normalization.py:132-145`).

### Reading order and alignment

- Default order trusts OCR offsets. Geometry-only order is naïve top-to-bottom,
  left-to-right; no RTL, bidirectional, vertical, or rotated text handling exists.
- Greedy line grouping is pixel-scale-dependent and can merge columns
  (`ocr_utils.py:149-158`).
- Alignment is character-offset-only, using token coverage but no annotation
  coverage, sequence alignment, fuzzy fallback, or geometry fallback
  (`alignment.py:27-62`).
- BIO is single-label. Nested/overlapping annotations conflict, and default
  `keep_first` favors the first sorted annotation (`alignment.py:98-165`).
- Line-item grouping uses a fixed 12-pixel center-Y threshold, so it varies by DPI
  and cannot represent wrapped or empty cells (`alignment.py:296-320`).

### Quality controls

Current reports measure annotation availability and text match only
(`quality.py:75-148`). They omit OCR confidence, invalid/duplicate boxes, page
coverage, reading order, CER/WER, and structural coverage.
Documents with zero annotations receive perfect `1.0` rates
(`quality.py:125-145`), and filtering difficult unmatched annotations can make
downstream scores look cleaner (`quality.py:187-224`). There is no automated
test suite.

## What else should be implemented

### A. Fix benchmark validity first

| Addition | Design | Effort | Priority |
|---|---|---:|---:|
| Strategy contract | `layout_vocabulary()`, emitted-token audit, source-map invariant, deterministic config hash | M | P0 |
| Paired cohort validator | identical IDs, tokens, labels, source maps, folds; quarantine failures | S–M | P0 |
| Original-token evaluation | collapse overflow/subwords, span/value/document metrics | M | P0 |
| Fair budgets | equal original-token context and equal compute reports | M | P0 |
| Preparation CLI | one output layout and manifest; replace notebook-only export | M | P0 |
| Tests | golden OCR fixtures plus serializer, alignment, path, cohort, and metric invariants | M | P0 |

### B. Canonical OCR and reversible repair

Extend `OCRToken` with:

- `raw_text`, normalized `text`, and a raw↔normalized character map;
- recognition/detection confidence when supplied by the OCR provider;
- polygon, baseline, rotation, and writing direction when present;
- language/script, OCR engine/model/version, and alternate hypotheses when present;
- page/region/block/line/table/cell hierarchy;
- raw provider fields plus the provenance of every OCR-output transformation.

Implement an ordered, versioned repair pipeline:

1. Unicode NFC for model text and optional NFKC for matching-only views;
2. control/soft-hyphen/ligature cleanup without losing provenance;
3. duplicate/overlap suppression;
4. same-line fragmented-token merge;
5. fused-token split using geometry, punctuation, dictionaries, or conservative
   text patterns;
6. end-of-line dehyphenation with lexical-hyphen protection;
7. locale-aware number/date normalization;
8. full offset remapping for annotations.

Unicode supplies the normative basis for
[normalization](https://unicode.org/reports/tr15/),
[text segmentation](https://www.unicode.org/reports/tr29/), and
[bidirectional ordering](https://www.unicode.org/reports/tr9/).

Post-correction should remain optional and retain raw text: learned correction
can alter identifiers or amounts. A BART OCR post-correction study demonstrates
that transformer correction can improve noisy OCR, but its domain-specific risk
must be measured: [ACL paper](https://aclanthology.org/2021.wnut-1.31/).

### C. What cannot be recovered without images

Do not plan image enhancement, OCR reruns, crop re-recognition, visual layout
detection, or OCR-free extraction. OCR output cannot recover glyph pixels,
background noise, ruling lines, stamps, figures, handwriting missed by OCR, or
the distinction between recognition and detection failures.

The output-only pipeline should instead preserve uncertainty explicitly:

- never invent missing text or geometry silently;
- retain raw and repaired text together;
- mark missing boxes, sizes, confidence, hierarchy, and orientation;
- quarantine structurally invalid documents or route them to a fallback;
- report performance separately for complete and incomplete OCR metadata.

### D. Better post-OCR layout serializers

#### 1. `whitespace_gap`

Emit bounded tokens for page breaks, line breaks, paragraph breaks, indentation,
and coarse horizontal gaps. This is the best next low-cost baseline because it
preserves human-visible whitespace with a small vocabulary and far less sequence
inflation than full bbox strings.

Example:

```text
[PAGE] [INDENT_2] Invoice [GAP_4] 12345 [NL] [INDENT_2] Date ...
```

Prerequisite: robust line/paragraph grouping. Use normalized gaps relative to
median character height/width, not raw pixels.

#### 2. `relative_geometry`

Encode bounded relative facts rather than a unique absolute bbox string:

```text
[SAME_LINE] [DX_3] [DY_0] [W_2] token
```

Useful features include delta from predecessor, nearest left/above neighbor,
overlap ratios, alignment, containment, and normalized width/height. A stronger
variant adds relative spatial bias to attention, following the motivation of
[BROS](https://arxiv.org/abs/2108.04539), instead of consuming sequence positions.

#### 3. `hierarchical_region`

Infer and serialize page → region → block → paragraph → line → token using only
OCR text, geometry, block metadata, repetition across pages, and style when
available. Use stable markers such as `[TITLE]`, `[HEADER]`, `[PARAGRAPH]`,
`[TABLE]`, `[CAPTION]`, and `[FOOTER]`, not arbitrary region IDs. Start with
deterministic rules, then compare an OCR-feature region classifier. StructuralLM
supports the usefulness of cell/segment-level structure:
[StructuralLM](https://arxiv.org/abs/2105.11210).

#### 4. `table_cell`

Infer tables from aligned text boxes, repeated X positions, row baselines,
numeric/text column patterns, and OCR block types. Recover rows, columns, header
cells, and likely spans, then serialize stable structure:

```text
[TABLE] [ROW] [HEADER_CELL] Qty [CELL] Price [ROW] [CELL] 2 [CELL] 19.95
```

Microsoft's PubTables-1M work provides structure-recognition, functional-analysis,
and canonicalization ideas:
[PubTables-1M](https://arxiv.org/abs/2110.00061).

Without pixels or ruling lines, borderless-table and empty-cell recovery is
fundamentally ambiguous. Represent uncertainty and evaluate only against datasets
with table/cell ground truth; do not claim exact cell recovery from boxes alone.

#### 5. `directional_neighbors` / layout graph

Build word/line/block/cell nodes with typed edges: nearest left/right/above/below,
same line/block/table, containment, alignment, reading predecessor, and candidate
key→value. A simple baseline serializes a bounded number of neighbor relations;
a stronger baseline uses a graph or spatial-attention model. Spatial graph work
supports both layout grouping and key-information extraction:
[post-OCR paragraph GCN](https://arxiv.org/abs/2101.12741) and
[spatial dual-modality graph reasoning](https://arxiv.org/abs/2103.14470).

#### 6. `confidence_aware`

When the OCR output includes confidence, preserve it and compare:

- confidence bucket as a feature;
- confidence-calibrated loss weight;
- an uncertainty marker or abstention policy.

Do not simply delete low-confidence tokens: confidence correlates with script,
scan quality, document age, and rare fonts. Confidence-aware OCR error detection
is an established direction:
[ConfBERT](https://arxiv.org/abs/2409.04117).

#### 7. `style_aware`

Use the existing but ignored `style` field for bounded features such as bold,
italic, relative font-size quantile, capitalization, and alignment. Style should
be normalized within each page/document to avoid provider-specific raw values.

#### 8. `header_footer_aware`

Detect repeated top/bottom page regions across a document. Mark headers, footers,
page numbers, and continuation blocks; do not delete them by default because
some datasets label fields in those regions.

#### 9. `multi_ocr_consensus` (only if multiple OCR outputs already exist)

If two or more OCR outputs were retained upstream, normalize and spatially align
their tokens, retain alternatives/confidences, and choose a calibrated consensus
or serialize uncertainty. Do not include this strategy when only one OCR result
exists; images are unavailable, so another engine cannot be run later.

#### 10. Character/byte-level noisy-text baseline

Compare the existing wordpiece models with a character- or byte-level encoder,
using the same OCR order and structural markers. This does not correct OCR, but
it reduces dependence on a clean word vocabulary when OCR splits words or
introduces spelling noise. Report the extra sequence length and compute cost.
Relevant output-only model families include
[ByT5](https://arxiv.org/abs/2105.13626) and
[CANINE](https://research.google/pubs/canine-pre-training-an-efficient-tokenization-free-encoder-for-language-understanding/).

### E. Learned reading order and multilingual support

Replace the greedy LTR path with two baselines:

1. a deterministic directed precedence graph over regions/lines, followed by
   cycle resolution and topological ordering;
2. a learned permutation model such as
   [LayoutReader](https://arxiv.org/abs/2108.11591).

Handle spanning titles, headers/footers, captions, sidebars, tables, nested
columns, RTL/BiDi text, vertical writing, and rotated regions. Report successor
edge F1 or pairwise/Kendall order accuracy, not only downstream entity F1.

For multilingual modeling, add script/language detection and compare LayoutXLM
or LiLT-style paths rather than applying ASCII cleanup and LTR order to every
document: [LayoutXLM](https://arxiv.org/abs/2104.08836) and
[LiLT](https://arxiv.org/abs/2202.13669).

### F. True numeric-layout baselines without pixels

Add a separate training path that aligns normalized bboxes to subtokens and
passes them as model tensors. LayoutLM supports token classification with
normalized `(x0,y0,x1,y1)` boxes without requiring page images:
[official LayoutLM documentation](https://huggingface.co/docs/transformers/model_doc/layoutlm).

Recommended comparisons:

- text-only baseline;
- text plus atomic layout-marker baseline;
- BROS-style text + relative numeric geometry;
- LayoutLM/LiLT-style text + absolute numeric boxes;
- OCR-layout graph model using text, boxes, and typed edges.

This distinguishes “does numeric layout help?” from “can a plain text tokenizer
learn a coordinate string?” It stays fully compatible with the no-image
constraint. BROS specifically motivates relative 2D layout without relying on
visual features: [BROS](https://arxiv.org/abs/2108.04539).

### G. Relations and structured outputs

BIO tagging cannot represent which repeated key belongs to which value, which
cells form one line item, or which caption belongs to which table. Add:

- semantic entity recognition followed by spatially pruned relation candidates;
- explicit `key_of`, `value_of`, `same_line_item`, and `caption_of` links;
- geometry-aware line-item targets rather than occurrence-index pairing;
- constrained JSON schema, stable scalar/list types, explicit missing values,
  and a real seq2seq trainer/evaluator for `t5_json`.

Explicit geometric relation modeling is strongly motivated by GeoLayoutLM:
[GeoLayoutLM](https://arxiv.org/abs/2304.10759). StructuralLM also supports
cell-level rather than word-only structure:
[StructuralLM](https://arxiv.org/abs/2105.11210).

### H. Robustness augmentation

Create training-only OCR-output augmentations and fixed corrupted test sets:

- character substitutions from observed OCR confusion matrices;
- token deletion, duplication, merge, and split;
- punctuation/whitespace loss and hyphenation changes;
- bbox jitter, box dropout, page-size corruption, and quantization-boundary noise;
- reading-order swaps, block-ID loss, and hierarchy corruption;
- confidence perturbation when confidence exists;
- synthetic translation/scale/crop transforms applied consistently to all boxes.

Evaluate each corruption type and severity independently. Because these errors
are no longer generated by rerunning OCR on altered images, treat them as
controlled stress tests. Learn corruption distributions from paired clean/OCR
corpora when available, and do not claim that arbitrary synthetic noise exactly
represents production OCR failures.

## Evaluation matrix for new work

| Dimension | Required measures |
|---|---|
| OCR text | CER, WER, normalized value accuracy, correction precision/recall |
| Geometry | invalid/out-of-page/zero-area rate, polygon→bbox loss, bbox jitter sensitivity |
| Ordering | successor-edge F1, pairwise order accuracy/Kendall correlation |
| Alignment | annotation coverage, token coverage, conflict rate, retained positives by label |
| Extraction | original OCR-token F1, strict entity span F1, normalized field F1, document exact match |
| Structure | relation F1, line-item row F1, table structure metric such as GriTS |
| Efficiency | marker fragmentation, original-token coverage, chunks/doc, input tokens, updates, time, peak memory |
| Robustness | clean plus per-corruption severity, OCR confidence, language/script, document type, sparse/dense layout |
| Generalization | random-doc CV and grouped template/vendor/source CV |

## Concrete implementation roadmap

### Phase 0 — trustworthy benchmark

- unify preparation paths in a CLI;
- enforce paired cohorts and grouped folds;
- add serializer vocabulary/source-map contracts;
- add original-token/entity metrics and coverage checks;
- add reorder-only/marker-only ablations;
- add unit/integration/golden tests.

### Phase 1 — robust canonical OCR

- extend schema with confidence, polygon, direction, engine, language, hierarchy,
  raw text, alternates, and provenance when those fields exist upstream;
- add provider adapters and geometry validation;
- add reversible Unicode/token repair;
- add full OCR-output quality metrics and coordinate/text diagnostics.

### Phase 2 — strongest low-cost new strategies

- `whitespace_gap`;
- robust graph-based reading order;
- `relative_geometry`;
- `confidence_aware` when confidence exists;
- `hierarchical_region`.

### Phase 3 — structured layouts

- table/cell recovery;
- key/value and line-item relations;
- geometry-aware T5 targets and seq2seq training;
- header/footer and multi-page continuation handling.

### Phase 4 — stronger baselines and robustness

- numeric-layout BROS/LiLT/LayoutLM path without pixels;
- multilingual/script-aware paths;
- OCR-output augmentation and fixed robustness suites.

## Recommended immediate next three changes

1. Implement paired cohort validation plus original-OCR-token evaluation. This
   makes every existing result more trustworthy.
2. Implement the layout-token vocabulary contract and `whitespace_gap` baseline.
   This produces a clean, low-cardinality test of layout text markers.
3. Extend `OCRToken` with optional confidence/polygon/hierarchy/provenance and
   add robust line and reading-order reconstruction. This unlocks confidence,
   graph, semantic-region, table, and multilingual strategies without another
   schema migration.
