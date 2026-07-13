#!/usr/bin/env bash

MODELS="bert-mlsm,bert_small,distilbert_base,minilm_l12_h384,modernbert_base"
DATASETS=(charity_reports fcc_invoices multi_docs ndas registration_form ad_buy_form)
STRATEGIES=(
  precedence_graph_order
  key_value_row_pairs
  key_value_anchor_pairs
  block_aware
  column_aware
  compact_bbox_token
  line_aware
  lmdx_coord_suffix
  page_aware
  plain_text
  rowcol_bucket
  xycut_aware
)
COMMON_ARGS=(
  --max-length 512
  --word-window-size 0
  --tokenizer-stride 0
  --train-batch-size 32
  --eval-batch-size 8
  --grad-accum 2
  --mixed-precision auto
  --gradient-checkpointing true
  --eval-accumulation-steps 8
  --dataloader-num-workers 4
)

for dataset in "${DATASETS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    python src/training/execute_experiment_new.py \
      --models "$MODELS" --datasets "$dataset" --strategies "$strategy" \
      "${COMMON_ARGS[@]}"
    python src/training/execute_experiment_new.py \
      --models "$MODELS" --datasets "$dataset" --strategies "$strategy" \
      "${COMMON_ARGS[@]}" \
      --loss-function focal --focal-gamma 2.0 --focal-alpha 0.75
  done
done
