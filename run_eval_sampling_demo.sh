#!/usr/bin/env bash

ALL_MODELS="bert-mlsm,bert_small,distilbert_base,minilm_l12_h384,modernbert_base"
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
  --sampling 40
  --n-folds 2
  --epochs 3
)

for dataset in "${DATASETS[@]}"; do
  models="$ALL_MODELS"
  if [[ "$dataset" == "charity_reports" ]]; then
    models="bert_small"
  fi
  for strategy in "${STRATEGIES[@]}"; do
    python src/training/execute_experiment_new.py \
      --models "$models" --datasets "$dataset" --strategies "$strategy" \
      "${COMMON_ARGS[@]}"
  done
done
