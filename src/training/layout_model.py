from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import DataCollatorForTokenClassification


LAYOUT_ROLE_TOKENS: dict[str, str] = {
    "page": "[LAYOUT_PAGE]",
    "page_feature": "[LAYOUT_PAGE]",
    "block": "[LAYOUT_BLOCK]",
    "line": "[LAYOUT_LINE]",
    "row_bucket": "[LAYOUT_ROW]",
    "col_bucket": "[LAYOUT_COL]",
    "bbox_x0": "[LAYOUT_X0]",
    "bbox_y0": "[LAYOUT_Y0]",
    "bbox_x1": "[LAYOUT_X1]",
    "bbox_y1": "[LAYOUT_Y1]",
    "bbox_width": "[LAYOUT_WIDTH]",
    "bbox_height": "[LAYOUT_HEIGHT]",
    "column": "[LAYOUT_COLUMN]",
    "xycut_region": "[LAYOUT_REGION]",
    "coord_suffix": "[LAYOUT_COORD]",
    "compact_bbox": "[LAYOUT_BBOX]",
}
UNKNOWN_LAYOUT_TOKEN = "[LAYOUT_UNKNOWN]"
ALL_LAYOUT_TOKENS = tuple(sorted({*LAYOUT_ROLE_TOKENS.values(), UNKNOWN_LAYOUT_TOKEN}))


def canonical_layout_token(role: Any) -> str:
    """Return a bounded, strategy-independent token for a layout item."""
    return LAYOUT_ROLE_TOKENS.get(str(role), UNKNOWN_LAYOUT_TOKEN)


@dataclass
class DataCollatorForTokenClassificationWithLayout:
    """Pad numeric word layout alongside the normal token-classification batch."""

    tokenizer: Any
    pad_to_multiple_of: int | None = None
    _base: DataCollatorForTokenClassification = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._base = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    @staticmethod
    def _pad_sequence(values, target_length: int, pad_value, padding_side: str):
        values = list(values)
        if len(values) > target_length:
            raise ValueError(
                f"Layout feature length {len(values)} exceeds padded input length {target_length}."
            )
        padding = [pad_value for _ in range(target_length - len(values))]
        return padding + values if padding_side == "left" else values + padding

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        clean_features: list[dict[str, Any]] = []
        bboxes = []
        bbox_masks = []
        page_ids = []
        for feature in features:
            item = dict(feature)
            input_length = len(item.get("input_ids", []))
            layout_values = {
                "bbox": item.pop("bbox", None),
                "bbox_mask": item.pop("bbox_mask", None),
                "page_ids": item.pop("page_ids", None),
            }
            for name, values in layout_values.items():
                if values is None or len(values) != input_length:
                    actual = None if values is None else len(values)
                    raise ValueError(
                        f"{name} must align one-to-one with input_ids; "
                        f"got {actual} values for {input_length} input ids."
                    )
            bboxes.append(layout_values["bbox"])
            bbox_masks.append(layout_values["bbox_mask"])
            page_ids.append(layout_values["page_ids"])
            # Metric-only columns normally disappear through Trainer's
            # remove_unused_columns path.  Pop them defensively so the collator
            # remains safe in direct/unit-test use as well.
            for key in list(item):
                if key.startswith("metric_") or key in {
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
                }:
                    item.pop(key, None)
            clean_features.append(item)

        batch = self._base(clean_features)
        target_length = int(batch["input_ids"].shape[1])
        side = getattr(self.tokenizer, "padding_side", "right")
        batch["bbox"] = torch.tensor(
            [
                self._pad_sequence(values, target_length, [0, 0, 0, 0], side)
                for values in bboxes
            ],
            dtype=torch.long,
        )
        batch["bbox_mask"] = torch.tensor(
            [self._pad_sequence(values, target_length, 0, side) for values in bbox_masks],
            dtype=torch.bool,
        )
        batch["page_ids"] = torch.tensor(
            [self._pad_sequence(values, target_length, -1, side) for values in page_ids],
            dtype=torch.long,
        )
        return batch


class NumericLayoutTokenClassifier(torch.nn.Module):
    """Add normalized OCR geometry to any token classifier via ``inputs_embeds``.

    The projection is initialized to zero.  A wrapped pretrained model therefore
    starts with exactly its original text behavior while the projection can learn
    numeric layout during fine-tuning.  No model-specific CLI option is needed.
    """

    main_input_name = "input_ids"

    def __init__(self, wrapped_model: torch.nn.Module):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.config = wrapped_model.config
        embedding = wrapped_model.get_input_embeddings()
        hidden_size = int(getattr(embedding, "embedding_dim", embedding.weight.shape[1]))
        # x0, y0, x1, y1 and page number; bias-free keeps missing layout at zero.
        self.layout_projection = torch.nn.Linear(5, hidden_size, bias=False)
        torch.nn.init.zeros_(self.layout_projection.weight)
        self._wrapped_forward_signature = inspect.signature(wrapped_model.forward)
        # Transformers' Trainer consults this flag before passing loss-scaling
        # kwargs such as num_items_in_batch.  A generic **kwargs parameter does
        # not imply that a token classifier actually consumes those values.
        self.accepts_loss_kwargs = bool(
            getattr(wrapped_model, "accepts_loss_kwargs", False)
        )

    def get_input_embeddings(self):
        return self.wrapped_model.get_input_embeddings()

    def get_output_embeddings(self):
        getter = getattr(self.wrapped_model, "get_output_embeddings", None)
        return getter() if getter is not None else None

    def resize_token_embeddings(self, *args, **kwargs):
        return self.wrapped_model.resize_token_embeddings(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        method = getattr(self.wrapped_model, "gradient_checkpointing_enable", None)
        if method is None:
            raise TypeError(
                f"{type(self.wrapped_model).__name__} does not support gradient checkpointing."
            )
        return method(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        method = getattr(self.wrapped_model, "gradient_checkpointing_disable", None)
        if method is not None:
            return method(*args, **kwargs)
        return None

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.wrapped_model, "is_gradient_checkpointing", False))

    def _supports(self, name: str) -> bool:
        return name in self._wrapped_forward_signature.parameters

    def _accepts_arbitrary_kwargs(self) -> bool:
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in self._wrapped_forward_signature.parameters.values()
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        bbox=None,
        bbox_mask=None,
        page_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required.")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if bbox is not None:
            if not self._supports("inputs_embeds"):
                raise TypeError(
                    f"{type(self.wrapped_model).__name__} does not accept inputs_embeds; "
                    "numeric OCR layout cannot be injected safely."
                )
            bbox_values = bbox.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            bbox_values = bbox_values.clamp(0, 1000) / 1000.0
            if bbox_mask is not None:
                bbox_values = bbox_values * bbox_mask.to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                ).unsqueeze(-1)
            if page_ids is None:
                page_value = torch.zeros_like(bbox_values[..., :1])
            else:
                page_value = page_ids.to(device=inputs_embeds.device)
                page_valid = page_value >= 0
                page_value = ((page_value.clamp(0, 255) + 1).to(inputs_embeds.dtype) / 256.0)
                page_value = page_value.unsqueeze(-1) * page_valid.to(
                    inputs_embeds.dtype
                ).unsqueeze(-1)
            layout_values = torch.cat([bbox_values, page_value], dim=-1)
            inputs_embeds = inputs_embeds + self.layout_projection(layout_values)

        candidates = {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        call_kwargs = {
            key: value
            for key, value in candidates.items()
            if value is not None and self._supports(key)
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if self._supports(key):
                call_kwargs[key] = value
            elif (
                key == "num_items_in_batch"
                and self.accepts_loss_kwargs
                and self._accepts_arbitrary_kwargs()
            ):
                call_kwargs[key] = value
        return self.wrapped_model(**call_kwargs)
