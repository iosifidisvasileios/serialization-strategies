from __future__ import annotations

from .base import BaseSerializer, SerializedItem
from .bbox_token import discretized_bbox_features
from ..preprocessing.schema import CanonicalDocument, OCRToken


class CompactBBoxTokenSerializer(BaseSerializer):
    """Represent each OCR token's full bbox as one compact layout token.

    Compared with BBoxTokenSerializer, this emits one bbox token per OCR token
    instead of separate page/x0/y0/x1/y1 tokens.

    Example:
        [B_0_12_08_18_10] token

    If these bbox strings are later added to the tokenizer vocabulary as
    special tokens, this becomes a low-overhead coordinate-token strategy.
    """

    name = "compact_bbox_token"

    def __init__(
        self,
        n_buckets: int = 100,
        position: str = "prefix",
        bbox_token_template: str = "[B_{page}_{x0}_{y0}_{x1}_{y1}]",
        unknown_bbox_token: str = "[B_UNK]",
        include_width_height_attrs: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if position not in {"prefix", "suffix"}:
            raise ValueError("position must be 'prefix' or 'suffix'.")
        self.n_buckets = n_buckets
        self.position = position
        self.bbox_token_template = bbox_token_template
        self.unknown_bbox_token = unknown_bbox_token
        self.include_width_height_attrs = include_width_height_attrs

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        items: list[SerializedItem] = []

        for i, token in enumerate(doc.tokens):
            features = discretized_bbox_features(token, n_buckets=self.n_buckets)
            bbox_item = self._bbox_item(token, features)
            real_item = self.token_item(
                i,
                token.text,
                page=token.page,
                **features,
            )

            if self.position == "prefix":
                items.append(bbox_item)
                items.append(real_item)
            else:
                items.append(real_item)
                items.append(bbox_item)

        return items

    def _bbox_item(
        self,
        token: OCRToken,
        features: dict[str, int | None],
    ) -> SerializedItem:
        text = format_compact_bbox_token(
            page=token.page,
            features=features,
            template=self.bbox_token_template,
            unknown_token=self.unknown_bbox_token,
        )

        attrs = {
            "page": token.page,
            "x0": features.get("x0"),
            "y0": features.get("y0"),
            "x1": features.get("x1"),
            "y1": features.get("y1"),
        }
        if self.include_width_height_attrs:
            attrs["width"] = features.get("width")
            attrs["height"] = features.get("height")

        return self.layout_item(
            text,
            role="compact_bbox",
            **attrs,
        )


def format_compact_bbox_token(
    page: int,
    features: dict[str, int | None],
    template: str = "[B_{page}_{x0}_{y0}_{x1}_{y1}]",
    unknown_token: str = "[B_UNK]",
) -> str:
    required = [features.get("x0"), features.get("y0"), features.get("x1"), features.get("y1")]
    if any(value is None for value in required):
        return unknown_token

    return template.format(
        page=page,
        x0=features["x0"],
        y0=features["y0"],
        x1=features["x1"],
        y1=features["y1"],
        width=features.get("width"),
        height=features.get("height"),
    )
