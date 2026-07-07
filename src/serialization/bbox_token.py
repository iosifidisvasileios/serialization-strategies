from __future__ import annotations

from .base import BaseSerializer, SerializedItem
from ..preprocessing.schema import CanonicalDocument, OCRToken


class BBoxTokenSerializer(BaseSerializer):
    """Prefix each OCR token with discretized bbox/page tokens.

    Example:
        [P_0] [X0_12] [Y0_08] [X1_18] [Y1_10] token

    This is the most layout-explicit non-LayoutLM strategy. It increases
    sequence length substantially, so it should be compared with length limits.
    """

    name = "bbox_token"

    def __init__(
        self,
        n_buckets: int = 100,
        include_page: bool = True,
        include_x0: bool = True,
        include_y0: bool = True,
        include_x1: bool = True,
        include_y1: bool = True,
        include_width: bool = False,
        include_height: bool = False,
        token_templates: dict[str, str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_buckets = n_buckets
        self.include_page = include_page
        self.include_x0 = include_x0
        self.include_y0 = include_y0
        self.include_x1 = include_x1
        self.include_y1 = include_y1
        self.include_width = include_width
        self.include_height = include_height
        self.token_templates = token_templates or {
            "page": "[P_{page}]",
            "x0": "[X0_{value}]",
            "y0": "[Y0_{value}]",
            "x1": "[X1_{value}]",
            "y1": "[Y1_{value}]",
            "width": "[W_{value}]",
            "height": "[H_{value}]",
            "unk": "[{name}_UNK]",
        }

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        items: list[SerializedItem] = []

        for i, token in enumerate(doc.tokens):
            features = discretized_bbox_features(token, n_buckets=self.n_buckets)

            if self.include_page:
                items.append(
                    self.layout_item(
                        self.token_templates["page"].format(page=token.page),
                        role="page_feature",
                        page=token.page,
                    )
                )

            for name, enabled in [
                ("x0", self.include_x0),
                ("y0", self.include_y0),
                ("x1", self.include_x1),
                ("y1", self.include_y1),
                ("width", self.include_width),
                ("height", self.include_height),
            ]:
                if not enabled:
                    continue

                value = features.get(name)
                if value is None:
                    text = self.token_templates["unk"].format(name=name.upper())
                else:
                    text = self.token_templates[name].format(value=value)

                items.append(
                    self.layout_item(
                        text,
                        role=f"bbox_{name}",
                        page=token.page,
                        **{name: value},
                    )
                )

            items.append(
                self.token_item(
                    i,
                    token.text,
                    page=token.page,
                    **features,
                )
            )

        return items


def discretized_bbox_features(
    token: OCRToken,
    n_buckets: int = 100,
) -> dict[str, int | None]:
    try:
        x0, y0, x1, y1 = token.normalized_bbox(scale=1000)
    except Exception:
        return {
            "x0": None,
            "y0": None,
            "x1": None,
            "y1": None,
            "width": None,
            "height": None,
        }

    def bucket(v: int) -> int:
        return max(0, min(n_buckets - 1, int(v * n_buckets / 1000)))

    return {
        "x0": bucket(x0),
        "y0": bucket(y0),
        "x1": bucket(x1),
        "y1": bucket(y1),
        "width": bucket(max(0, x1 - x0)),
        "height": bucket(max(0, y1 - y0)),
    }
