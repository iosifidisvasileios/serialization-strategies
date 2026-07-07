from __future__ import annotations

from .base import BaseSerializer, SerializedItem
from ..preprocessing.schema import CanonicalDocument, OCRToken


class LMDXCoordSuffixSerializer(BaseSerializer):
    """Append compact coordinate suffix tokens after OCR tokens.

    Inspired by text-only layout grounding approaches where quantized location
    markers are represented as text. The default emits one ignored coordinate
    token after each real OCR token:

        token [L_page_row_col]

    The real OCR token receives the BIO label. The coordinate suffix is a
    synthetic layout token with label=-100 and loss_mask=False.
    """

    name = "lmdx_coord_suffix"

    def __init__(
        self,
        n_buckets: int = 100,
        coord_mode: str = "center",
        include_page_markers: bool = True,
        include_coord_suffix: bool = True,
        page_token_template: str = "[PAGE_{page}]",
        center_token_template: str = "[L_{page}_{row}_{col}]",
        bbox_token_template: str = "[L_{page}_{x0}_{y0}_{x1}_{y1}]",
        unknown_coord_token: str = "[L_UNK]",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if coord_mode not in {"center", "bbox"}:
            raise ValueError("coord_mode must be 'center' or 'bbox'.")
        self.n_buckets = n_buckets
        self.coord_mode = coord_mode
        self.include_page_markers = include_page_markers
        self.include_coord_suffix = include_coord_suffix
        self.page_token_template = page_token_template
        self.center_token_template = center_token_template
        self.bbox_token_template = bbox_token_template
        self.unknown_coord_token = unknown_coord_token

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        items: list[SerializedItem] = []
        previous_page = None

        for i, token in enumerate(doc.tokens):
            if self.include_page_markers and token.page != previous_page:
                items.append(
                    self.layout_item(
                        self.page_token_template.format(page=token.page),
                        role="page",
                        page=token.page,
                    )
                )
                previous_page = token.page

            features = lmdx_coord_features(token, n_buckets=self.n_buckets)

            items.append(
                self.token_item(
                    i,
                    token.text,
                    page=token.page,
                    **features,
                )
            )

            if self.include_coord_suffix:
                coord_text = self._format_coord_token(token.page, features)
                items.append(
                    self.layout_item(
                        coord_text,
                        role="coord_suffix",
                        page=token.page,
                        coord_mode=self.coord_mode,
                        **features,
                    )
                )

        return items

    def _format_coord_token(self, page: int, features: dict[str, int | None]) -> str:
        if any(value is None for value in features.values()):
            return self.unknown_coord_token

        if self.coord_mode == "center":
            return self.center_token_template.format(
                page=page,
                row=features["row"],
                col=features["col"],
            )

        return self.bbox_token_template.format(
            page=page,
            x0=features["x0"],
            y0=features["y0"],
            x1=features["x1"],
            y1=features["y1"],
        )


def lmdx_coord_features(token: OCRToken, n_buckets: int = 100) -> dict[str, int | None]:
    try:
        x0, y0, x1, y1 = token.normalized_bbox(scale=1000)
    except Exception:
        return {
            "x0": None,
            "y0": None,
            "x1": None,
            "y1": None,
            "row": None,
            "col": None,
        }

    def bucket(value: int) -> int:
        return max(0, min(n_buckets - 1, int(value * n_buckets / 1000)))

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    return {
        "x0": bucket(x0),
        "y0": bucket(y0),
        "x1": bucket(x1),
        "y1": bucket(y1),
        "row": bucket(cy),
        "col": bucket(cx),
    }
