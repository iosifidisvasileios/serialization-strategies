from __future__ import annotations

from .base import BaseSerializer, SerializedItem
from ..preprocessing.ocr_utils import token_row_col_buckets
from ..preprocessing.schema import CanonicalDocument, OCRToken


class RowColBucketSerializer(BaseSerializer):
    """Prefix each OCR token with coarse row/column bucket tokens.

    Example:
        [PAGE_0] [R_12] [C_08] Invoice [R_13] [C_08] Date ...

    Only the real OCR tokens receive labels/loss. Bucket tokens are ignored.
    """

    name = "rowcol_bucket"

    def __init__(
        self,
        n_buckets: int = 100,
        include_page_markers: bool = True,
        include_row: bool = True,
        include_col: bool = True,
        page_token_template: str = "[PAGE_{page}]",
        row_token_template: str = "[R_{row}]",
        col_token_template: str = "[C_{col}]",
        unknown_row_token: str = "[R_UNK]",
        unknown_col_token: str = "[C_UNK]",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_buckets = n_buckets
        self.include_page_markers = include_page_markers
        self.include_row = include_row
        self.include_col = include_col
        self.page_token_template = page_token_template
        self.row_token_template = row_token_template
        self.col_token_template = col_token_template
        self.unknown_row_token = unknown_row_token
        self.unknown_col_token = unknown_col_token

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

            row, col = safe_row_col(token, n_buckets=self.n_buckets)

            if self.include_row:
                text = self.unknown_row_token if row is None else self.row_token_template.format(row=row)
                items.append(
                    self.layout_item(
                        text,
                        role="row_bucket",
                        page=token.page,
                        row=row,
                    )
                )

            if self.include_col:
                text = self.unknown_col_token if col is None else self.col_token_template.format(col=col)
                items.append(
                    self.layout_item(
                        text,
                        role="col_bucket",
                        page=token.page,
                        col=col,
                    )
                )

            items.append(
                self.token_item(
                    i,
                    token.text,
                    page=token.page,
                    row=row,
                    col=col,
                )
            )

        return items


def safe_row_col(token: OCRToken, n_buckets: int = 100) -> tuple[int | None, int | None]:
    try:
        return token_row_col_buckets(token, n_buckets=n_buckets)
    except Exception:
        return None, None
