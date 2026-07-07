from __future__ import annotations

from .base import BaseSerializer, SerializedItem
from ..preprocessing.ocr_utils import line_group_tokens
from ..preprocessing.schema import CanonicalDocument, OCRToken


class LineAwareSerializer(BaseSerializer):
    """Group tokens by visual line and insert line markers.

    This changes token order from character/reading order to visual layout order.
    source_token_indices preserve the inverse mapping for evaluation.
    """

    name = "line_aware"

    def __init__(
        self,
        include_page_markers: bool = True,
        page_token_template: str = "[PAGE_{page}]",
        line_token_template: str = "[LINE_{line_id}]",
        y_threshold: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_page_markers = include_page_markers
        self.page_token_template = page_token_template
        self.line_token_template = line_token_template
        self.y_threshold = y_threshold

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_index = {id(token): i for i, token in enumerate(doc.tokens)}
        lines = line_group_tokens(doc.tokens, y_threshold=self.y_threshold)

        items: list[SerializedItem] = []
        previous_page = None

        for line_id, line in enumerate(lines):
            if not line:
                continue

            page = line[0].page

            if self.include_page_markers and page != previous_page:
                items.append(
                    self.layout_item(
                        self.page_token_template.format(page=page),
                        role="page",
                        page=page,
                    )
                )
                previous_page = page

            items.append(
                self.layout_item(
                    self.line_token_template.format(line_id=line_id),
                    role="line",
                    page=page,
                    line_id=line_id,
                    y_center=_line_center_y(line),
                )
            )

            for token in line:
                idx = token_index[id(token)]
                items.append(
                    self.token_item(
                        idx,
                        token.text,
                        page=token.page,
                        line_id=line_id,
                    )
                )

        return items


def _line_center_y(tokens: list[OCRToken]) -> float | None:
    if not tokens:
        return None
    return sum(t.center_y for t in tokens) / len(tokens)
