from __future__ import annotations

from .base import BaseSerializer, SerializedItem
from ..preprocessing.schema import CanonicalDocument


class PageAwareSerializer(BaseSerializer):
    """Insert a page marker before tokens from each page.

    Example:
        [PAGE_0] token token [PAGE_1] token ...
    """

    name = "page_aware"

    def __init__(
        self,
        page_token_template: str = "[PAGE_{page}]",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.page_token_template = page_token_template

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        items: list[SerializedItem] = []
        previous_page = None

        for i, token in enumerate(doc.tokens):
            if token.page != previous_page:
                items.append(
                    self.layout_item(
                        self.page_token_template.format(page=token.page),
                        role="page",
                        page=token.page,
                    )
                )
                previous_page = token.page

            items.append(self.token_item(i, token.text, page=token.page))

        return items
