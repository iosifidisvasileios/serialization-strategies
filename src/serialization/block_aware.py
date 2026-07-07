from __future__ import annotations

from typing import Optional

from .base import BaseSerializer, SerializedItem
from ..preprocessing.schema import CanonicalDocument, OCRBlock, OCRToken


class BlockAwareSerializer(BaseSerializer):
    """Insert block markers before OCR blocks.

    This serializer supports two cases:
        1. tokens already have token.block_id set;
        2. block ids are inferred from doc.blocks by span overlap.
    """

    name = "block_aware"

    def __init__(
        self,
        include_page_markers: bool = True,
        page_token_template: str = "[PAGE_{page}]",
        block_token_template: str = "[BLOCK_{block_id}]",
        fallback_block_token: str = "[BLOCK_UNK]",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_page_markers = include_page_markers
        self.page_token_template = page_token_template
        self.block_token_template = block_token_template
        self.fallback_block_token = fallback_block_token

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_block_ids = infer_token_block_ids(doc.tokens, doc.blocks)
        items: list[SerializedItem] = []
        previous_page = None
        previous_block = object()

        for i, token in enumerate(doc.tokens):
            block_id = token_block_ids[i]

            if self.include_page_markers and token.page != previous_page:
                items.append(
                    self.layout_item(
                        self.page_token_template.format(page=token.page),
                        role="page",
                        page=token.page,
                    )
                )
                previous_page = token.page
                previous_block = object()

            if block_id != previous_block:
                if block_id is None:
                    block_text = self.fallback_block_token
                else:
                    block_text = self.block_token_template.format(block_id=block_id)

                items.append(
                    self.layout_item(
                        block_text,
                        role="block",
                        page=token.page,
                        block_id=block_id,
                    )
                )
                previous_block = block_id

            items.append(
                self.token_item(
                    i,
                    token.text,
                    page=token.page,
                    block_id=block_id,
                )
            )

        return items


def infer_token_block_ids(
    tokens: list[OCRToken],
    blocks: list[OCRBlock],
) -> list[Optional[int]]:
    """Infer each token's block id from existing token/block metadata."""
    if not tokens:
        return []

    if all(token.block_id is not None for token in tokens):
        return [token.block_id for token in tokens]

    blocks_by_page: dict[int, list[OCRBlock]] = {}
    for block in blocks or []:
        blocks_by_page.setdefault(block.page, []).append(block)

    for page_blocks in blocks_by_page.values():
        page_blocks.sort(key=lambda b: (b.start, b.end, b.bbox[1], b.bbox[0]))

    output: list[Optional[int]] = []

    for token in tokens:
        if token.block_id is not None:
            output.append(token.block_id)
            continue

        candidates = []
        for block in blocks_by_page.get(token.page, []):
            overlap = max(0, min(token.end, block.end) - max(token.start, block.start))
            if overlap > 0:
                candidates.append((overlap, block.block_id))

        if candidates:
            candidates.sort(reverse=True)
            output.append(candidates[0][1])
        else:
            output.append(None)

    return output
