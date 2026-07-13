from __future__ import annotations

from collections import defaultdict

from .base import BaseSerializer, SerializedItem
from ..preprocessing.ocr_utils import line_group_tokens
from ..preprocessing.schema import CanonicalDocument, OCRToken


class PrecedenceGraphOrderSerializer(BaseSerializer):
    """Order visual lines with simple above/below and left/right precedence edges."""

    name = "precedence_graph_order"

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_indices = {id(token): index for index, token in enumerate(doc.tokens)}
        lines = order_lines_by_precedence(line_group_tokens(doc.tokens))
        items: list[SerializedItem] = []
        previous_page = None

        for line_id, line in enumerate(lines):
            if not line:
                continue
            page = line[0].page
            if page != previous_page:
                items.append(self.layout_item(f"[PAGE_{page}]", role="page", page=page))
                previous_page = page
            items.append(
                self.layout_item(
                    f"[ORDER_LINE_{line_id}]",
                    role="line",
                    page=page,
                    line_id=line_id,
                )
            )
            for token in line:
                items.append(
                    self.token_item(
                        token_indices[id(token)], token.text, page=page, line_id=line_id
                    )
                )
        return items


def order_lines_by_precedence(lines: list[list[OCRToken]]) -> list[list[OCRToken]]:
    """Topologically order lines using geometry, with deterministic OCR-offset ties."""
    by_page: dict[int, list[list[OCRToken]]] = defaultdict(list)
    for line in lines:
        if line:
            by_page[line[0].page].append(line)

    output: list[list[OCRToken]] = []
    for page in sorted(by_page):
        page_lines = by_page[page]
        boxes = [_line_bbox(line) for line in page_lines]
        edges = {index: set() for index in range(len(page_lines))}
        indegree = [0] * len(page_lines)

        for left in range(len(page_lines)):
            for right in range(left + 1, len(page_lines)):
                relation = _precedence(boxes[left], boxes[right])
                if relation is None:
                    continue
                before, after = (left, right) if relation < 0 else (right, left)
                if after not in edges[before]:
                    edges[before].add(after)
                    indegree[after] += 1

        remaining = set(range(len(page_lines)))
        while remaining:
            ready = [index for index in remaining if indegree[index] == 0]
            if not ready:  # Defensive deterministic cycle break.
                ready = list(remaining)
            current = min(ready, key=lambda index: _line_sort_key(page_lines[index]))
            remaining.remove(current)
            output.append(page_lines[current])
            for follower in edges[current]:
                indegree[follower] -= 1
    return output


def _line_bbox(line: list[OCRToken]) -> tuple[int, int, int, int]:
    return (
        min(token.x0 for token in line),
        min(token.y0 for token in line),
        max(token.x1 for token in line),
        max(token.y1 for token in line),
    )


def _precedence(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int | None:
    """Return -1 when first precedes second, +1 for the reverse, else None."""
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    horizontal_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
    vertical_overlap = max(0, min(ay1, by1) - max(ay0, by0))
    min_height = max(1, min(ay1 - ay0, by1 - by0))

    if horizontal_overlap > 0 and ay1 <= by0:
        return -1
    if horizontal_overlap > 0 and by1 <= ay0:
        return 1
    if vertical_overlap / min_height >= 0.5:
        if ax1 <= bx0:
            return -1
        if bx1 <= ax0:
            return 1
    return None


def _line_sort_key(line: list[OCRToken]) -> tuple[int, int, int]:
    return (
        min(token.y0 for token in line),
        min(token.x0 for token in line),
        min(token.start for token in line),
    )
