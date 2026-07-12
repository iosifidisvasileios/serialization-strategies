from __future__ import annotations

from statistics import median

from .base import BaseSerializer, SerializedItem
from ..preprocessing.ocr_utils import line_group_tokens
from ..preprocessing.schema import CanonicalDocument, OCRToken


class _KeyValueBaseSerializer(BaseSerializer):
    def _page_marker(self, page: int) -> SerializedItem:
        return self.layout_item(f"[PAGE_{page}]", role="page", page=page)

    def _line_items(
        self,
        line: list[OCRToken],
        token_indices: dict[int, int],
        line_id: int,
    ) -> list[SerializedItem]:
        page = line[0].page
        items = [
            self.layout_item(
                f"[LINE_{line_id}]", role="line", page=page, line_id=line_id
            )
        ]
        items.extend(
            self.token_item(token_indices[id(token)], token.text, page=page, line_id=line_id)
            for token in line
        )
        return items

    def _pair_items(
        self,
        key_tokens: list[OCRToken],
        value_tokens: list[OCRToken],
        token_indices: dict[int, int],
        line_id: int,
        pair_id: int,
        orientation: str,
    ) -> list[SerializedItem]:
        page = key_tokens[0].page
        items = [
            self.layout_item(
                f"[LINE_{line_id}]", role="line", page=page, line_id=line_id
            ),
            self.layout_item(
                f"[FIELD_{pair_id}]",
                role="field",
                page=page,
                pair_id=pair_id,
                orientation=orientation,
            ),
            self.layout_item("[KEY]", role="key", page=page, pair_id=pair_id),
        ]
        items.extend(
            self.token_item(
                token_indices[id(token)], token.text, page=page, pair_id=pair_id, kv_role="key"
            )
            for token in key_tokens
        )
        items.append(self.layout_item("[VALUE]", role="value", page=page, pair_id=pair_id))
        items.extend(
            self.token_item(
                token_indices[id(token)], token.text, page=page, pair_id=pair_id, kv_role="value"
            )
            for token in value_tokens
        )
        return items


class KeyValueRowPairsSerializer(_KeyValueBaseSerializer):
    """Split visual lines at a reliable large gap into key/value candidates."""

    name = "key_value_row_pairs"

    def __init__(self, min_gap_ratio: float = 0.04, gap_height_multiplier: float = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_gap_ratio = min_gap_ratio
        self.gap_height_multiplier = gap_height_multiplier

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_indices = {id(token): index for index, token in enumerate(doc.tokens)}
        lines = line_group_tokens(doc.tokens)
        items: list[SerializedItem] = []
        previous_page = None
        pair_id = 0

        for line_id, line in enumerate(lines):
            if not line:
                continue
            page = line[0].page
            if page != previous_page:
                items.append(self._page_marker(page))
                previous_page = page
            split = split_line_on_large_gap(
                line,
                min_gap_ratio=self.min_gap_ratio,
                gap_height_multiplier=self.gap_height_multiplier,
            )
            if split is None:
                items.extend(self._line_items(line, token_indices, line_id))
                continue
            key_tokens, value_tokens = split
            items.extend(
                self._pair_items(
                    key_tokens, value_tokens, token_indices, line_id, pair_id, "right"
                )
            )
            pair_id += 1
        return items


class KeyValueAnchorPairsSerializer(_KeyValueBaseSerializer):
    """Pair colon/equal anchored labels with text to their right or directly below."""

    name = "key_value_anchor_pairs"

    def __init__(self, max_below_gap_heights: float = 4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_below_gap_heights = max_below_gap_heights

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_indices = {id(token): index for index, token in enumerate(doc.tokens)}
        lines = line_group_tokens(doc.tokens)
        items: list[SerializedItem] = []
        consumed: set[int] = set()
        previous_page = None
        pair_id = 0

        for line_id, line in enumerate(lines):
            if line_id in consumed or not line:
                continue
            page = line[0].page
            if page != previous_page:
                items.append(self._page_marker(page))
                previous_page = page
            anchor = _anchor_position(line)
            if anchor is None:
                items.extend(self._line_items(line, token_indices, line_id))
                continue

            key_tokens = line[: anchor + 1]
            value_tokens = line[anchor + 1 :]
            orientation = "right"
            if not value_tokens:
                below_id = _aligned_line_below(
                    lines,
                    line_id,
                    key_tokens,
                    consumed,
                    max_gap_heights=self.max_below_gap_heights,
                )
                if below_id is not None:
                    consumed.add(below_id)
                    value_tokens = lines[below_id]
                    orientation = "below"
            if not value_tokens:
                items.extend(self._line_items(line, token_indices, line_id))
                continue

            items.extend(
                self._pair_items(
                    key_tokens, value_tokens, token_indices, line_id, pair_id, orientation
                )
            )
            pair_id += 1
        return items


def split_line_on_large_gap(
    line: list[OCRToken], min_gap_ratio: float = 0.04, gap_height_multiplier: float = 2.0
) -> tuple[list[OCRToken], list[OCRToken]] | None:
    if len(line) < 2:
        return None
    ordered = sorted(line, key=lambda token: (token.x0, token.start))
    page_width = next((token.page_width for token in ordered if token.page_width), None)
    if page_width is None:
        page_width = max(token.x1 for token in ordered) - min(token.x0 for token in ordered)
    median_height = median(max(1, token.height) for token in ordered)
    minimum_gap = max(float(page_width) * min_gap_ratio, median_height * gap_height_multiplier)
    gaps = [ordered[index + 1].x0 - ordered[index].x1 for index in range(len(ordered) - 1)]
    split_index = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[split_index] < minimum_gap:
        return None
    return (ordered[: split_index + 1], ordered[split_index + 1 :])


def _anchor_position(line: list[OCRToken]) -> int | None:
    for index, token in enumerate(line):
        text = token.text.strip()
        if text in {":", "="} or text.endswith(":"):
            return index
    return None


def _aligned_line_below(
    lines: list[list[OCRToken]],
    line_id: int,
    key_tokens: list[OCRToken],
    consumed: set[int],
    max_gap_heights: float,
) -> int | None:
    page = key_tokens[0].page
    key_x0 = min(token.x0 for token in key_tokens)
    key_y1 = max(token.y1 for token in key_tokens)
    key_height = max(1, max(token.y1 for token in key_tokens) - min(token.y0 for token in key_tokens))
    candidates = []
    for candidate_id in range(line_id + 1, len(lines)):
        if candidate_id in consumed or not lines[candidate_id]:
            continue
        candidate = lines[candidate_id]
        if candidate[0].page != page:
            break
        vertical_gap = min(token.y0 for token in candidate) - key_y1
        horizontal_delta = abs(min(token.x0 for token in candidate) - key_x0)
        if 0 <= vertical_gap <= key_height * max_gap_heights and horizontal_delta <= key_height * 3:
            candidates.append((vertical_gap, horizontal_delta, candidate_id))
    return min(candidates)[2] if candidates else None
