from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Optional

from .base import BaseSerializer, SerializedItem
from ..preprocessing.ocr_utils import group_tokens_by_page, line_group_tokens
from ..preprocessing.schema import CanonicalDocument, OCRToken


@dataclass(slots=True)
class ColumnRegion:
    """Detected vertical column region on a page."""

    column_id: int
    page: int
    x0: int
    x1: int
    token_indices: list[int]


class ColumnAwareSerializer(BaseSerializer):
    """Serialize each page by detected visual columns.

    This is a deterministic reading-order reconstruction baseline.
    It groups tokens into coarse page columns, emits one column marker per
    detected column, then serializes tokens inside each column top-to-bottom
    and left-to-right.

    Synthetic tokens are ignored by the token-classification loss.
    Real OCR tokens keep their original source_token_index so predictions can
    always be collapsed back to the canonical token stream.
    """

    name = "column_aware"

    def __init__(
        self,
        include_page_markers: bool = True,
        include_column_markers: bool = True,
        page_token_template: str = "[PAGE_{page}]",
        column_token_template: str = "[COL_{column_id}]",
        max_columns: int = 4,
        min_tokens_per_column: int = 8,
        min_gap_ratio: float = 0.08,
        gap_width_multiplier: float = 2.0,
        use_line_order_within_column: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_page_markers = include_page_markers
        self.include_column_markers = include_column_markers
        self.page_token_template = page_token_template
        self.column_token_template = column_token_template
        self.max_columns = max_columns
        self.min_tokens_per_column = min_tokens_per_column
        self.min_gap_ratio = min_gap_ratio
        self.gap_width_multiplier = gap_width_multiplier
        self.use_line_order_within_column = use_line_order_within_column

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_indices_by_id = {id(token): i for i, token in enumerate(doc.tokens)}
        page_tokens = group_tokens_by_page(doc.tokens)
        items: list[SerializedItem] = []

        for page in sorted(page_tokens):
            tokens = page_tokens[page]
            if not tokens:
                continue

            if self.include_page_markers:
                items.append(
                    self.layout_item(
                        self.page_token_template.format(page=page),
                        role="page",
                        page=page,
                    )
                )

            page_indices = [token_indices_by_id[id(token)] for token in tokens]
            columns = detect_column_regions(
                doc.tokens,
                page_indices,
                max_columns=self.max_columns,
                min_tokens_per_column=self.min_tokens_per_column,
                min_gap_ratio=self.min_gap_ratio,
                gap_width_multiplier=self.gap_width_multiplier,
            )

            for column in columns:
                if self.include_column_markers:
                    items.append(
                        self.layout_item(
                            self.column_token_template.format(column_id=column.column_id),
                            role="column",
                            page=page,
                            column_id=column.column_id,
                            x0=column.x0,
                            x1=column.x1,
                            n_tokens=len(column.token_indices),
                        )
                    )

                ordered_indices = order_indices_within_column(
                    doc.tokens,
                    column.token_indices,
                    use_line_order=self.use_line_order_within_column,
                )

                for idx in ordered_indices:
                    token = doc.tokens[idx]
                    items.append(
                        self.token_item(
                            idx,
                            token.text,
                            page=token.page,
                            column_id=column.column_id,
                            column_x0=column.x0,
                            column_x1=column.x1,
                        )
                    )

        return items


def detect_column_regions(
    tokens: list[OCRToken],
    token_indices: list[int],
    max_columns: int = 4,
    min_tokens_per_column: int = 8,
    min_gap_ratio: float = 0.08,
    gap_width_multiplier: float = 2.0,
) -> list[ColumnRegion]:
    """Detect coarse vertical columns using large horizontal whitespace gaps."""
    if not token_indices:
        return []

    if len(token_indices) < min_tokens_per_column * 2:
        return [_single_column(tokens, token_indices, column_id=0)]

    page = tokens[token_indices[0]].page
    page_width = _page_width(tokens, token_indices)
    widths = [max(1, tokens[i].width) for i in token_indices]
    median_width = median(widths) if widths else 1.0

    # Iteratively split regions on the largest reliable horizontal gap.
    regions = [list(token_indices)]

    while len(regions) < max_columns:
        best = None

        for region_id, region in enumerate(regions):
            candidate = _best_horizontal_split(
                tokens,
                region,
                page_width=page_width,
                min_tokens_per_side=min_tokens_per_column,
                min_gap=max(page_width * min_gap_ratio, median_width * gap_width_multiplier),
            )
            if candidate is None:
                continue

            gap_size, left_indices, right_indices = candidate
            if best is None or gap_size > best[0]:
                best = (gap_size, region_id, left_indices, right_indices)

        if best is None:
            break

        _, region_id, left_indices, right_indices = best
        regions.pop(region_id)
        regions.extend([left_indices, right_indices])
        regions.sort(key=lambda idxs: min(tokens[i].x0 for i in idxs))

    columns = []
    for col_id, indices in enumerate(regions):
        x0 = min(tokens[i].x0 for i in indices)
        x1 = max(tokens[i].x1 for i in indices)
        columns.append(
            ColumnRegion(
                column_id=col_id,
                page=page,
                x0=x0,
                x1=x1,
                token_indices=indices,
            )
        )

    columns.sort(key=lambda c: (c.x0, c.x1))
    for i, col in enumerate(columns):
        col.column_id = i

    return columns


def _best_horizontal_split(
    tokens: list[OCRToken],
    token_indices: list[int],
    page_width: int,
    min_tokens_per_side: int,
    min_gap: float,
) -> Optional[tuple[float, list[int], list[int]]]:
    if len(token_indices) < min_tokens_per_side * 2:
        return None

    intervals = sorted((tokens[i].x0, tokens[i].x1) for i in token_indices)
    merged = _merge_intervals(intervals, tolerance=0)

    if len(merged) < 2:
        return None

    best_gap = None
    for (left_start, left_end), (right_start, right_end) in zip(merged, merged[1:]):
        gap = right_start - left_end
        if gap < min_gap:
            continue

        split_x = (left_end + right_start) / 2.0
        left_indices = [i for i in token_indices if tokens[i].center_x < split_x]
        right_indices = [i for i in token_indices if tokens[i].center_x >= split_x]

        if len(left_indices) < min_tokens_per_side or len(right_indices) < min_tokens_per_side:
            continue

        # Reject cuts that are tiny relative to the page even if token widths are small.
        if page_width > 0 and gap / page_width < 0.02:
            continue

        if best_gap is None or gap > best_gap[0]:
            best_gap = (float(gap), left_indices, right_indices)

    return best_gap


def _merge_intervals(intervals: list[tuple[int, int]], tolerance: int = 0) -> list[tuple[int, int]]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + tolerance:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _single_column(
    tokens: list[OCRToken],
    token_indices: list[int],
    column_id: int,
) -> ColumnRegion:
    page = tokens[token_indices[0]].page if token_indices else 0
    return ColumnRegion(
        column_id=column_id,
        page=page,
        x0=min(tokens[i].x0 for i in token_indices) if token_indices else 0,
        x1=max(tokens[i].x1 for i in token_indices) if token_indices else 0,
        token_indices=list(token_indices),
    )


def order_indices_within_column(
    tokens: list[OCRToken],
    token_indices: list[int],
    use_line_order: bool = True,
) -> list[int]:
    if not use_line_order:
        return sorted(token_indices, key=lambda i: (tokens[i].page, tokens[i].y0, tokens[i].x0, tokens[i].start))

    selected_tokens = [tokens[i] for i in token_indices]
    id_to_index = {id(token): i for i, token in zip(token_indices, selected_tokens)}
    ordered = []

    for line in line_group_tokens(selected_tokens):
        for token in line:
            ordered.append(id_to_index[id(token)])

    return ordered


def _page_width(tokens: list[OCRToken], token_indices: list[int]) -> int:
    for idx in token_indices:
        if tokens[idx].page_width:
            return int(tokens[idx].page_width)
    return max(tokens[i].x1 for i in token_indices) - min(tokens[i].x0 for i in token_indices)
