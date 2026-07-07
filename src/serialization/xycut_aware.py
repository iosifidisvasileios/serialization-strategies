from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .base import BaseSerializer, SerializedItem
from ..preprocessing.ocr_utils import group_tokens_by_page, line_group_tokens
from ..preprocessing.schema import CanonicalDocument, OCRToken


@dataclass(slots=True)
class XYCutRegion:
    """Leaf or internal XY-cut region."""

    region_id: int
    page: int
    token_indices: list[int]
    bbox: list[int]
    depth: int
    split_axis: Optional[str] = None
    children: list["XYCutRegion"] = field(default_factory=list)


class XYCutAwareSerializer(BaseSerializer):
    """Serialize tokens using a deterministic recursive XY-cut order.

    The page is recursively split along large whitespace gaps. Leaf regions are
    emitted in reading order. This provides a non-LayoutLM baseline for testing
    whether layout-aware reading-order reconstruction improves token
    classification.
    """

    name = "xycut_aware"

    def __init__(
        self,
        include_page_markers: bool = True,
        include_region_markers: bool = True,
        page_token_template: str = "[PAGE_{page}]",
        region_token_template: str = "[REGION_{region_id}]",
        max_depth: int = 6,
        min_tokens_per_region: int = 8,
        min_gap_ratio_x: float = 0.06,
        min_gap_ratio_y: float = 0.035,
        prefer_horizontal: bool = True,
        use_line_order_within_region: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_page_markers = include_page_markers
        self.include_region_markers = include_region_markers
        self.page_token_template = page_token_template
        self.region_token_template = region_token_template
        self.max_depth = max_depth
        self.min_tokens_per_region = min_tokens_per_region
        self.min_gap_ratio_x = min_gap_ratio_x
        self.min_gap_ratio_y = min_gap_ratio_y
        self.prefer_horizontal = prefer_horizontal
        self.use_line_order_within_region = use_line_order_within_region

    def make_items(self, doc: CanonicalDocument) -> list[SerializedItem]:
        token_indices_by_id = {id(token): i for i, token in enumerate(doc.tokens)}
        page_tokens = group_tokens_by_page(doc.tokens)
        items: list[SerializedItem] = []
        next_region_id = 0

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
            region, next_region_id = build_xycut_tree(
                doc.tokens,
                page_indices,
                next_region_id=next_region_id,
                depth=0,
                max_depth=self.max_depth,
                min_tokens_per_region=self.min_tokens_per_region,
                min_gap_ratio_x=self.min_gap_ratio_x,
                min_gap_ratio_y=self.min_gap_ratio_y,
                prefer_horizontal=self.prefer_horizontal,
            )

            for leaf in iter_xycut_leaves(region):
                if self.include_region_markers:
                    items.append(
                        self.layout_item(
                            self.region_token_template.format(region_id=leaf.region_id),
                            role="xycut_region",
                            page=page,
                            region_id=leaf.region_id,
                            depth=leaf.depth,
                            bbox=leaf.bbox,
                            n_tokens=len(leaf.token_indices),
                        )
                    )

                ordered_indices = order_indices_within_region(
                    doc.tokens,
                    leaf.token_indices,
                    use_line_order=self.use_line_order_within_region,
                )
                for idx in ordered_indices:
                    token = doc.tokens[idx]
                    items.append(
                        self.token_item(
                            idx,
                            token.text,
                            page=token.page,
                            region_id=leaf.region_id,
                            region_bbox=leaf.bbox,
                            region_depth=leaf.depth,
                        )
                    )

        return items


def build_xycut_tree(
    tokens: list[OCRToken],
    token_indices: list[int],
    next_region_id: int = 0,
    depth: int = 0,
    max_depth: int = 6,
    min_tokens_per_region: int = 8,
    min_gap_ratio_x: float = 0.06,
    min_gap_ratio_y: float = 0.035,
    prefer_horizontal: bool = True,
) -> tuple[XYCutRegion, int]:
    page = tokens[token_indices[0]].page if token_indices else 0
    bbox = _region_bbox(tokens, token_indices)
    region_id = next_region_id
    next_region_id += 1

    region = XYCutRegion(
        region_id=region_id,
        page=page,
        token_indices=list(token_indices),
        bbox=bbox,
        depth=depth,
    )

    if depth >= max_depth or len(token_indices) < min_tokens_per_region * 2:
        return region, next_region_id

    split = find_xycut_split(
        tokens,
        token_indices,
        bbox=bbox,
        min_tokens_per_side=min_tokens_per_region,
        min_gap_ratio_x=min_gap_ratio_x,
        min_gap_ratio_y=min_gap_ratio_y,
        prefer_horizontal=prefer_horizontal,
    )

    if split is None:
        return region, next_region_id

    axis, first_indices, second_indices = split
    region.split_axis = axis

    child_a, next_region_id = build_xycut_tree(
        tokens,
        first_indices,
        next_region_id=next_region_id,
        depth=depth + 1,
        max_depth=max_depth,
        min_tokens_per_region=min_tokens_per_region,
        min_gap_ratio_x=min_gap_ratio_x,
        min_gap_ratio_y=min_gap_ratio_y,
        prefer_horizontal=prefer_horizontal,
    )
    child_b, next_region_id = build_xycut_tree(
        tokens,
        second_indices,
        next_region_id=next_region_id,
        depth=depth + 1,
        max_depth=max_depth,
        min_tokens_per_region=min_tokens_per_region,
        min_gap_ratio_x=min_gap_ratio_x,
        min_gap_ratio_y=min_gap_ratio_y,
        prefer_horizontal=prefer_horizontal,
    )

    region.children = [child_a, child_b]
    return region, next_region_id


def find_xycut_split(
    tokens: list[OCRToken],
    token_indices: list[int],
    bbox: list[int],
    min_tokens_per_side: int,
    min_gap_ratio_x: float,
    min_gap_ratio_y: float,
    prefer_horizontal: bool,
) -> Optional[tuple[str, list[int], list[int]]]:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])

    x_candidate = _best_axis_split(
        tokens,
        token_indices,
        axis="x",
        region_size=width,
        min_gap_ratio=min_gap_ratio_x,
        min_tokens_per_side=min_tokens_per_side,
    )
    y_candidate = _best_axis_split(
        tokens,
        token_indices,
        axis="y",
        region_size=height,
        min_gap_ratio=min_gap_ratio_y,
        min_tokens_per_side=min_tokens_per_side,
    )

    if x_candidate is None and y_candidate is None:
        return None
    if x_candidate is None:
        return "y", y_candidate[1], y_candidate[2]
    if y_candidate is None:
        return "x", x_candidate[1], x_candidate[2]

    x_score = x_candidate[0] / max(1, width)
    y_score = y_candidate[0] / max(1, height)

    if prefer_horizontal and y_score >= x_score * 0.85:
        return "y", y_candidate[1], y_candidate[2]
    if x_score > y_score:
        return "x", x_candidate[1], x_candidate[2]
    return "y", y_candidate[1], y_candidate[2]


def _best_axis_split(
    tokens: list[OCRToken],
    token_indices: list[int],
    axis: str,
    region_size: int,
    min_gap_ratio: float,
    min_tokens_per_side: int,
) -> Optional[tuple[float, list[int], list[int]]]:
    if len(token_indices) < min_tokens_per_side * 2:
        return None

    if axis == "x":
        intervals = sorted((tokens[i].x0, tokens[i].x1) for i in token_indices)
        center = lambda i: tokens[i].center_x
    elif axis == "y":
        intervals = sorted((tokens[i].y0, tokens[i].y1) for i in token_indices)
        center = lambda i: tokens[i].center_y
    else:
        raise ValueError("axis must be 'x' or 'y'")

    merged = _merge_intervals(intervals, tolerance=0)
    if len(merged) < 2:
        return None

    min_gap = region_size * min_gap_ratio
    best = None

    for (a0, a1), (b0, b1) in zip(merged, merged[1:]):
        gap = b0 - a1
        if gap < min_gap:
            continue

        split_value = (a1 + b0) / 2.0
        first = [i for i in token_indices if center(i) < split_value]
        second = [i for i in token_indices if center(i) >= split_value]

        if len(first) < min_tokens_per_side or len(second) < min_tokens_per_side:
            continue

        if best is None or gap > best[0]:
            best = (float(gap), first, second)

    return best


def iter_xycut_leaves(region: XYCutRegion) -> list[XYCutRegion]:
    if not region.children:
        return [region]

    leaves: list[XYCutRegion] = []
    for child in region.children:
        leaves.extend(iter_xycut_leaves(child))
    return leaves


def order_indices_within_region(
    tokens: list[OCRToken],
    token_indices: list[int],
    use_line_order: bool = True,
) -> list[int]:
    if not use_line_order:
        return sorted(token_indices, key=lambda i: (tokens[i].page, tokens[i].y0, tokens[i].x0, tokens[i].start))

    selected = [tokens[i] for i in token_indices]
    token_to_index = {id(token): i for i, token in zip(token_indices, selected)}
    ordered = []
    for line in line_group_tokens(selected):
        for token in line:
            ordered.append(token_to_index[id(token)])
    return ordered


def _region_bbox(tokens: list[OCRToken], token_indices: list[int]) -> list[int]:
    if not token_indices:
        return [0, 0, 0, 0]
    return [
        min(tokens[i].x0 for i in token_indices),
        min(tokens[i].y0 for i in token_indices),
        max(tokens[i].x1 for i in token_indices),
        max(tokens[i].y1 for i in token_indices),
    ]


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
