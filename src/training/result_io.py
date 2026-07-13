from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_csv_rows(
    rows: pd.DataFrame, path: Path, key_columns: list[str] | None = None
) -> pd.DataFrame:
    """Append rows to a shared CSV, replacing an earlier row with the same key."""
    frames = []
    if path.exists():
        frames.append(pd.read_csv(path))
    if not rows.empty:
        frames.append(rows)
    if not frames:
        return rows

    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = [column for column in (key_columns or []) if column in merged.columns]
    merged = merged.drop_duplicates(subset=keys or None, keep="last")
    merged.to_csv(path, index=False)
    return merged
