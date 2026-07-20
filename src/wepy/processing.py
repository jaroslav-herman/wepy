"""
In-place transformations on :class:`~elchem.dataset.ExperimentDataset` objects.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .dataset import ExperimentDataset, SampleDataset

_DEFAULT_CURRENT_PRIORITY: tuple[str, ...] = ("i_ma", "i_a")


def split_dataframe_by_column(
    df: pd.DataFrame, column: str = "cycle_number"
) -> list[pd.DataFrame]:
    """
    Split *df* into one dataframe per distinct value of *column*.

    Rows where *column* is NaN are omitted from every group. Group order follows
    the first appearance of each distinct value in row order (``sort=False``
    groupby). If *column* is not in *df*, returns ``[df]`` unchanged.

    Parameters
    ----------
    df
        Input table.
    column
        Name of the grouping column (often ``\"cycle_number\"`` for CV data).

    Returns
    -------
    list[pd.DataFrame]
        Non-overlapping slices of *df* (copies), one per group key. Empty if
        *column* exists but no rows have finite values in that column.
    """
    if column not in df.columns:
        return [df]
    sub = df[df[column].notna()]
    if sub.empty:
        return []
    return [group.copy() for _, group in sub.groupby(column, sort=False)]


def _infer_jump_threshold(dx: np.ndarray) -> float:
    """
    Infer a step-size threshold from consecutive |?x| values.

    Uses the bulk of small steps to set a baseline, then lowers the threshold
    when a single step dominates (stitched LSV segments).
    """
    d = np.asarray(dx, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d > 1e-15]
    if d.size == 0:
        return float("inf")
    med = float(np.median(d))
    dmax = float(np.max(d))
    p75 = float(np.percentile(d, 75))
    d_small = d[d <= p75 + 1e-15]
    if d_small.size == 0:
        d_small = d
    p80_small = float(np.percentile(d_small, 80))
    base = max(med * 8.0, p80_small * 4.0, 1e-9)
    if dmax > base * 1.5:
        return max(base, dmax * 0.35)
    return base


def split_dataframe_by_jumps(
    df: pd.DataFrame,
    x: str = "ewe_v",
    threshold: float | None = None,
) -> list[pd.DataFrame]:
    """
    Split *df* into contiguous row blocks separated by large steps in *x*.

    Consecutive rows are compared in **dataframe row order** (not sorted by *x*).
    When ``threshold`` is ``None``, it is inferred from the distribution of
    ``|?x|`` between adjacent rows (see :func:`_infer_jump_threshold`).

    Parameters
    ----------
    df
        Measurement table.
    x
        Abscissa column (default ``ewe_v``).
    threshold
        If set, a new segment starts wherever ``abs(x[i] - x[i-1]) > threshold``.

    Returns
    -------
    list[pd.DataFrame]
        One or more non-overlapping copies of slices of *df*. Empty segments are
        omitted. A single-row dataframe yields one segment.
    """
    if x not in df.columns:
        raise KeyError(f"split_dataframe_by_jumps: column {x!r} not in dataframe.")
    xv = pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    if n <= 1:
        return [df.copy()]
    dx = np.abs(np.diff(xv))
    thr = float(threshold) if threshold is not None else _infer_jump_threshold(dx)
    if not np.isfinite(thr) or thr <= 0:
        return [df.copy()]
    boundaries = [0]
    for i in range(len(dx)):
        if np.isfinite(dx[i]) and dx[i] > thr:
            boundaries.append(i + 1)
    boundaries.append(n)
    out: list[pd.DataFrame] = []
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        if b > a:
            out.append(df.iloc[a:b].copy())
    return out if out else [df.iloc[0:0].copy()]


def split_dataframe_by_contiguous_runs(df: pd.DataFrame, column: str) -> list[pd.DataFrame]:
    """
    Split *df* wherever *column* changes from the previous row (row-order runs).

    Useful for ``ox_red``-style markers where each constant run is one sweep
    segment and value flips should not be connected by a line.
    """
    if column not in df.columns:
        raise KeyError(f"split_dataframe_by_contiguous_runs: column {column!r} not in dataframe.")
    change = df[column].ne(df[column].shift())
    block = change.cumsum()
    return [group.copy() for _, group in df.groupby(block, sort=False)]


def _effective_area_cm2(
    sample: SampleDataset,
    area_cm2: float | None,
) -> float:
    if area_cm2 is not None:
        return float(area_cm2)
    if sample.area_cm2 is not None:
        return float(sample.area_cm2)
    raise ValueError(
        f"Cannot compute current density for sample {sample.sample_id!r}: "
        "pass a positive ``area_cm2`` to ``add_current_density``, or set "
        "``area_cm2`` in the sample definition when calling ``load_samples`` "
        "(e.g. {{'path': ..., 'area_cm2': 0.196}})."
    )


def _pick_current_column(
    df: pd.DataFrame,
    current_column_priority: Sequence[str],
) -> tuple[str, str] | None:
    """Return (column_name, unit) with unit 'mA' or 'A'."""
    for name in current_column_priority:
        if name not in df.columns:
            continue
        if name == "i_ma":
            return name, "mA"
        if name == "i_a":
            return name, "A"
    return None


def add_current_density(
    dataset: ExperimentDataset,
    area_cm2: float | None = None,
    current_column_priority: Sequence[str] = _DEFAULT_CURRENT_PRIORITY,
    output_column: str = "j_ma_cm2",
) -> ExperimentDataset:
    """
    Add a current density column (mA per cm^2) to every measurement that has a
    recognized current column.

    Operates in place on each measurement's :class:`pandas.DataFrame` and
    returns the same ``dataset`` instance for chaining.

    Current is taken from the first matching column in
    ``current_column_priority`` (default: ``i_ma`` then ``i_a``). Original
    current columns are never modified.

    Parameters
    ----------
    dataset
        Loaded experiment data.
    area_cm2
        Electrode geometric area in cm^2, applied to all samples when given.
        If ``None``, each sample uses ``sample.area_cm2``; if that is also
        missing for a sample, :class:`ValueError` is raised.
    current_column_priority
        Canonical column names to try in order. Supported sources: ``i_ma``
        (milliamps) and ``i_a`` (amps); output is always mA per cm^2.
    output_column
        Name of the added column (default ``j_ma_cm2``).

    Returns
    -------
    ExperimentDataset
        The same ``dataset`` object.

    Examples
    --------
    ``dataset = add_current_density(dataset, area_cm2=0.196)`` then
    ``df.plot(x=\"ewe_v\", y=\"j_ma_cm2\")`` on a measurement dataframe.
    """
    priority = tuple(current_column_priority)
    for sample in dataset.samples.values():
        eff = _effective_area_cm2(sample, area_cm2)
        if eff <= 0:
            raise ValueError(
                f"area_cm2 must be positive for sample {sample.sample_id!r}; got {eff!r}."
            )

        for m in sample.measurements:
            df = m.dataframe
            picked = _pick_current_column(df, priority)
            if picked is None:
                continue
            col, unit = picked
            current = pd.to_numeric(df[col], errors="coerce")
            if unit == "mA":
                j = current / eff
            else:
                j = (current * 1000.0) / eff
            df[output_column] = j

    return dataset
