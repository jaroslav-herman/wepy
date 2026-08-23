"""
Matplotlib helpers for :class:`~elchem.dataset.Measurement` objects.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .dataset import ExperimentDataset, Measurement, SampleDataset, select_samples
from .processing import (
    split_dataframe_by_column,
    split_dataframe_by_contiguous_runs,
    split_dataframe_by_jumps,
)
from .plots import apply_plot_style, style_legend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

MeasurementLike = Measurement | ExperimentDataset

# Publication-oriented defaults (column name -> axis label).
_AXIS_LABEL_MAP: dict[str, str] = {
    "ewe_v": "Voltage (V)",
    "ewe_rhe_v": "Voltage vs RHE (V)",
    "ewe_vs_rhe_v": "Voltage vs RHE (V)",
    "i_ma": "Current (mA)",
    "j_ma_cm2": r"Current density (mA cm$^{-2}$)",
    "time_s": "Time (s)",
    "re_z_ohm": "Re(Z) (\u03a9)",  # U+03A9 capital Omega
    "im_z_ohm": "Im(Z) (\u03a9)",
    "minus_im_z_ohm": "-Im(Z) (\u03a9)",
    "freq_hz": "Frequency (Hz)",
    "frequency_hz": "Frequency (Hz)",
}


def _axis_label(column: str) -> str:
    return _AXIS_LABEL_MAP.get(column, column.replace("_", " "))


def _is_plot_filter_all(val: Any) -> bool:
    """True if *val* means 'do not filter' (``None`` or ``\"all\"``, case-insensitive)."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() == "all":
        return True
    return False


def _style_legend(ax: Any, *, frameon: bool = False) -> None:
    leg = ax.get_legend()
    if leg is not None:
        style_legend(leg, frameon=frameon)


def _validate_xy_columns(df: pd.DataFrame, x: str, y: str, y2: str | None = None) -> None:
    need = [x, y]
    if y2 is not None:
        need.append(y2)
    missing = [c for c in need if c not in df.columns]
    if missing:
        avail = list(df.columns)
        raise KeyError(
            f"Column(s) not found in measurement.dataframe: {missing!r}. "
            f"Available columns: {avail!r}"
        )


def _segment_matches_cycle(
    segment_key: Any, cycle: int | float | Sequence[Any] | str | None
) -> bool:
    """True if *segment_key* is selected by *cycle* (scalar or collection)."""
    if _is_plot_filter_all(cycle):
        return True
    targets: list[Any] = (
        list(cycle) if isinstance(cycle, (list, tuple, set)) else [cycle]
    )
    for t in targets:
        if pd.isna(segment_key) and pd.isna(t):
            return True
        try:
            if segment_key == t or float(segment_key) == float(t):
                return True
        except (TypeError, ValueError):
            if segment_key == t:
                return True
    return False


def _str_or_sequence(val: str | Sequence[str] | Path | None) -> list[str] | None:
    if _is_plot_filter_all(val):
        return None
    if isinstance(val, (str, Path)):
        return [str(val)]
    return [str(v) for v in val]


def _int_or_sequence(val: int | Sequence[int] | None) -> list[int] | None:
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return [val]
    return [int(v) for v in val]


def _segment_indices_from_selector(sel: Any) -> set[int] | None:
    """Parse ``segment`` or ``cycle`` (alias) into 0-based segment indices."""
    if _is_plot_filter_all(sel):
        return None
    if isinstance(sel, bool):
        raise TypeError("segment / cycle index selector must not be a bool.")
    if isinstance(sel, (int, float)):
        return {int(sel)}
    return {int(x) for x in sel}


def _sorted_cycle_values(df: pd.DataFrame) -> list[Any]:
    if "cycle_number" not in df.columns:
        return []
    vals = df["cycle_number"].dropna().unique().tolist()

    def key(v: Any) -> tuple:
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    return sorted(vals, key=key)


def _resolve_plot_segments(
    df: pd.DataFrame,
    *,
    x: str,
    split_by: str,
    cycle: int | float | Sequence[Any] | str | None,
    segment: int | float | Sequence[Any] | str | None,
) -> list[pd.DataFrame]:
    """
    Build ordered segment dataframes for *split_by*, applying *cycle* or *segment*.

    For ``ox_red``, ``jumps``, or ``auto`` when it uses jumps, pass **segment** for
    0-based segment indices. **cycle** is accepted as a convenience alias when
    **segment** is omitted (e.g. ``plot_dataset(..., split_by="auto", cycle=1)``).
    For true ``cycle_number`` grouping, use **cycle** for cycle column values only.
    ``None`` or ``\"all\"`` for ``cycle`` / ``segment`` means no restriction.
    """
    if split_by not in ("cycle_number", "ox_red", "jumps", "auto"):
        raise ValueError(
            "split_by must be None, 'cycle_number', 'ox_red', 'jumps', or 'auto'; "
            f"got {split_by!r}."
        )

    cycle = None if _is_plot_filter_all(cycle) else cycle
    segment = None if _is_plot_filter_all(segment) else segment

    kind: str
    if split_by == "auto":
        if "cycle_number" in df.columns and df["cycle_number"].notna().any():
            segs = split_dataframe_by_column(df, "cycle_number")
            if not segs:
                raise ValueError(
                    "split_by='auto' selected cycle_number but no finite cycle values exist."
                )
            kind = "cycle_number"
        else:
            segs = split_dataframe_by_jumps(df, x=x)
            kind = "jumps"
    elif split_by == "cycle_number":
        segs = split_dataframe_by_column(df, "cycle_number")
        if not segs:
            raise ValueError(
                "split_by='cycle_number': no rows with finite cycle_number values."
            )
        kind = "cycle_number"
    elif split_by == "ox_red":
        if "ox_red" not in df.columns:
            raise ValueError(
                "split_by='ox_red' requires an 'ox_red' column in the measurement dataframe."
            )
        segs = split_dataframe_by_contiguous_runs(df, "ox_red")
        kind = "ox_red"
    else:
        segs = split_dataframe_by_jumps(df, x=x)
        kind = "jumps"

    if kind in ("jumps", "ox_red"):
        idx_src = segment if segment is not None else cycle
        wanted = _segment_indices_from_selector(idx_src)
        if wanted is None:
            return segs
        n = len(segs)
        bad = sorted(i for i in wanted if i < 0 or i >= n)
        if bad:
            avail = list(range(n))
            if len(bad) == 1:
                raise ValueError(
                    f"Requested segment index {bad[0]} not found. Available segment indices: {avail}"
                )
            raise ValueError(
                f"Requested segment indices {bad} not found. Available segment indices: {avail}"
            )
        return [segs[i] for i in sorted(wanted)]

    # cycle_number
    if segment is not None:
        raise ValueError(
            "segment= is only valid for split_by 'ox_red', 'jumps', or 'auto' when "
            "jump-style splitting is used. With cycle_number, use cycle= to pick "
            "cycle values."
        )
    if "cycle_number" not in df.columns:
        if cycle is not None:
            raise ValueError(
                f"Requested cycle {cycle!r} not found. No 'cycle_number' column in dataframe."
            )
        return segs
    if cycle is None:
        return segs
    filtered = [
        seg
        for seg in segs
        if _segment_matches_cycle(seg["cycle_number"].iloc[0], cycle)
    ]
    if not filtered:
        avail = _sorted_cycle_values(df)
        raise ValueError(
            f"Requested cycle {cycle!r} not found. Available cycles: {avail}"
        )
    return filtered


def _default_plot_label(measurement: Measurement) -> str:
    return measurement.source_path.stem


def _resolve_measurement(
    obj: MeasurementLike,
    sample_id: str | None,
    measurement_index: int,
) -> Measurement:
    """Normalize ``Measurement`` or pick one row from ``ExperimentDataset``."""
    if isinstance(obj, Measurement):
        return obj
    if isinstance(obj, ExperimentDataset):
        if not obj.samples:
            raise ValueError("ExperimentDataset has no samples to plot.")
        sid = sample_id
        if sid is None:
            keys = list(obj.samples.keys())
            if len(keys) == 1:
                sid = keys[0]
            else:
                raise ValueError(
                    "plot_measurement: pass a single Measurement, or pass ExperimentDataset "
                    "together with sample_id=... when there are multiple samples. "
                    f"Available sample ids: {keys!r}. Example: "
                    "dataset.samples['105'].measurements[0]"
                )
        if sid not in obj.samples:
            raise KeyError(
                f"Unknown sample_id {sid!r}. Available: {list(obj.samples.keys())!r}"
            )
        sample = obj.samples[sid]
        if not sample.measurements:
            raise ValueError(f"Sample {sid!r} has no measurements.")
        if measurement_index < 0 or measurement_index >= len(sample.measurements):
            raise IndexError(
                f"measurement_index={measurement_index} out of range for sample {sid!r} "
                f"(have {len(sample.measurements)} measurement(s))."
            )
        return sample.measurements[measurement_index]
    raise TypeError(
        "plot_measurement expected Measurement or ExperimentDataset; "
        f"got {type(obj).__name__!r}."
    )


def _sanitize_export_basename(raw: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", raw, flags=re.ASCII)
    s = re.sub(r"_+", "_", s).strip("._-")
    return (s or "figure")[:200]


def _default_export_basename(measurement: Measurement) -> str:
    parts: list[str] = []
    meta = measurement.metadata or {}
    sample = meta.get("sample") or {}
    sid = sample.get("id")
    if sid is not None and str(sid).strip():
        parts.append(str(sid).strip())
    if measurement.technique:
        parts.append(str(measurement.technique))
    stem = measurement.source_path.stem
    if stem:
        parts.append(stem)
    return _sanitize_export_basename("_".join(parts))


def _save_figure(fig: Any, path: str | Path, *, dpi: int = 300) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=dpi)


def _save_publication_pair(fig: Any, out_dir: Path, base: str) -> None:
    base = _sanitize_export_basename(base)
    _save_figure(fig, out_dir / f"{base}.png")
    _save_figure(fig, out_dir / f"{base}.pdf")


def plot_measurement(
    measurement: MeasurementLike,
    x: str = "ewe_v",
    y: str = "j_ma_cm2",
    ax: Any | None = None,
    label: str | None = None,
    title: str | None = None,
    marker: str | None = None,
    linestyle: str = "-",
    grid: bool = False,
    figsize: tuple[float, float] = (6, 4),
    save_path: str | Path | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    y2: str | None = None,
    y2label: str | None = None,
    label2: str | None = None,
    save: bool = False,
    path_OUT: str | Path | None = None,
    export_name: str | None = None,
    invert_y: bool = False,
    sample_id: str | None = None,
    measurement_index: int = 0,
    show_legend: bool = True,
    legend_frameon: bool = False,
    split_by: str | None = None,
    cycle: int | float | Sequence[Any] | str | None = None,
    segment: int | float | Sequence[Any] | str | None = None,
    technique: str | Sequence[str] | None = None,
    alpha: float = 1.0,
    **plot_kwargs: Any,
) -> tuple["Figure", "Axes", Any | None]:
    """
    Plot one column versus another from a measurement dataframe.

    **Recommended usage:** pass a concrete :class:`~elchem.dataset.Measurement`
    (for example ``dataset.samples[sample_id].measurements[i]``). That keeps
    the call explicit about which file and which sample you are plotting.

    **Convenience:** you may pass an :class:`~elchem.dataset.ExperimentDataset`
    instead. If the dataset contains **exactly one** sample, that sample's
    measurement at ``measurement_index`` (default ``0``) is plotted. If there
    are **multiple** samples, you **must** pass ``sample_id`` (and optionally
    ``measurement_index``); otherwise a :class:`ValueError` is raised with a
    hint listing available ids.

    Parameters
    ----------
    measurement
        Prefer a :class:`~elchem.dataset.Measurement`. Alternatively, an
        :class:`~elchem.dataset.ExperimentDataset` (see notes above).
    sample_id
        Required for multi-sample datasets when ``measurement`` is an
        :class:`~elchem.dataset.ExperimentDataset`; ignored when a
        :class:`~elchem.dataset.Measurement` is passed.
    measurement_index
        Which measurement file within the sample (default ``0``).
    x, y
        DataFrame column names.
    ax
        If ``None``, a new figure and axis are created using ``figsize``.
    label
        Line label for the legend; defaults to the plotted file's ``source_path.stem``.
    title
        If not ``None``, set as the axes title. The default is **no** title (publication style).
    marker, linestyle
        Passed through to the primary :meth:`pandas.DataFrame.plot`.
    grid
        If ``True``, enable a light default grid on the primary axis.
    invert_y
        If ``True``, call ``ax.invert_yaxis()`` after plotting the primary series.
    figsize
        Used only when ``ax`` is ``None``.
    save_path
        If set, save the figure once to this path (format from extension), with
        ``bbox_inches="tight"`` and ``dpi=300``. Independent of ``save`` / ``path_OUT``.
    xlabel, ylabel
        Axis labels; if ``None``, use the built-in map for ``x`` / ``y`` column names.
    y2
        Optional second y column plotted on ``ax.twinx()`` against the same ``x``.
    y2label
        Right axis label; if ``None``, use the built-in map for ``y2`` when possible.
    label2
        Legend label for the ``y2`` series; defaults to the ``y2`` column name.
    save
        If ``True``, write publication exports (``.png`` and ``.pdf``) under ``path_OUT``.
    path_OUT
        Directory for exports when ``save`` is ``True`` (required in that case).
    export_name
        Base filename without extension when ``save`` is ``True``. If omitted, built from
        ``sample_id``, ``technique``, and source stem when available.
    **plot_kwargs
        Extra keyword arguments for the primary :meth:`pandas.DataFrame.plot`
        (``x`` / ``y`` are filtered out).
    show_legend
        If ``True`` (default), draw a legend when labeled artists exist. If ``False``,
        suppress the legend for this call.
    legend_frameon
        If a legend is drawn, set whether its frame is visible (default ``False``).
    split_by
        How to break the trace into separate lines:

        * ``None`` — one line for the whole dataframe.
        * ``\"cycle_number\"`` — group by that column (CV cycles).
        * ``\"ox_red\"`` — split when ``ox_red`` changes row-to-row (contiguous runs).
        * ``\"jumps\"`` — split on large steps in *x* (stitched LSV segments).
        * ``\"auto\"`` — use ``cycle_number`` if that column has finite values, else *x* jumps.
    cycle
        * If splitting by **cycle_number** (including ``auto`` when it picks cycles):
          restrict to one or more values of the ``cycle_number`` column.
          Use ``None`` or ``\"all\"`` for every cycle.
        * If splitting by **jumps** / **ox_red** (or ``auto`` using jumps) and
          ``segment`` is omitted: **same as ``segment``** — convenient 0-based segment
          index or list of indices (e.g. ``cycle=1`` plots only segment ``1``).
          Use ``None`` or ``\"all\"`` for every segment.
    segment
        0-based segment index or list, for ``ox_red``, ``jumps``, or ``auto`` when
        using jump-style splitting. If both ``segment`` and ``cycle`` are passed in
        that mode, **segment** takes precedence (unless one is ``\"all\"``, which
        clears that selector). ``None`` or ``\"all\"`` includes every segment.
    technique
        ``None`` or ``\"all\"`` (case-insensitive) imposes no technique check; otherwise
        the measurement's ``technique`` must appear in this string or list.
    alpha
        Opacity for primary (and matching twin) lines.

    Returns
    -------
    fig, ax, ax2
        ``ax2`` is ``None`` unless ``y2`` was given.

    Examples
    --------
    Explicit measurement (preferred)::

        m = dataset.samples['105'].measurements[0]
        fig, ax, _ = plot_measurement(m, x='ewe_v', y='j_ma_cm2')

    Dataset shortcut (single sample only)::

        fig, ax, _ = plot_measurement(dataset, x='ewe_v', y='j_ma_cm2')

    Dataset with several samples (pass ``sample_id``)::

        fig, ax, _ = plot_measurement(
            dataset, sample_id='105', measurement_index=0,
            x='ewe_v', y='j_ma_cm2',
        )
    """
    m = _resolve_measurement(measurement, sample_id, measurement_index)
    df = m.dataframe
    _validate_xy_columns(df, x, y, y2)

    if not _is_plot_filter_all(technique):
        allowed_lst = _str_or_sequence(technique)
        if allowed_lst:
            allowed = set(allowed_lst)
            tval = m.technique
            if tval is None or tval not in allowed:
                raise ValueError(
                    f"plot_measurement: technique filter {technique!r} does not match "
                    f"this measurement (technique={tval!r})."
                )

    plot_kwargs = {
        k: v
        for k, v in plot_kwargs.items()
        if k
        not in {
            "x",
            "y",
            "legend",
            "split_by",
            "cycle",
            "segment",
            "alpha",
            "technique",
        }
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    apply_plot_style(ax)

    if label is None:
        label = _default_plot_label(m)

    ax2: Any | None = None

    if split_by is not None:
        segments = _resolve_plot_segments(
            df, x=x, split_by=split_by, cycle=cycle, segment=segment
        )
        if not segments:
            raise ValueError(
                f"plot_measurement: no data to plot after split_by={split_by!r}."
            )
        pk = dict(plot_kwargs)
        line_color = pk.pop("color", None)
        if line_color is None:
            (dn,) = ax.plot([float("nan")], [float("nan")])
            line_color = dn.get_color()
            dn.remove()
        line_kw: dict[str, Any] = {"linestyle": linestyle, "alpha": alpha, **pk}
        if marker is not None:
            line_kw["marker"] = marker
        for i, seg in enumerate(segments):
            seg_label = "_nolegend_"
            if show_legend and i == 0:
                seg_label = label
            ax.plot(seg[x], seg[y], color=line_color, label=seg_label, **line_kw)
        if show_legend:
            _style_legend(ax, frameon=legend_frameon)
        if y2 is not None:
            ax2 = ax.twinx()
            apply_plot_style(ax2)
            lab2_base = str(y2) if label2 is None else label2
            y2_kw: dict[str, Any] = {
                "linestyle": linestyle,
                "color": "C1",
                "alpha": alpha,
            }
            if marker is not None:
                y2_kw["marker"] = marker
            for i, seg in enumerate(segments):
                seg_l2 = "_nolegend_"
                if show_legend and i == 0:
                    seg_l2 = lab2_base
                ax2.plot(seg[x], seg[y2], label=seg_l2, **y2_kw)
            if show_legend:
                _style_legend(ax2, frameon=legend_frameon)
            ax2.set_ylabel(y2label if y2label is not None else _axis_label(y2))
    else:
        plot_call_kw: dict[str, Any] = {
            "ax": ax,
            "x": x,
            "y": y,
            "label": label,
            "linestyle": linestyle,
            "legend": show_legend,
            "alpha": alpha,
            **plot_kwargs,
        }
        if marker is not None:
            plot_call_kw["marker"] = marker

        df.plot(**plot_call_kw)
        if show_legend:
            _style_legend(ax, frameon=legend_frameon)

        if y2 is not None:
            ax2 = ax.twinx()
            apply_plot_style(ax2)
            if label2 is None:
                label2 = str(y2)
            plot2_kw: dict[str, Any] = {
                "ax": ax2,
                "x": x,
                "y": y2,
                "label": label2,
                "linestyle": linestyle,
                "color": "C1",
                "legend": show_legend,
                "alpha": alpha,
            }
            if marker is not None:
                plot2_kw["marker"] = marker
            df.plot(**plot2_kw)
            if show_legend:
                _style_legend(ax2, frameon=legend_frameon)
            ax2.set_ylabel(y2label if y2label is not None else _axis_label(y2))

    ax.set_xlabel(xlabel if xlabel is not None else _axis_label(x))
    ax.set_ylabel(ylabel if ylabel is not None else _axis_label(y))
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if invert_y:
        ax.invert_yaxis()

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    if save:
        if path_OUT is None:
            raise ValueError(
                "When save=True, path_OUT must be set to an output directory "
                "(e.g. path_OUT=Path('figures'))."
            )
        out_dir = Path(path_OUT).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        base = export_name if export_name else _default_export_basename(m)
        base = _sanitize_export_basename(base)
        _save_publication_pair(fig, out_dir, base)

    return fig, ax, ax2


def _default_dataset_export_basename(
    subset: ExperimentDataset,
    x: str,
    y: str,
    technique: str | Sequence[str] | None,
) -> str:
    keys = "_".join(sorted(subset.samples.keys())) if subset.samples else "empty"
    if technique is None or _is_plot_filter_all(technique):
        t = "alltech"
    elif isinstance(technique, (list, tuple, set)):
        t = "_".join(sorted(str(u) for u in technique))
    else:
        t = str(technique)
    return _sanitize_export_basename(f"dataset_{keys}_{x}_{y}_{t}")


def _dataset_curve_label(
    sample: SampleDataset,
    m: Measurement,
    *,
    n_curves_total: int,
    n_curves_for_sample: int,
) -> str:
    base = (sample.label and str(sample.label).strip()) or sample.sample_id
    if n_curves_total <= 1:
        return str(base)
    parts = [str(base)]
    if m.technique:
        parts.append(str(m.technique))
    if n_curves_for_sample > 1:
        parts.append(m.source_path.stem)
    return " | ".join(parts)


def plot_dataset(
    dataset: ExperimentDataset,
    x: str = "ewe_v",
    y: str = "j_ma_cm2",
    technique: str | Sequence[str] | None = None,
    to_plot: Literal["all"] | Sequence[str] | str | None = "all",
    sample_id: str | Sequence[str] | None = None,
    filename_contains: str | Sequence[str] | None = None,
    measurement_index: int | Sequence[int] | None = None,
    ax: Any | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool = True,
    frameon: bool = False,
    grid: bool = False,
    save: bool = False,
    path_OUT: str | Path | None = None,
    export_name: str | None = None,
    split_by: str | None = None,
    cycle: int | float | Sequence[Any] | str | None = None,
    segment: int | float | Sequence[Any] | str | None = None,
    alpha: float = 1.0,
    **plot_kwargs: Any,
) -> tuple["Figure", "Axes"]:
    """
    Plot the same ``x`` / ``y`` columns for many measurements across selected samples.

    Uses :func:`~elchem.dataset.select_samples` and draws each curve with
    :func:`plot_measurement` on a shared axes.

    Parameters
    ----------
    dataset
        :class:`~elchem.dataset.ExperimentDataset` to draw from.
    to_plot
        ``None``, ``\"all\"`` (case-insensitive), or sample ids passed to
        :func:`~elchem.dataset.select_samples`. A single id may be passed as a string.
    sample_id
        Further restrict to these sample id(s) (must be within the ``to_plot`` subset).
    technique
        ``None`` or ``\"all\"`` (case-insensitive) for all techniques; otherwise only
        measurements whose ``technique`` is in this string or list.
    filename_contains
        ``None`` or ``\"all\"`` for no filename filter; otherwise keep measurements
        whose ``source_path.name`` contains one of these substrings (case-sensitive).
    measurement_index
        If set, only measurement list indices matching this int or list (per sample).
    legend
        If ``True`` (default), show a legend aggregating all labeled curves after
        plotting. Set ``False`` to hide it. Legend frame uses ``frameon`` (default
        ``False``).
    split_by, cycle, segment, technique, alpha
        Passed to :func:`plot_measurement` for each curve (see there for ``split_by``
        modes: ``cycle_number``, ``ox_red``, ``jumps``, ``auto``). For jump-style
        splits, ``segment`` selects 0-based segments; ``cycle`` is accepted as an
        alias when ``segment`` is omitted. Use ``None`` or ``\"all\"`` for ``cycle`` /
        ``segment`` / ``technique`` to include every segment, cycle value, or technique.
    frameon
        Passed to :func:`plot_measurement` and used when rebuilding the combined legend.
    save, path_OUT, export_name
        When ``save=True``, write ``.png`` and ``.pdf`` under ``path_OUT`` (required).
        Default export basename uses sorted sample ids, ``x``, ``y``, and ``technique``.

    Returns
    -------
    fig, ax
    """
    if not isinstance(dataset, ExperimentDataset):
        raise TypeError(
            f"plot_dataset expected ExperimentDataset; got {type(dataset).__name__!r}."
        )

    subset = select_samples(dataset, to_plot)

    if sample_id is not None:
        wanted = set(_str_or_sequence(sample_id))
        unknown = wanted - set(subset.samples.keys())
        if unknown:
            raise KeyError(
                f"sample_id not in current selection: {sorted(unknown)!r}. "
                f"Available after to_plot: {list(subset.samples.keys())!r}."
            )
        subset = ExperimentDataset(
            samples={sid: subset.samples[sid] for sid in subset.samples if sid in wanted}
        )

    pm_plot_kwargs = dict(plot_kwargs)
    figsize = pm_plot_kwargs.pop("figsize", (6, 4))

    tech_list = _str_or_sequence(technique)
    tech_set = set(tech_list) if tech_list is not None else None
    name_substrings = _str_or_sequence(filename_contains)
    mi_list = _int_or_sequence(measurement_index)
    mi_set = set(mi_list) if mi_list is not None else None

    curves: list[tuple[SampleDataset, Measurement]] = []
    for sid in sorted(subset.samples.keys()):
        sample = subset.samples[sid]
        for i, m in enumerate(sample.measurements):
            if mi_set is not None and i not in mi_set:
                continue
            if tech_set is not None and (m.technique is None or m.technique not in tech_set):
                continue
            if name_substrings is not None and not any(
                sub in m.source_path.name for sub in name_substrings
            ):
                continue
            curves.append((sample, m))

    if not curves:
        raise ValueError(
            "plot_dataset: no measurements matched the combined filters "
            f"(to_plot={to_plot!r}, sample_id={sample_id!r}, technique={technique!r}, "
            f"measurement_index={measurement_index!r}, filename_contains={filename_contains!r})."
        )

    per_sample_counts: Counter[str] = Counter()
    for sample, _ in curves:
        per_sample_counts[sample.sample_id] += 1
    n_total = len(curves)

    pm_kw = {
        k: v
        for k, v in pm_plot_kwargs.items()
        if k
        not in {
            "ax",
            "x",
            "y",
            "label",
            "title",
            "grid",
            "save",
            "path_OUT",
            "export_name",
            "invert_y",
            "y2",
            "y2label",
            "label2",
            "save_path",
            "split_by",
            "cycle",
            "segment",
            "alpha",
            "sample_id",
            "filename_contains",
            "technique",
            "measurement_index",
            "to_plot",
        }
    }

    current_ax = ax
    fig: Any = None

    for idx, (sample, m) in enumerate(curves):
        lbl = _dataset_curve_label(
            sample,
            m,
            n_curves_total=n_total,
            n_curves_for_sample=per_sample_counts[sample.sample_id],
        )
        plot_ax = current_ax if (idx > 0 or ax is not None) else None
        fig, current_ax, _ = plot_measurement(
            m,
            x=x,
            y=y,
            ax=plot_ax,
            label=lbl,
            title=None,
            grid=grid,
            figsize=figsize,
            xlabel=xlabel,
            ylabel=ylabel,
            show_legend=legend,
            legend_frameon=frameon,
            split_by=split_by,
            cycle=cycle,
            segment=segment,
            technique=technique,
            alpha=alpha,
            **pm_kw,
        )

    if title is not None:
        current_ax.set_title(title)
    if legend:
        handles, lab_list = current_ax.get_legend_handles_labels()
        if handles:
            current_ax.legend(handles, lab_list, frameon=frameon)
            _style_legend(current_ax, frameon=frameon)
        else:
            leg = current_ax.get_legend()
            if leg is not None:
                leg.remove()
    else:
        leg = current_ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.tight_layout()

    if save:
        if path_OUT is None:
            raise ValueError(
                "When save=True, path_OUT must be set to an output directory "
                "(e.g. path_OUT=Path('figures'))."
            )
        out_dir = Path(path_OUT).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        base = (
            export_name
            if export_name
            else _default_dataset_export_basename(subset, x, y, technique)
        )
        _save_publication_pair(fig, out_dir, base)

    return fig, current_ax
