"""
RDE / Tafel helpers aligned with ``old_code/elchem_handy_V4_3.py``.

These routines reproduce the **numeric behaviour** of the legacy
``tafel_transform`` and ``calculate_tafel_slopes`` helpers (current density,
overpotential, log-current, linear regression, and the overpotential-at-fixed-
current-density heuristic). No new physical models are introduced.

Legacy reference (same repository)
------------------------------------
- ``tafel_transform`` (approx. lines 2187–2264): ``eta = V - E_rev``,
  ``j = I_mA / area_cm2``, ``log10(abs(j))`` with NaN where ``j <= 0``; CV branch
  subsamples every 10 points and keeps the rising-voltage sweep segment.
- ``calculate_tafel_slopes`` (approx. lines 2267–2400): ``linregress(eta, log10 j)``
  on a ``Tafel_ranges`` window; ``Tafel_slope_mV_dec = 1000 / slope``;
  ``10**intercept`` is the exchange current density at ``eta = 0`` on the fitted
  line. In this package it is exposed as :attr:`~TafelRegressionLegacy.j0_ma_cm2`
  (same unit as *j* in the fitted ``y`` column, typically mA cm^-2).

No separate onset-potential routine was found in the searched legacy module;
only the current-density crossing used for the overpotential-at-threshold metric.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .dataset import ExperimentDataset, Measurement, SampleDataset, select_samples
from .plotting import (
    _int_or_sequence,
    _is_plot_filter_all,
    _resolve_plot_segments,
    _save_publication_pair,
    _sanitize_export_basename,
    _str_or_sequence,
    _style_legend,
)
from .plots import apply_plot_style

ProcedureLSVorCV = Literal["LSV", "CV"]


def current_density_mA_cm2(current_ma: np.ndarray, area_cm2: float) -> np.ndarray:
    """
    Current density ``j = I / area`` with *I* in mA and *area* in cm^2.

    Matches the legacy ``tafel_transform`` convention and
    :func:`~elchem.processing.add_current_density` (output in mA cm^-2).
    """
    if area_cm2 <= 0:
        raise ValueError("area_cm2 must be positive.")
    return np.asarray(current_ma, dtype=float) / float(area_cm2)


def tafel_overpotential_V(voltage_V: np.ndarray, E_rev_V: float) -> np.ndarray:
    """Overpotential ``η = V - E_rev`` (same as legacy ``tafel_transform``)."""
    return np.asarray(voltage_V, dtype=float) - float(E_rev_V)


def tafel_log10_abs_current_density_ma_cm2(
    current_ma: np.ndarray, area_cm2: float
) -> np.ndarray:
    """
    ``log10(abs(j))`` with ``j = I_mA / area_cm2``; NaN where ``j <= 0`` (legacy errstate).
    """
    j = current_density_mA_cm2(current_ma, area_cm2)
    out = np.full_like(j, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = j > 0
        out[pos] = np.log10(np.abs(j[pos]))
    return out


def current_ma_at_voltage_linear_interp(
    voltage_V: np.ndarray, current_ma: np.ndarray, target_V: float
) -> float:
    """
    Linear interpolation of *current* (mA) at *target_V* (V), legacy-style 1D ``interp``.

    Assumes *voltage_V* is already monotone enough for a unique bracket; uses
    ``numpy.interp`` like typical notebook ``np.interp(target, V, I)``.
    """
    v = np.asarray(voltage_V, dtype=float)
    i = np.asarray(current_ma, dtype=float)
    if v.size < 2:
        raise ValueError("Need at least two points for interpolation.")
    return float(np.interp(float(target_V), v, i))


def _linregress_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return ``(slope, intercept, r_squared)`` matching ``scipy.stats.linregress``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        raise ValueError("Need at least two finite points for regression.")
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    denom = np.sum(dx**2)
    if denom == 0:
        raise ValueError("Degenerate x range for regression.")
    slope = float(np.sum(dx * (y - ym)) / denom)
    intercept = float(ym - slope * xm)
    r = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else 0.0
    return slope, intercept, r**2


def cv_rising_voltage_mask_tafel_legacy(
    transformed_curve: np.ndarray,
    *,
    voltage_col: int = 7,
    subsample: int = 10,
) -> np.ndarray:
    """
    Boolean mask selecting rows kept by the legacy CV ``tafel_transform`` branch.

    Ported from ``elchem_handy_V4_3.tafel_transform`` (subsample step + rising *η*).
    """
    curve = np.asarray(transformed_curve, dtype=float)
    n = curve.shape[0]
    subsample_indices = np.arange(0, n, subsample, dtype=int)
    subsampled = curve[subsample_indices]
    voltage_diff = np.diff(
        subsampled[:, voltage_col], prepend=subsampled[0, voltage_col]
    )
    rising_mask = voltage_diff > 0
    rising_indices = subsample_indices[rising_mask]
    if rising_indices.size == 0:
        return np.zeros(n, dtype=bool)
    filtered_idx = np.hstack(
        [np.arange(int(s), min(int(s) + subsample, n), dtype=int) for s in rising_indices]
    )
    mask = np.zeros(n, dtype=bool)
    mask[filtered_idx] = True
    return mask


def build_tafel_columns_lsv_cv(
    voltage_V: np.ndarray,
    current_ma: np.ndarray,
    area_cm2: float,
    E_rev_V: float,
    procedure: ProcedureLSVorCV,
    *,
    cycle_number: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a 2D array in the **legacy column layout** after ``tafel_transform``:

    - **LSV** ``(N, 9)``: columns 6 and 7 hold ``eta`` and ``log10(j)``; column 8 is
      the cycle number (default 1, or ``cycle_number`` when given). Legacy ``tafel_transform`` sets ``cycle_col = -1`` (last
      column); it must **not** share index 7 with ``current_col``, or the cycle
      write would overwrite ``log10(j)``. ``correct_R`` / RDE LSV arrays use at
      least nine columns (cycle after current).
    - **CV** ``(N, 10)``: columns 7 and 8 hold ``eta`` and ``log10(j)``; column 9 is cycle.

    Only columns used by regression / threshold logic are filled consistently
    with legacy indices (voltage 6 or 7, current 7 or 8, cycle last or 9).
    """
    v = np.asarray(voltage_V, dtype=float).ravel()
    i_ma = np.asarray(current_ma, dtype=float).ravel()
    if v.size != i_ma.size:
        raise ValueError("voltage_V and current_ma must have the same length.")
    n = v.size
    eta = tafel_overpotential_V(v, E_rev_V)
    logj = tafel_log10_abs_current_density_ma_cm2(i_ma, area_cm2)
    if procedure == "LSV":
        if cycle_number is None:
            cyc = np.ones(n, dtype=float)
        else:
            cyc = np.asarray(cycle_number, dtype=float).ravel()
            if cyc.size != n:
                raise ValueError("cycle_number length must match voltage.")
        out = np.zeros((n, 9), dtype=float)
        out[:, 6] = eta
        out[:, 7] = logj
        out[:, 8] = cyc
        return out
    if cycle_number is None:
        cyc = np.ones(n, dtype=float)
    else:
        cyc = np.asarray(cycle_number, dtype=float).ravel()
        if cyc.size != n:
            raise ValueError("cycle_number length must match voltage.")
    out = np.zeros((n, 10), dtype=float)
    out[:, 7] = eta
    out[:, 8] = logj
    out[:, 9] = cyc
    return out


def apply_cv_tafel_rising_filter_legacy(
    transformed_curve: np.ndarray, procedure: ProcedureLSVorCV
) -> np.ndarray:
    """Apply legacy CV rising-branch filter; LSV arrays are returned unchanged."""
    curve = np.asarray(transformed_curve, dtype=float)
    if procedure != "CV":
        return curve
    m = cv_rising_voltage_mask_tafel_legacy(curve, voltage_col=7)
    return curve[m]


@dataclass(frozen=True)
class TafelRegressionLegacy:
    """
    Numeric outputs mirroring one legacy ``calculate_tafel_slopes`` cycle dict.

    Regression fits ``log10(j)`` vs overpotential ``eta`` (V). Values are
    unchanged from legacy arithmetic; only public names document units.

    Attributes
    ----------
    intercept
        Intercept of ``log10(j)`` vs ``eta`` (dimensionless log10 axis).
    j0_ma_cm2
        ``10**intercept``: current density at ``eta = 0`` on the fitted line,
        in the **same unit as *j*** used for the fitted ``y`` column. When ``y``
        comes from :func:`tafel_log10_abs_current_density_ma_cm2`, that is
        **mA cm^-2**. This is **not** converted to A cm^-2 unless you divide by
        1000 yourself.
    """

    cycle: float
    slope_mV_dec: float | None
    intercept: float | None
    r_squared: float | None
    j0_ma_cm2: float | None
    overpotential_at_j_threshold_mV: float | None


def tafel_regress_and_threshold_legacy(
    tafel_curve: np.ndarray,
    corrected_voltage_V: np.ndarray,
    corrected_current_ma: np.ndarray,
    *,
    area_cm2: float,
    E_rev_V: float,
    tafel_range_eta_V: tuple[float, float],
    procedure: ProcedureLSVorCV,
    j_threshold_mA_cm2: float = 10.0,
    cycle_num: float = 1.0,
) -> TafelRegressionLegacy:
    """
    One-cycle equivalent of ``calculate_tafel_slopes`` / ``tafel_transform`` pipeline.

    *tafel_curve* must already contain **η** in the voltage column and **log10 j**
    in the current column (indices 6/7 LSV or 7/8 CV). *Corrected* arrays are the
    same segment in **physical** ``V`` and ``I_mA`` for the threshold metric.

    The returned :attr:`~TafelRegressionLegacy.j0_ma_cm2` is ``10**intercept`` from
    the linear fit; it uses the same current-density unit as *j* in that ``log10(j)``
    column (mA cm^-2 when built via :func:`build_tafel_columns_lsv_cv` and legacy
    ``tafel_transform``). Numeric values match legacy ``exchange_current_density``.
    """
    tc = np.asarray(tafel_curve, dtype=float)
    v_raw = np.asarray(corrected_voltage_V, dtype=float).ravel()
    i_raw = np.asarray(corrected_current_ma, dtype=float).ravel()
    if procedure == "LSV":
        voltage_col, current_col = 6, 7
    else:
        voltage_col, current_col = 7, 8

    eta = tc[:, voltage_col]
    logj = tc[:, current_col]
    lo, hi = tafel_range_eta_V
    m = (eta >= lo) & (eta <= hi)
    eta_f = eta[m]
    logj_f = logj[m]
    valid = np.isfinite(eta_f) & np.isfinite(logj_f)
    eta_f, logj_f = eta_f[valid], logj_f[valid]
    if eta_f.size < 2:
        # Legacy ``calculate_tafel_slopes``: if fewer than two regression points,
        # all outputs including ``overpotential_*`` are ``None`` (no threshold pass).
        return TafelRegressionLegacy(cycle_num, None, None, None, None, None)

    slope, intercept, r2 = _linregress_xy(eta_f, logj_f)
    try:
        slope_mV_dec = 1000.0 / slope
        j0_ma_cm2 = 10.0**intercept
    except ZeroDivisionError:
        slope_mV_dec = None
        j0_ma_cm2 = None

    eta_thr = _overpotential_mv_at_j_threshold_legacy(
        v_raw, i_raw, area_cm2, E_rev_V, j_threshold_mA_cm2
    )
    return TafelRegressionLegacy(
        cycle_num, slope_mV_dec, intercept, r2, j0_ma_cm2, eta_thr
    )


def save_tafel_regression_csv(
    samples: Sequence[tuple[str, Sequence[TafelRegressionLegacy]]],
    output_file: str | Path,
    *,
    j_threshold_mA_cm2: float = 10.0,
) -> None:
    """
    Write Tafel regression rows to CSV with unit-consistent column headers.

    Numeric cells match :class:`TafelRegressionLegacy` (same values legacy
    ``save_tafel_results_to_csv`` would write for slope, intercept, R², and
    ``10**intercept``). The exchange-current column is labeled ``j0_ma_cm2``
    (mA cm^-2 when ``y`` is ``log10(j)`` with *j* in mA cm^-2), not A cm^-2.
    """
    path = Path(output_file)
    eta_col = f"overpotential_at_{j_threshold_mA_cm2:g}_mA_cm2_mV"
    headers = [
        "sample",
        "cycle",
        "slope_mV_dec",
        "intercept_log10_j",
        "r_squared",
        "j0_ma_cm2",
        eta_col,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for sample_label, results in samples:
            for res in results:
                writer.writerow(
                    [
                        sample_label,
                        _csv_cell(res.cycle),
                        _csv_cell(res.slope_mV_dec),
                        _csv_cell(res.intercept),
                        _csv_cell(res.r_squared),
                        _csv_cell(res.j0_ma_cm2),
                        _csv_cell(res.overpotential_at_j_threshold_mV),
                    ]
                )


def _csv_cell(value: float | None) -> str | float:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def _normalize_tafel_range_eta_V(
    raw: Any, *, sample_id: str, source: str
) -> tuple[float, float]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise ValueError(
            f"Sample {sample_id!r}: {source} must be a (eta_min, eta_max) pair; got {raw!r}."
        )
    eta_lo, eta_hi = float(raw[0]), float(raw[1])
    if eta_lo > eta_hi:
        raise ValueError(
            f"Sample {sample_id!r}: {source} requires eta_min <= eta_max; got {raw!r}."
        )
    return eta_lo, eta_hi


def _tafel_range_eta_V_for_sample(
    sample: SampleDataset,
    *,
    default: tuple[float, float],
    measurement: Measurement | None = None,
) -> tuple[float, float]:
    """
    Resolve the Tafel fit window for one sample.

    Priority: ``sample.extra['tafel_range_eta_V']``, then
    ``measurement.metadata['sample']['tafel_range_eta_V']`` (set by
    :func:`~elchem.dataset.load_samples`), else *default*.
    """
    raw = sample.extra.get("tafel_range_eta_V")
    if raw is None and measurement is not None:
        sample_meta = (measurement.metadata or {}).get("sample") or {}
        raw = sample_meta.get("tafel_range_eta_V")
    if raw is None:
        return float(default[0]), float(default[1])
    return _normalize_tafel_range_eta_V(
        raw, sample_id=sample.sample_id, source="tafel_range_eta_V"
    )


def _tafel_range_from_df_row(row: pd.Series) -> tuple[float, float] | None:
    eta_min = row.get("tafel_range_eta_min_V")
    eta_max = row.get("tafel_range_eta_max_V")
    if pd.isna(eta_min) or pd.isna(eta_max):
        return None
    return float(eta_min), float(eta_max)


def _require_sample_area_cm2(sample: SampleDataset) -> float:
    if sample.area_cm2 is None:
        raise ValueError(
            f"Sample {sample.sample_id!r} has no area_cm2. Set area_cm2 in the samples "
            f"dictionary when calling load_samples, e.g. "
            f"{sample.sample_id!r}: {{'path': <folder>, 'area_cm2': 0.196}}."
        )
    area = float(sample.area_cm2)
    if area <= 0:
        raise ValueError(
            f"Sample {sample.sample_id!r}: area_cm2 must be positive; got {area!r}."
        )
    return area


def _segment_or_cycle_label(seg_df: pd.DataFrame, segment_index: int) -> Any:
    if "cycle_number" in seg_df.columns and seg_df["cycle_number"].notna().any():
        return seg_df["cycle_number"].iloc[0]
    return segment_index


def _procedure_for_technique(technique: str | None) -> ProcedureLSVorCV:
    if technique == "LSV":
        return "LSV"
    if technique == "CV":
        return "CV"
    raise ValueError(
        f"Tafel analysis supports techniques 'LSV' and 'CV' only; got {technique!r}."
    )


_TAFEL_ETA_AXIS = "Overpotential \u03b7 (V)"
_TAFEL_LOGJ_AXIS = r"log10(|j|) (j in mA cm$^{-2}$)"


@dataclass(frozen=True)
class TafelSegmentResult:
    """One analyzed segment: Tafel axes and legacy regression outputs."""

    sample: SampleDataset
    measurement: Measurement
    segment_or_cycle: Any
    eta_V: np.ndarray
    log10_j: np.ndarray
    regression: TafelRegressionLegacy
    procedure: ProcedureLSVorCV
    tafel_range_eta_V: tuple[float, float]


def _tafel_filter_curves(
    dataset: ExperimentDataset,
    *,
    to_plot: Literal["all"] | Sequence[str] | str | None,
    sample_id: str | Sequence[str] | None,
    technique: str | Sequence[str] | None,
    filename_contains: str | Sequence[str] | None,
    measurement_index: int | Sequence[int] | str | None,
    context: str,
) -> list[tuple[SampleDataset, Measurement]]:
    if not isinstance(dataset, ExperimentDataset):
        raise TypeError(
            f"{context} expected ExperimentDataset; got {type(dataset).__name__!r}."
        )

    subset = select_samples(dataset, to_plot)
    if sample_id is not None:
        wanted = set(_str_or_sequence(sample_id) or [])
        unknown = wanted - set(subset.samples.keys())
        if unknown:
            raise KeyError(
                f"sample_id not in current selection: {sorted(unknown)!r}. "
                f"Available after to_plot: {list(subset.samples.keys())!r}."
            )
        subset = ExperimentDataset(
            samples={sid: subset.samples[sid] for sid in subset.samples if sid in wanted}
        )

    tech_list = _str_or_sequence(technique)
    tech_set = set(tech_list) if tech_list is not None else None
    name_substrings = _str_or_sequence(filename_contains)
    mi_list = (
        None
        if _is_plot_filter_all(measurement_index)
        else _int_or_sequence(measurement_index)  # type: ignore[arg-type]
    )
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
            f"{context}: no measurements matched the combined filters "
            f"(to_plot={to_plot!r}, sample_id={sample_id!r}, technique={technique!r}, "
            f"measurement_index={measurement_index!r}, filename_contains={filename_contains!r})."
        )
    return curves


def _eta_logj_from_tafel_columns(
    cols: np.ndarray, procedure: ProcedureLSVorCV
) -> tuple[np.ndarray, np.ndarray]:
    if procedure == "LSV":
        return cols[:, 6].copy(), cols[:, 7].copy()
    return cols[:, 7].copy(), cols[:, 8].copy()


def _iter_tafel_segments(
    dataset: ExperimentDataset,
    *,
    to_plot: Literal["all"] | Sequence[str] | str | None = "all",
    sample_id: str | Sequence[str] | None = None,
    technique: str | Sequence[str] | None = "LSV",
    filename_contains: str | Sequence[str] | None = "all",
    measurement_index: int | Sequence[int] | str | None = "all",
    split_by: str | None = "auto",
    cycle: int | float | Sequence[Any] | str | None = "all",
    segment: int | float | Sequence[Any] | str | None = "all",
    voltage_column: str = "ewe_v",
    current_column: str = "i_ma",
    E_rev_V: float = 1.23,
    tafel_range_eta_V: tuple[float, float] = (0.05, 0.25),
    j_threshold_mA_cm2: float = 10.0,
    context: str = "tafel_analysis",
) -> list[TafelSegmentResult]:
    curves = _tafel_filter_curves(
        dataset,
        to_plot=to_plot,
        sample_id=sample_id,
        technique=technique,
        filename_contains=filename_contains,
        measurement_index=measurement_index,
        context=context,
    )
    results: list[TafelSegmentResult] = []

    for sample, m in curves:
        eta_lo, eta_hi = _tafel_range_eta_V_for_sample(
            sample, default=tafel_range_eta_V, measurement=m
        )
        area_cm2 = _require_sample_area_cm2(sample)
        proc = _procedure_for_technique(m.technique)
        df = m.dataframe
        if voltage_column not in df.columns:
            raise KeyError(
                f"Column {voltage_column!r} not found in {m.source_path.name}. "
                f"Available: {list(df.columns)!r}."
            )
        if current_column not in df.columns:
            raise KeyError(
                f"Column {current_column!r} not found in {m.source_path.name}. "
                f"Available: {list(df.columns)!r}."
            )

        if split_by is None:
            segments = [(df, 0)]
        else:
            seg_dfs = _resolve_plot_segments(
                df,
                x=voltage_column,
                split_by=split_by,
                cycle=cycle,
                segment=segment,
            )
            segments = [(seg, i) for i, seg in enumerate(seg_dfs)]

        for seg_df, seg_idx in segments:
            v = seg_df[voltage_column].to_numpy(dtype=float)
            i_raw = seg_df[current_column].to_numpy(dtype=float)
            if current_column == "i_a":
                i_ma = i_raw * 1000.0
            else:
                i_ma = i_raw

            cyc_arr = None
            if "cycle_number" in seg_df.columns:
                cyc_arr = seg_df["cycle_number"].to_numpy(dtype=float)

            cols = build_tafel_columns_lsv_cv(
                v, i_ma, area_cm2, E_rev_V, proc, cycle_number=cyc_arr
            )
            if proc == "CV":
                mask = cv_rising_voltage_mask_tafel_legacy(cols, voltage_col=7)
                cols = cols[mask]
                v = v[mask]
                i_ma = i_ma[mask]

            seg_label = _segment_or_cycle_label(seg_df, seg_idx)
            try:
                cycle_num = float(seg_label)
            except (TypeError, ValueError):
                cycle_num = 1.0

            res = tafel_regress_and_threshold_legacy(
                cols,
                v,
                i_ma,
                area_cm2=area_cm2,
                E_rev_V=E_rev_V,
                tafel_range_eta_V=(eta_lo, eta_hi),
                procedure=proc,
                j_threshold_mA_cm2=j_threshold_mA_cm2,
                cycle_num=cycle_num,
            )
            eta, logj = _eta_logj_from_tafel_columns(cols, proc)
            results.append(
                TafelSegmentResult(
                    sample=sample,
                    measurement=m,
                    segment_or_cycle=seg_label,
                    eta_V=eta,
                    log10_j=logj,
                    regression=res,
                    procedure=proc,
                    tafel_range_eta_V=(eta_lo, eta_hi),
                )
            )
    return results


def _tafel_row_from_segment(
    seg: TafelSegmentResult,
    *,
    E_rev_V: float,
    j_threshold_mA_cm2: float,
) -> dict[str, Any]:
    eta_lo, eta_hi = seg.tafel_range_eta_V
    res = seg.regression
    m = seg.measurement
    return {
        "sample_id": seg.sample.sample_id,
        "sample_label": seg.sample.label,
        "technique": m.technique,
        "source_file": m.source_path.name,
        "segment_or_cycle": seg.segment_or_cycle,
        "E_rev_V": float(E_rev_V),
        "tafel_range_eta_min_V": eta_lo,
        "tafel_range_eta_max_V": eta_hi,
        "j_threshold_mA_cm2": float(j_threshold_mA_cm2),
        "slope_mV_dec": res.slope_mV_dec,
        "intercept": res.intercept,
        "r_squared": res.r_squared,
        "j0_ma_cm2": res.j0_ma_cm2,
        "overpotential_at_j_threshold_mV": res.overpotential_at_j_threshold_mV,
    }


def _tafel_plot_label(seg: TafelSegmentResult, *, n_segments: int) -> str:
    base = (seg.sample.label and str(seg.sample.label).strip()) or seg.sample.sample_id
    if n_segments <= 1:
        return str(base)
    parts = [str(base)]
    if seg.measurement.technique:
        parts.append(str(seg.measurement.technique))
    parts.append(seg.measurement.source_path.stem)
    if seg.segment_or_cycle not in (0, 1, None):
        parts.append(f"seg={seg.segment_or_cycle}")
    return " | ".join(parts)


def _log10j_slope_from_regression(res: TafelRegressionLegacy) -> float | None:
    if res.slope_mV_dec is None or res.slope_mV_dec == 0:
        return None
    return 1000.0 / float(res.slope_mV_dec)


def _regression_line_from_tafel_row(row: pd.Series) -> tuple[float | None, float | None]:
    intercept = row.get("intercept")
    slope_mV_dec = row.get("slope_mV_dec")
    if pd.isna(intercept) or pd.isna(slope_mV_dec) or slope_mV_dec == 0:
        return None, None
    slope = 1000.0 / float(slope_mV_dec)
    return slope, float(intercept)


def tafel_analysis(
    dataset: ExperimentDataset,
    *,
    to_plot: Literal["all"] | Sequence[str] | str | None = "all",
    sample_id: str | Sequence[str] | None = None,
    technique: str | Sequence[str] | None = "LSV",
    filename_contains: str | Sequence[str] | None = "all",
    measurement_index: int | Sequence[int] | str | None = "all",
    split_by: str | None = "auto",
    cycle: int | float | Sequence[Any] | str | None = "all",
    segment: int | float | Sequence[Any] | str | None = "all",
    voltage_column: str = "ewe_v",
    current_column: str = "i_ma",
    E_rev_V: float = 1.23,
    tafel_range_eta_V: tuple[float, float] = (0.05, 0.25),
    j_threshold_mA_cm2: float = 10.0,
    save: bool = False,
    path_OUT: str | Path | None = None,
    export_name: str = "tafel_results",
) -> pd.DataFrame:
    """
    Run legacy-compatible Tafel regression on filtered measurements (notebook API).

    Uses the same sample / measurement / filename / technique filters and the same
    segment / cycle splitting as :func:`~elchem.plotting.plot_dataset`, then calls
    :func:`build_tafel_columns_lsv_cv` and :func:`tafel_regress_and_threshold_legacy`
    without changing regression math.

    Parameters
    ----------
    dataset
        Loaded experiment data.
    to_plot, sample_id, technique, filename_contains, measurement_index
        Same meaning as in :func:`~elchem.plotting.plot_dataset`.
    split_by, cycle, segment
        Passed to the plotting segment resolver (default ``split_by='auto'``).
    voltage_column, current_column
        Columns for potential (V) and current (mA). ``i_a`` is converted to mA.
    E_rev_V, tafel_range_eta_V, j_threshold_mA_cm2
        Legacy Tafel parameters (overpotential window and j crossing threshold).
        Per-sample ``tafel_range_eta_V`` in the samples dict (or
        ``metadata['sample']['tafel_range_eta_V']`` after :func:`~elchem.dataset.load_samples`)
        overrides the global *tafel_range_eta_V* for that sample. Used ranges are stored
        as ``tafel_range_eta_min_V`` / ``tafel_range_eta_max_V`` in the output.
    save, path_OUT, export_name
        When ``save=True``, write ``{path_OUT}/{export_name}.csv``.

    Returns
    -------
    pandas.DataFrame
        One row per analyzed segment. ``j0_ma_cm2`` is ``10**intercept`` in the same
        unit as *j* in the fitted ``log10(j)`` column (mA cm^-2 with default inputs).
    """
    segments = _iter_tafel_segments(
        dataset,
        to_plot=to_plot,
        sample_id=sample_id,
        technique=technique,
        filename_contains=filename_contains,
        measurement_index=measurement_index,
        split_by=split_by,
        cycle=cycle,
        segment=segment,
        voltage_column=voltage_column,
        current_column=current_column,
        E_rev_V=E_rev_V,
        tafel_range_eta_V=tafel_range_eta_V,
        j_threshold_mA_cm2=j_threshold_mA_cm2,
        context="tafel_analysis",
    )
    rows = [
        _tafel_row_from_segment(
            seg, E_rev_V=E_rev_V, j_threshold_mA_cm2=j_threshold_mA_cm2
        )
        for seg in segments
    ]
    out = pd.DataFrame(rows)

    if save:
        if path_OUT is None:
            raise ValueError(
                "When save=True, path_OUT must be set to an output directory "
                "(e.g. path_OUT=Path('output_examples'))."
            )
        out_dir = Path(path_OUT).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{export_name}.csv"
        out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return out


def plot_tafel_analysis(
    dataset: ExperimentDataset,
    tafel_df: pd.DataFrame | None = None,
    *,
    to_plot: Literal["all"] | Sequence[str] | str | None = "all",
    sample_id: str | Sequence[str] | None = None,
    technique: str | Sequence[str] | None = "LSV",
    filename_contains: str | Sequence[str] | None = "all",
    measurement_index: int | Sequence[int] | str | None = "all",
    split_by: str | None = "auto",
    cycle: int | float | Sequence[Any] | str | None = "all",
    segment: int | float | Sequence[Any] | str | None = "all",
    voltage_column: str = "ewe_v",
    current_column: str = "i_ma",
    E_rev_V: float = 1.23,
    tafel_range_eta_V: tuple[float, float] = (0.05, 0.25),
    j_threshold_mA_cm2: float = 10.0,
    show_fit: bool = True,
    show_fit_region: bool = True,
    fit_line_color: str = "black",
    fit_linewidth: float = 2.5,
    fit_linestyle: str = "--",
    fit_marker_size: float = 3,
    fit_marker_alpha: float = 0.7,
    trace_alpha: float = 0.35,
    trace_linewidth: float = 1.0,
    save: bool = False,
    path_OUT: str | Path | None = None,
    export_name: str = "tafel_plot",
    figsize: tuple[float, float] = (6, 4),
    ax: Any | None = None,
    legend_frameon: bool = False,
) -> tuple[Any, Any]:
    """
    Tafel plot (η vs log10|j|) with optional fit window and regression line.

    Uses the same sample / measurement / filename / technique filters and segment
    splitting as :func:`tafel_analysis`, then plots η = E − E_rev and
    log10(|j|) with j in mA cm⁻². The linear overlay reuses the legacy regression
    from :func:`tafel_regress_and_threshold_legacy` (or matching rows in *tafel_df*
    when supplied).

    Parameters
    ----------
    dataset
        Loaded experiment data.
    tafel_df
        Optional dataframe from :func:`tafel_analysis`; when a row matches a
        segment (sample_id, source_file, segment_or_cycle), its intercept,
        slope, and stored ``tafel_range_eta_min_V`` / ``tafel_range_eta_max_V``
        are preferred for the fit line and highlighted window.
    show_fit, show_fit_region
        Overlay the regression line and/or highlight points inside the Tafel
        fit window (per-sample or from *tafel_df* when matched).
    fit_line_color, fit_linewidth, fit_linestyle
        Style of the Tafel regression line (defaults emphasize the fit over the trace).
    fit_marker_size, fit_marker_alpha
        Markers for points inside the fit window.
    trace_alpha, trace_linewidth
        Style of the full η vs log10|j| trace.
    save, path_OUT, export_name
        When ``save=True``, write ``.png`` and ``.pdf`` under *path_OUT*.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    segments = _iter_tafel_segments(
        dataset,
        to_plot=to_plot,
        sample_id=sample_id,
        technique=technique,
        filename_contains=filename_contains,
        measurement_index=measurement_index,
        split_by=split_by,
        cycle=cycle,
        segment=segment,
        voltage_column=voltage_column,
        current_column=current_column,
        E_rev_V=E_rev_V,
        tafel_range_eta_V=tafel_range_eta_V,
        j_threshold_mA_cm2=j_threshold_mA_cm2,
        context="plot_tafel_analysis",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    apply_plot_style(ax)

    n_seg = len(segments)
    for idx, seg in enumerate(segments):
        eta = seg.eta_V
        logj = seg.log10_j
        label = _tafel_plot_label(seg, n_segments=n_seg)

        (line,) = ax.plot([], [])
        color = line.get_color()
        line.remove()

        matched_row = None
        if tafel_df is not None and not tafel_df.empty:
            key_mask = (
                (tafel_df["sample_id"] == seg.sample.sample_id)
                & (tafel_df["source_file"] == seg.measurement.source_path.name)
            )
            if "segment_or_cycle" in tafel_df.columns:
                key_mask &= tafel_df["segment_or_cycle"] == seg.segment_or_cycle
            hits = tafel_df.loc[key_mask]
            if len(hits) == 1:
                matched_row = hits.iloc[0]

        range_from_df = (
            _tafel_range_from_df_row(matched_row) if matched_row is not None else None
        )
        if range_from_df is not None:
            eta_lo, eta_hi = range_from_df
        else:
            eta_lo, eta_hi = seg.tafel_range_eta_V

        finite = np.isfinite(eta) & np.isfinite(logj)
        ax.plot(
            eta[finite],
            logj[finite],
            "-",
            linewidth=trace_linewidth,
            alpha=trace_alpha,
            color=color,
            label=label,
        )

        if show_fit_region:
            in_range = finite & (eta >= eta_lo) & (eta <= eta_hi)
            if np.any(in_range):
                region_label = "fit window" if idx == 0 else "_nolegend_"
                ax.plot(
                    eta[in_range],
                    logj[in_range],
                    "o",
                    ms=fit_marker_size,
                    alpha=fit_marker_alpha,
                    color=color,
                    label=region_label,
                )

        slope: float | None
        intercept: float | None
        if matched_row is not None:
            slope, intercept = _regression_line_from_tafel_row(matched_row)
        else:
            res = seg.regression
            intercept = res.intercept
            slope = _log10j_slope_from_regression(res)

        if show_fit and slope is not None and intercept is not None:
            eta_line = np.linspace(eta_lo, eta_hi, 100)
            logj_line = slope * eta_line + intercept
            fit_label = "Tafel fit" if idx == 0 else "_nolegend_"
            ax.plot(
                eta_line,
                logj_line,
                fit_linestyle,
                linewidth=fit_linewidth,
                color=fit_line_color,
                label=fit_label,
            )

    ax.set_xlabel(_TAFEL_ETA_AXIS)
    ax.set_ylabel(_TAFEL_LOGJ_AXIS)
    handles, lab_list = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lab_list, frameon=legend_frameon)
        _style_legend(ax, frameon=legend_frameon)
    fig.tight_layout()

    if save:
        if path_OUT is None:
            raise ValueError(
                "When save=True, path_OUT must be set to an output directory "
                "(e.g. path_OUT=Path('figures'))."
            )
        out_dir = Path(path_OUT).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        base = _sanitize_export_basename(export_name)
        _save_publication_pair(fig, out_dir, base)

    return fig, ax


def _overpotential_mv_at_j_threshold_legacy(
    voltage_V: np.ndarray,
    current_ma: np.ndarray,
    area_cm2: float,
    E_rev_V: float,
    j_threshold_mA_cm2: float,
) -> float | None:
    """Legacy ``overpotential_{limit}_mA_cm2`` (mean of up to first 5 crossings)."""
    j = current_density_mA_cm2(current_ma, area_cm2)
    v = np.asarray(voltage_V, dtype=float)
    target_mask = j >= float(j_threshold_mA_cm2)
    if not np.any(target_mask):
        return float("nan")
    matching_indices = np.where(target_mask)[0]
    if matching_indices.size == 0:
        return float("nan")
    num_values = min(5, matching_indices.size)
    selected_indices = matching_indices[:num_values]
    return float(1000.0 * np.mean(v[selected_indices] - float(E_rev_V)))
