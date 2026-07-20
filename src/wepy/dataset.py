"""
Notebook-oriented loading: sample definitions to structured measurements (DataFrames).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union

import pandas as pd

from .mpt_reader import read_mpt_dataframe

SampleInput = Union[str, dict[str, Any]]


def _resolved_folder(path: str | Path) -> Path:
    """Coerce user-facing folder input to an absolute, resolved :class:`Path`."""
    return Path(path).expanduser().resolve()


@dataclass
class Measurement:
    """One loaded file (e.g. a single .mpt technique run).

    ``source_path`` is always a resolved :class:`pathlib.Path` (not ``str``).
    ``technique`` is a canonical short code (e.g. ``CV``, ``LSV``) when detected.
    """

    dataframe: pd.DataFrame
    source_path: Path
    technique: str | None
    metadata: dict[str, Any]


@dataclass
class SampleDataset:
    """All measurements for one sample ID.

    ``folder`` is always a resolved :class:`pathlib.Path`.
    """

    sample_id: str
    folder: Path
    label: str
    area_cm2: float | None
    extra: dict[str, Any] = field(default_factory=dict)
    measurements: list[Measurement] = field(default_factory=list)


@dataclass
class ExperimentDataset:
    """Full load result: sample_id to SampleDataset."""

    samples: dict[str, SampleDataset]

    def __getitem__(self, sample_id: str) -> SampleDataset:
        return self.samples[sample_id]

    def __iter__(self):
        return iter(self.samples.items())

    def keys(self):
        return self.samples.keys()


def ensure_output_dir(path_out: str | Path) -> Path:
    """Create ``path_out`` if needed; return resolved :class:`pathlib.Path`."""
    p = Path(path_out).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_sample_entry(sample_id: str, entry: SampleInput) -> SampleDataset:
    """
    Normalize user-facing sample value to a SampleDataset shell (no measurements yet).

    Accepts:
      - str / Path: folder path (``label`` defaults to ``sample_id``).
      - dict: must contain ``path``; optional ``label``, ``area_cm2``; other keys kept in ``extra``.
      If ``label`` is missing or blank, it defaults to ``sample_id``.
    """
    if isinstance(entry, (str, Path)):
        folder = _resolved_folder(entry)
        return SampleDataset(
            sample_id=sample_id,
            folder=folder,
            label=str(sample_id),
            area_cm2=None,
            extra={},
            measurements=[],
        )

    if not isinstance(entry, dict):
        raise TypeError(
            f"Sample {sample_id!r}: expected str, Path, or dict, got {type(entry).__name__}"
        )

    if "path" not in entry:
        raise KeyError(f"Sample {sample_id!r}: dict entries must include key 'path'")

    folder = _resolved_folder(entry["path"])
    raw_label = entry.get("label")
    if raw_label is None or (isinstance(raw_label, str) and not str(raw_label).strip()):
        label = str(sample_id)
    else:
        label = str(raw_label).strip()
    area = entry.get("area_cm2")
    if area is not None:
        area = float(area)

    reserved = {"path", "label", "area_cm2"}
    extra = {k: v for k, v in entry.items() if k not in reserved}

    return SampleDataset(
        sample_id=sample_id,
        folder=folder,
        label=label,
        area_cm2=area,
        extra=extra,
        measurements=[],
    )


def _load_one_mpt(
    sample_id: str,
    sample: SampleDataset,
    mpt_path: Path,
    encoding: str,
) -> Measurement:
    df, header_meta, technique = read_mpt_dataframe(mpt_path, encoding=encoding)
    src = mpt_path.resolve()
    meta: dict[str, Any] = {
        "biologic_header": header_meta,
        "sample": {
            "id": sample_id,
            "label": sample.label,
            "area_cm2": sample.area_cm2,
            "folder": sample.folder,
            **sample.extra,
        },
    }
    return Measurement(
        dataframe=df,
        source_path=src,
        technique=technique,
        metadata=meta,
    )


def load_samples(
    samples: Mapping[str, SampleInput],
    *,
    glob_pattern: str = "*.mpt",
    encoding: str = "cp855",
    on_error: Literal["skip", "raise"] = "skip",
) -> ExperimentDataset:
    """
    Load all matching measurements under each sample folder.

    Parameters
    ----------
    samples
        Map sample_id to folder path (``str`` or :class:`pathlib.Path`) or
        dict with at least ``path`` (``str`` or :class:`pathlib.Path`).
    glob_pattern
        Glob relative to each sample folder (default ``*.mpt``).
    encoding
        Text encoding for BioLogic ASCII exports (default cp855, EC-Lab Windows locale).
    on_error
        If ``'skip'``, unreadable files emit a warning and are omitted; if ``'raise'``, propagate.

    Returns
    -------
    ExperimentDataset
        Indexed by sample ID; each value holds a list of :class:`Measurement`.
    """
    out: dict[str, SampleDataset] = {}

    for sid, entry in samples.items():
        shell = normalize_sample_entry(sid, entry)
        if not shell.folder.is_dir():
            msg = f"Sample {sid!r}: not a directory: {shell.folder}"
            if on_error == "raise":
                raise FileNotFoundError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            out[sid] = shell
            continue

        paths = sorted(shell.folder.glob(glob_pattern))
        measurements: list[Measurement] = []
        for p in paths:
            if not p.is_file() or p.suffix.lower() != ".mpt":
                continue
            try:
                measurements.append(
                    _load_one_mpt(sid, shell, p, encoding=encoding)
                )
            except Exception as e:
                if on_error == "raise":
                    raise
                warnings.warn(
                    f"Sample {sid!r}: skipping {p}: {e}",
                    UserWarning,
                    stacklevel=2,
                )

        shell.measurements = measurements
        out[sid] = shell

    return ExperimentDataset(samples=out)


def _sample_selection_is_all(to_plot: Any) -> bool:
    """True if *to_plot* selects every sample (``None`` or ``\"all\"``, case-insensitive)."""
    if to_plot is None:
        return True
    if isinstance(to_plot, str) and to_plot.strip().lower() == "all":
        return True
    return False


def select_samples(
    dataset: ExperimentDataset,
    to_plot: Literal["all"] | Sequence[str] | str | None = "all",
) -> ExperimentDataset:
    """
    Return a new :class:`ExperimentDataset` containing only the selected sample IDs.

    ``to_plot`` may be ``None``, ``\"all\"`` (case-insensitive), or ``'all'`` to return
    all samples; a single sample id string; or a sequence of sample ids.
    """
    if _sample_selection_is_all(to_plot):
        return ExperimentDataset(samples=dict(dataset.samples))

    if isinstance(to_plot, str):
        wanted = [to_plot]
    else:
        wanted = list(to_plot)
    missing = [k for k in wanted if k not in dataset.samples]
    if missing:
        raise KeyError(f"Unknown sample id(s): {missing}")

    subset = {k: dataset.samples[k] for k in wanted}
    return ExperimentDataset(samples=subset)
