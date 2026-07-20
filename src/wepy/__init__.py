"""EIS and DRT package for water electrolyzers measurements"""

__version__ = "0.0.1"

from .basics import *

# from .eis import *
from .iv_curve import *

"""Electrochemical processing package: I/O, datasets, legacy helpers."""

from .columns import canonicalize_biologic_column, dataframe_biologic_columns
from .dataset import (
    ExperimentDataset,
    Measurement,
    SampleDataset,
    ensure_output_dir,
    load_samples,
    normalize_sample_entry,
    select_samples,
)
from .legacy import *
from .mpt_reader import read_mpt_dataframe
from .processing import (
    add_current_density,
    split_dataframe_by_column,
    split_dataframe_by_contiguous_runs,
    split_dataframe_by_jumps,
)
from . import rde
from .plotting import plot_dataset, plot_measurement
from .rde import plot_tafel_analysis, tafel_analysis
from .technique import (
    canonize_technique_text,
    detect_technique,
    detect_technique_from_filename,
)

_LEGACY_IO = (
    "genfromtxt_mpt_robust",
    "loadFile",
    "loadFiles",
    "loadFilesCSV",
    "loadFilesDATXPS",
    "loadFilesGEN",
    "loadFilesMPT",
    "loadFilesTXT",
    "saveDat",
)

__all__ = [
    "ExperimentDataset",
    "Measurement",
    "SampleDataset",
    "add_current_density",
    "canonicalize_biologic_column",
    "canonicalize_technique_text",
    "dataframe_biologic_columns",
    "detect_technique",
    "detect_technique_from_filename",
    "ensure_output_dir",
    "load_samples",
    "normalize_sample_entry",
    "plot_dataset",
    "plot_measurement",
    "plot_tafel_analysis",
    "rde",
    "read_mpt_dataframe",
    "select_samples",
    "tafel_analysis",
    "split_dataframe_by_column",
    "split_dataframe_by_contiguous_runs",
    "split_dataframe_by_jumps",
] + list(_LEGACY_IO)


# modules = [
#     "basics",
#     "loopeis",
#     "eis"
# ]

# for module in modules:
#     try:
#         __import__(f"wepy.{module}", fromlist=[''])
#         print(f"Imported {module}")
#     except ImportError as e:
#         print(f"Failed to import {module}: {e}")


# __all__ = [
#     "basics",
#     "loopeis",
#     "eis",
# ]
