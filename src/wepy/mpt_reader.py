"""BioLogic .mpt to pandas DataFrame with metadata and technique detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .columns import dataframe_biologic_columns
from .io import genfromtxt_mpt_robust
from .technique import detect_technique as resolve_technique


def _parse_header_line(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, val = line.split(":", 1)
    key, val = key.strip(), val.strip()
    if not key:
        return None
    return key, val


def parse_mpt_header_lines(header_lines: list[str]) -> dict[str, str]:
    """Parse 'Key : value' lines from the EC-Lab header block (excluding column row)."""
    meta: dict[str, str] = {}
    for line in header_lines[:-1]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _parse_header_line(line)
        if parsed:
            k, v = parsed
            meta[k] = v
    return meta


def read_mpt_header_block(path: Path, encoding: str = "cp855") -> tuple[int, list[str]]:
    """
    Read nb_header_lines and return the full header as a list of lines (no newline).

    EC-Lab: first line is magic, second line is 'Nb header lines : N',
    then N-2 more lines so the header has N lines total; line N is first data row.
    """
    with path.open("r", encoding=encoding, errors="replace") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"Empty file: {path}")
        second = f.readline()
        if not second:
            raise ValueError(f"Missing header line 2: {path}")
        m = re.search(r":\s*(\d+)\s*$", second.strip())
        if not m:
            raise ValueError(
                f"Could not parse 'Nb header lines' from line 2 in {path}: {second!r}"
            )
        nb = int(m.group(1))
        header_lines = [
            first.rstrip("\n\r"),
            second.rstrip("\n\r"),
        ]
        need = nb - len(header_lines)
        for _ in range(need):
            ln = f.readline()
            if not ln:
                raise ValueError(
                    f"Expected {nb} header lines in {path}, file ended early "
                    f"after {len(header_lines)} lines"
                )
            header_lines.append(ln.rstrip("\n\r"))
    return nb, header_lines


def mpt_numeric_block_to_dataframe(
    arr: Any, raw_column_names: list[str]
) -> pd.DataFrame:
    """Attach canonical column names; trim or pad names to match array width."""
    if arr.size == 0:
        return pd.DataFrame()
    if getattr(arr, "ndim", 0) != 2:
        raise ValueError("Expected 2-D numeric array for .mpt body")

    ncols = arr.shape[1]
    names_in = list(raw_column_names[:ncols])
    while len(names_in) < ncols:
        names_in.append(f"col_{len(names_in)}")
    col_names = dataframe_biologic_columns(names_in)
    return pd.DataFrame(arr, columns=col_names[:ncols])


def read_mpt_dataframe(
    path: str | Path, encoding: str = "cp855"
) -> tuple[pd.DataFrame, dict[str, str], str | None]:
    """
    Load a single BioLogic .mpt file into a DataFrame with canonical columns.

    Returns
    -------
    dataframe
        Numeric measurement table.
    header_meta
        Key/value metadata parsed from the header (excludes column title row).
    technique
        Canonical short code (e.g. ``CV``, ``LSV``) if detected, else ``None``.
    """
    path = Path(path)
    nb, header_lines = read_mpt_header_block(path, encoding=encoding)

    header_meta = parse_mpt_header_lines(header_lines)
    technique = resolve_technique(header_meta, header_lines, path)

    col_line = header_lines[-1] if header_lines else ""
    raw_names = [c.strip() for c in col_line.split("\t") if c.strip() != ""]

    # If the last header line does not look like column titles, fall back.
    if len(raw_names) < 2 and "\t" not in col_line:
        raw_names = []

    arr = genfromtxt_mpt_robust(str(path), skip_header=nb, encoding=encoding)
    df = mpt_numeric_block_to_dataframe(arr, raw_names)

    return df, header_meta, technique
