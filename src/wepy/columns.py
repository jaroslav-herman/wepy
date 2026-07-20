"""Map BioLogic EC-Lab ASCII column titles to stable pandas column names."""

from __future__ import annotations

import re
from typing import Iterable


def _squash_underscores(s: str) -> str:
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def canonicalize_biologic_column(raw: str) -> str:
    """
    Convert a BioLogic .mpt column title to a safe, unique-friendly snake name.

    Examples:
        "Ewe/V" -> "ewe_V"
        "time/s" -> "time_s"
        "<I>/mA" -> "I_mA"
        "-Im(Z)/Ohm" -> "minus_Im_Z_Ohm"
    """
    name = (raw or "").strip()
    if not name:
        return "empty"

    # Angle-bracket tags: <Ewe>/V -> Ewe/V
    name = re.sub(r"<([^>]+)>", r"\1", name)

    leading_minus = name.startswith("-")
    if leading_minus:
        name = name[1:]

    # Replace common separators
    name = name.replace("/", "_").replace(" ", "_")
    name = re.sub(r"[\(\)\[\]\{\}]", "_", name)
    name = re.sub(r"[%\u00b0]", "_", name)
    name = re.sub(r"[^\w\+]", "_", name)
    name = _squash_underscores(name)

    if leading_minus and name:
        name = "minus_" + name

    if not name:
        return "empty"

    # Prefer lowercase for stable keys; keep + for I+ etc.
    name = name.lower()
    return name or "empty"


def dedupe_column_names(names: Iterable[str]) -> list[str]:
    """Ensure all names are unique by appending _2, _3, ..."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            out.append(base)
            continue
        seen[base] += 1
        out.append(f"{base}_{seen[base]}")
    return out


def dataframe_biologic_columns(df_columns: Iterable[str]) -> list[str]:
    """Apply canonicalization + deduplication to a sequence of raw header strings."""
    canon = [canonicalize_biologic_column(c) for c in df_columns]
    return dedupe_column_names(canon)
