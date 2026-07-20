"""
Map BioLogic technique descriptions and filenames to canonical short codes (CV, LSV, PEIS, ...).
"""

from __future__ import annotations

import re
from pathlib import Path

# Longer / more specific phrases first (substring search on normalized text).
_FULL_PHRASE_TO_CODE: tuple[tuple[str, str], ...] = (
    ("potentio electrochemical impedance spectroscopy", "PEIS"),
    ("galvano electrochemical impedance spectroscopy", "GEIS"),
    ("linear sweep voltammetry", "LSV"),
    ("cyclic voltammetry", "CV"),
    ("chronoamperometry", "CA"),
    ("chronopotentiometry", "CP"),
    ("open circuit potential", "OCP"),
    ("open circuit voltage", "OCP"),
    ("electrochemical impedance spectroscopy", "EIS"),
)

_HEADER_TECHNIQUE_KEYS: tuple[str, ...] = (
    "Technique",
    "Technic",
    "AC technique",
    "Acq. mode",
)

# Filename / raw-token scan: longer codes first so PEIS wins over EIS in substring search.
_FILENAME_CODE_ORDER: tuple[str, ...] = (
    "PEIS",
    "GEIS",
    "EIS",
    "LSV",
    "CV",
    "CA",
    "CP",
    "OCP",
)


def _squash_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def canonize_technique_text(text: str) -> str | None:
    """
    Map a BioLogic-style technique string to a canonical short code, if recognized.

    Recognizes full phrases (case-insensitive) and bare short tokens.
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    t = _squash_spaces(raw)

    for phrase, code in _FULL_PHRASE_TO_CODE:
        if phrase in t:
            return code

    u = raw.upper().replace(" ", "")
    for code in _FILENAME_CODE_ORDER:
        if u == code:
            return code
    return None


def _technique_from_header_meta(header_meta: dict[str, str]) -> str | None:
    # 1) Known keys first (explicit technique fields).
    for key in _HEADER_TECHNIQUE_KEYS:
        if key in header_meta and header_meta[key]:
            c = canonize_technique_text(header_meta[key])
            if c:
                return c
    # 2) Any other metadata value (phrase match only).
    for v in header_meta.values():
        if not v:
            continue
        c = canonize_technique_text(v)
        if c:
            return c
    return None


def _technique_from_header_raw(header_lines: list[str]) -> str | None:
    """Search full header text (excluding the last line = column titles)."""
    if len(header_lines) < 2:
        return None
    body_lines = header_lines[:-1]
    blob = "\n".join(body_lines)
    c = canonize_technique_text(blob)
    if c:
        return c
    for line in body_lines:
        c = canonize_technique_text(line)
        if c:
            return c
    return None


def _tokenize_filename_stem(stem: str) -> list[str]:
    parts = re.split(r"[_\-\s\d]+", stem, flags=re.IGNORECASE)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p or p.isdigit():
            continue
        out.append(p.upper())
    return out


def _last_code_in_string(s: str) -> str | None:
    """Rightmost boundary-delimited technique code in s."""
    u = s.upper()
    best_code: str | None = None
    best_pos = -1
    for code in _FILENAME_CODE_ORDER:
        for m in re.finditer(
            rf"(?<![A-Za-z]){re.escape(code)}(?![A-Za-z])",
            u,
        ):
            if m.start() >= best_pos:
                best_pos = m.start()
                best_code = code
    return best_code


def detect_technique_from_filename(path: str | Path) -> str | None:
    """
    Infer technique from filename stem (fallback when header metadata is absent).

    Tokenizes the stem on underscores, hyphens, whitespace, and digit boundaries,
    then collects every recognized short code (CV, LSV, CA, CP, OCP, PEIS, GEIS, EIS).
    When multiple codes appear—as in messy lab names like ``01_CV_ramp_05_LSV_C01``—the
    **last** token wins, because the trailing code usually names the actual measurement.

    If tokenization finds no codes, scans the full stem for boundary-delimited codes
    and again prefers the rightmost match.
    """
    stem = Path(path).stem
    recognized: list[str] = []
    for tok in _tokenize_filename_stem(stem):
        if tok in _FILENAME_CODE_ORDER:
            recognized.append(tok)
    if recognized:
        return recognized[-1]
    return _last_code_in_string(stem)


def detect_technique(
    header_meta: dict[str, str],
    header_lines: list[str],
    path: str | Path | None,
) -> str | None:
    """
    Return canonical short technique code.

    Priority: (1) header metadata, (2) raw header lines, (3) filename stem.
    Filename inference uses the last recognized technique token when several appear;
    see ``detect_technique_from_filename``.
    """
    t = _technique_from_header_meta(header_meta)
    if t:
        return t
    t = _technique_from_header_raw(header_lines)
    if t:
        return t
    if path is not None:
        return detect_technique_from_filename(path)
    return None
