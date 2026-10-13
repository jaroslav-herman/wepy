eisyfit-wepy
============

EIS-specific electrochemical data-processing utilities used by EIS Fitting.

The distribution is named `eisyfit-wepy` to avoid confusion with the
unrelated public `wepy` project. The installed Python import package remains
`wepy`.

## Installation

The package is published on PyPI:

```shell
pip install eisyfit-wepy
```

For a reproducible `uv` project, pin the released version in `pyproject.toml`:

```toml
[project]
dependencies = [
    "eisyfit-wepy>=0.1.6",
]
```

To install directly from a git tag instead, use:

```shell
pip install "eisyfit-wepy @ git+https://github.com/jaroslav-herman/wepy.git@v0.1.6"
```

## What's new

### 0.1.6

- Added the `py.typed` marker so type checkers use the inline annotations.
- Published installation instructions now point to PyPI instead of git tags.
- Fixed the `wepy.loopeis` module import (`import .basics` -> `import wepy.basics`).

### 0.1.5

- Fixed the `wepy.loopeis` module import so the module ships usable.

### 0.1.4

- Pinned `eisyfit-galvani>=0.5.1` for `.mpr` loop metadata support.

### 0.1.3

- Added `read_mpr`, which reads Bio-Logic `.mpr` files and adds a zero-based
  `loop` column derived from the Galvani loop metadata.
- `read_file` now uses the MPR-specific reader for `.mpr` files so loop
  metadata is retained.
- Added `read_mpt_dataframe` for the EIS Fitting text-import compatibility API.

## License

This EIS-specific fork is distributed under the GNU General Public License
version 3 or later. Its dependencies remain under their own licenses; retain
their notices when redistributing an installation.
