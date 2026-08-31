eisyfit-wepy
============

EIS-specific electrochemical data-processing utilities used by EIS Fitting.

The distribution is named `eisyfit-wepy` to avoid confusion with the
unrelated public `wepy` project. The installed Python import package remains
`wepy`.

## Installation

```shell
pip install "eisyfit-wepy @ git+https://github.com/jaroslav-herman/wepy.git@v0.1.3"
```

For a reproducible `uv` project, pin the same release in `pyproject.toml`:

```toml
[project]
dependencies = ["eisyfit-wepy"]

[tool.uv.sources]
eisyfit-wepy = { git = "https://github.com/jaroslav-herman/wepy.git", tag = "v0.1.3" }
```

## License

This EIS-specific fork is distributed under the GNU General Public License
version 3 or later. Its dependencies remain under their own licenses; retain
their notices when redistributing an installation.
