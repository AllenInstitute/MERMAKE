# Contributing

Thanks for helping with MERMAKE. The [developer guide](docs/developer.md) has the details; this is the short version.

## Setup

```bash
uv sync --extra cuda12 --group docs   # without a GPU: uv sync --group docs
```

## Before you open a pull request

```bash
uv run ruff check src tests scripts
uv run pytest tests/ -v
uv run --group docs mkdocs build --strict
```

- Tests marked `gpu` skip themselves without a GPU. If you change anything that touches the GPU code, run the full suite on a machine with one. CI only runs the CPU tests.
- Add a test for a bug you fix. Put it where it would have caught the bug.
- A change that alters results (coordinates, which maxima are found, scores) needs an entry under *Changed (affects results)* in [CHANGELOG.md](CHANGELOG.md). To check one, run the old and the new code on `tests/examples/single` and compare with `scripts/compare_xh.py`.
- A new setting goes in `mermake/config.py` with a description, see [Adding a setting](docs/developer.md#adding-a-setting).
- Public functions get NumPy-style docstrings (parameters, returns, array shapes and units, and whether arrays live on the GPU).
- New bugs you cannot fix right away go in [docs/known-issues.md](docs/known-issues.md).

## Style

Match the surrounding code (the package uses tabs). `ruff` runs a conservative rule set, it is not a formatter.

## License

MERMAKE is licensed under the GNU General Public License v3. By contributing you agree that your contributions are licensed the same way.
