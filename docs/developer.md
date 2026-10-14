# Developer guide

## Setup

```bash
git clone https://github.com/AllenInstitute/MERMAKE
cd MERMAKE
uv sync --extra cuda12 --group docs   # runtime, CuPy, dev tools and the docs tools
```

The `dev` group (pytest, ruff) is installed by default. Without a GPU, leave out the extra: `uv sync`.

## Layout

```
src/mermake/
  __init__.py       version only, no side effects
  __main__.py       python -m mermake
  cli.py            argument parsing, --print-config, --dry-run
  config.py         settings dataclasses, defaults, validation (no cupy)
  pipeline.py       the processing loop
  deconvolver.py    patch-wise FFT deconvolution
  maxima.py         local maxima and sub-voxel refinement
  align.py          drift between rounds
  points.py viz.py  point helpers, napari viewer
  example.toml      the example settings file, printed by the CLI and shown in the docs
  io/               readers, naming, queue, metadata
  gpu/              blur, fill, elementwise, fft, runtime, kernels/*.cu
tests/  docs/  scripts/
```

`cli.py` and `config.py` must not import cupy: the GPU is selected with `CUDA_VISIBLE_DEVICES`, which has to be set before cupy is first imported. A test enforces this for `cli`.

## Tests

```bash
uv run pytest tests/ -v       # everything that can run on this machine
uv run ruff check src tests scripts
```

Tests that need a GPU carry `@pytest.mark.gpu` (usually through a module-level `pytestmark`) and call `pytest.importorskip("cupy")`. `tests/conftest.py` skips them when no CUDA device is found, so the naming, config, XML, points and CLI tests run anywhere. `tests/test_pipeline_e2e.py` runs the `mermake` command on a small synthetic field of view with planted spots.

GitHub Actions runs lint, the CPU tests on Python 3.10 and 3.13, a wheel build and the docs build. The full suite has to be run on a GPU machine.

To compare against a previous version, run both on `tests/examples/single` and compare the output with `scripts/compare_xh.py`.

## Adding a setting

Add a field to `Paths` or `Params` in `config.py` with a `doc` (and a `unit` for numbers). Give `Params` fields defaults in `DEFAULTS` for both `hybs` and `dapi`. A test fails if a field has no description. The configuration reference, `--print-config` and the validation pick it up automatically.

## Adding a kernel

1. Put the CUDA source in `src/mermake/gpu/kernels/name.cu`. It is packaged by the `package-data` glob in `pyproject.toml`.
2. Load it with `kernel_source("name.cu")` from `mermake.gpu.runtime` and `cupy.RawKernel`, as `maxima.py` does.
3. Add a GPU test. Kernels are compiled by NVRTC the first time they are called, so nothing needs `nvcc`.
4. Add the file to the wheel check in `.github/workflows/ci.yml`.

## Documentation

```bash
uv run --group docs mkdocs serve          # live preview
uv run --group docs mkdocs build --strict # what CI runs
```

The API reference is generated from NumPy-style docstrings, and the configuration reference from `mermake.config`. Neither needs a GPU: the documentation is built without CuPy.

## Releasing

1. Update `CHANGELOG.md` and the number in `VERSION`.
2. Run the full test suite on a GPU machine and check the docs build.
3. `uv build`, and check that the wheel holds `gpu/kernels/*.cu` and `example.toml`.
4. Tag the commit with the version.
