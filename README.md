Introduction
------------
MERMAKE processes MERFISH, smFISH, and IF imaging data by detecting local intensity maxima (puncta) in 3D image stacks. In multiplexed experiments (MERFISH), these puncta are decoded using a user-supplied codebook, while in smFISH mode and IF, the puncta are simply called and reported as-is.

## Installation

MERMAKE needs an **NVIDIA GPU** and CuPy. Pick the extra that matches your CUDA version (`cuda12` for CUDA 12.x, `cuda11` for CUDA 11.x). The extras install CuPy together with the CUDA headers it needs to compile kernels (`cupy-cuda12x[ctk]`), so a separate CUDA Toolkit install is not required. `nvcc` is not needed either, since CuPy compiles kernels with NVRTC.

With [uv](https://docs.astral.sh/uv/):
```
uv sync --extra cuda12
uv run mermake settings.toml
```

With pip:
```
python3 -m pip install "mermake[cuda12]"
```

The `cuda11` and `cuda12` extras are mutually exclusive. Add `viz` (`mermake[cuda12,viz]`) for the napari viewer used by `mermake --check`. If CuPy cannot find a GPU, `mermake` stops at startup with an error; see the [CuPy install guide](https://docs.cupy.dev/en/stable/install.html) for driver and CUDA troubleshooting.

## MERMAKE pipe line
the general steps of the MERMAKE merfish pipeline is:
Here is a simple flow chart:

```mermaid
graph TD;
    A["microscope"]--image capture-->B["zarr \(images\)"];
    B["zarr \(images\)"]--fitting-->C["xfits \(local maxima\)"];
    C["xfits \(local maxima\)"]--decoding-->D["decoded"];
    C["xfits \(local maxima\)"]--segmentation-->S["segm"];
    D["decoded"]---->E["final spots"];
    S["segm"]---->E["final spots"];
    E["final spots"]---->F["h5ad"];

```

##  MERMAKE Usage

To run **MERMAKE**, you'll need to provide a configuration TOML file with a few key settings describing your experiment. 
```
mermake my_settings.toml
```

If mermake is run without providing a toml file, it will warn about the usage and print out the toml file format. Most users only need to edit the `[paths]` section.

| TOML Section | Variable Name     | Description                                                                 |
|--------------|-------------------|-----------------------------------------------------------------------------|
| `[paths]`    | `codebook`        | Path to the CSV codebook for decoding barcodes (for MERFISH data only)     |
| `[paths]`    | `psf_file`        | Path to the PSF file used for deconvolution (e.g., `.npy` or `.pkl`)       |
| `[paths]`    | `flat_field_tag`  | Prefix path for flat field correction files (e.g., `"Scope3_"`)            |
| `[paths]`    | `hyb_range`       | Range of hybridization rounds to process (e.g., `'H1_*_set1:H1_*_set3'`) |
| `[paths]`    | `hyb_folders`     | List of folders containing raw imaging data                                |
| `[paths]`    | `output_folder`   | Path to the folder where MERMAKE should save results                        |


>  All other sections (`[hybs]`, `[dapi]`, etc.) are preconfigured for most use cases and usually do not need to be changed.  Though to use multi-psfs you will want to set the `tilesize` to the size of the samping grid (ie 300).

---

###  Example Config (`config.toml`)

```toml
[paths]
codebook = "codebook.csv"
psf_file = "psfs/psf_scope3.npy"
flat_field_tag = "flat_field/Scope3_"
hyb_range = "H1_*_set1:H16_*_set3"
hyb_folders = ["experiment_folder"]
output_folder = "output"
background_range = 'H0_background_set1:H0_background_set3'
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#           you probably dont have to change any of the settings below                  #
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#

hyb_save =  '{fov}--{tag}--col{icol}__Xhfits.npz'
dapi_save = '{fov}--{tag}--dapiFeatures.npz'
drift_save = 'drift_Conv_zscan__{ifov:0>3}--_set{iset}.pkl'
regex = '''([A-z]+)(\d+)_([^_]+)_set(\d+)(.*)''' #use triple quotes to avoid double escape

[hybs]
tile_size = 500
overlap = 89
beta = 0.0001
threshold = 3600
blur_radius = 30
delta = 1
delta_fit = 3
sigmaZ = 1
sigmaXY = 1.5

[dapi]
tile_size = 500
overlap = 89
beta = 0.01
threshold = 3.0
blur_radius = 50
delta = 5
delta_fit = 5
sigmaZ = 1
sigmaXY = 1.5
```

## 🔍 If you only want to process the dapi channels a trick that you can use is
To set the `hyb_save` toml string to not be dynamic and with different names for each fov and hyb.
```
hyb_save =  'Xhfits.npz'
dapi_save = '{fov}--{tag}--dapiFeatures.npz'
```
What this does is trick mermake into thinking the hybs for each fov and channel has already been processed due
to the existing Xhfits.npz file

## Development

```
uv sync --extra cuda12   # runtime plus the dev group (pytest, ruff)
uv run pytest tests/ -v  # tests marked "gpu" skip themselves without cupy or a CUDA device
uv run ruff check
uv build
```

Tests that need a GPU (the kernels, deconvolution, alignment and the end-to-end run in `tests/test_pipeline_e2e.py`) are marked `gpu`. Without a CUDA device they are skipped, so the filename, config and CLI tests run anywhere (`uv sync && uv run pytest`). GitHub Actions runs lint, the CPU tests on Python 3.10 and 3.13, and a wheel build; the full suite has to be run on a GPU machine.

Source layout (`src/mermake/`):

| Module | Contents |
|--------|----------|
| `cli.py`, `__main__.py` | argument parsing, settings loading, `mermake` entry point |
| `pipeline.py` | the processing loop: dapi, per-channel maxima, background, drift, save |
| `io/` | `readers` (zarr/flat fields), `naming` (filename parsing, `FolderFilter`), `queue` (`ImageQueue`, `Block`), `metadata` (XML) |
| `gpu/` | CuPy kernels and helpers: `blur`, `fill`, `elementwise`, `fft`, `runtime`, and the `.cu` sources in `kernels/` |
| `deconvolver.py`, `maxima.py`, `align.py` | deconvolution, local-maxima detection, drift and alignment |
| `config.py`, `points.py`, `viz.py` | config helpers, point clustering helpers, napari viewer |

## Known issues

Bugs and rough edges found during the package restructuring. Entries are kept until fixed.

| # | Area | Issue | Status |
|---|------|-------|--------|
| 1 | Background subtraction | `background_range` has no effect on the output. In `pipeline.process_image` the `aligner` is reset to `None` for every image, and background images end in an early return, so the `elif aligner:` branch (which computes the shifted slices and runs `subtract_reflect`) can never run. The pre-refactor code behaved the same. | Open. Fixing it changes output, so decide the intended behavior first. |
| 2 | Test fixtures | `tests/examples/single/out/*.npz` do not match the output of the current code (nor of the code before the restructuring), so they are from an older version and cannot be used as a regression reference. | Open. Regenerate them or replace them with a small fixture. |
| 3 | Test data size | `tests/examples/` is about 287 MB tracked in git. | Open. Plan: a small fixture in git, the full data as a Code Ocean data asset. |
| 4 | Tests | `test_with_mocked_blur` in `tests/test_deconvolver.py` is never collected, so it does not run. | Open. |
| 5 | Config docs | The example config above and `example.toml` disagree (`drift_save`, `regex`, and `background_range`, which is active here and commented out in `example.toml`). | Open. `example.toml` is to become the single source. |
| 6 | Linting | `ruff check` reported 26 findings under the configured rule set (E4, E7, E9, F), including unused variables and a bare `except`. | Fixed in `30f6cb9`. CI now enforces it. |
| 7 | Maxima kernels | Neighbor lookups in `local_maxima_count` and `local_maxima` read out of bounds when `delta` is larger than an image dimension, which crashed the test suite with `cudaErrorIllegalAddress`. Real images are far larger than `delta`, so normal runs were not affected. | Fixed in `5bd0737`. |
| 8 | Point helpers | `points.estimate_step_size` raised `NameError` because `KDTree` was never imported. Nothing calls it. | Fixed in the `io`/`utils` split. |
| 9 | Compare script | `scripts/compare_xh.py` used undefined `ref_data`/`temp_data` in its `Xh_minus` check. | Fixed in the `__main__` split. |
| 10 | Deconvolution | Reported positions are shifted by +1 voxel along any axis whose padded length is odd. `Deconvolver` rolls the padded PSF by `-N // 2`, which is one voxel too far when `N` is odd. For z, `N = 2 * zpad + depth`, so every odd z depth (for example 35 planes per channel) gives z coordinates that are +1. x and y are not affected with the default even `tile_size` and `overlap`. A constant shift does not hurt decoding, but it matters when comparing to other data. Found by `tests/test_pipeline_e2e.py`. | Open. Fixing it changes output coordinates. |
| 11 | Dead code | `config.count_colors` globs for `*<name>.zarr.xml` but the metadata file is `<name>.xml`, so it would fail with `IndexError`. `Config`, `count_bits`, `count_colors` and `count_hybs` are not called anywhere. `points.find_two_means` is also unused and fits 3 clusters although its name says two. | Open. Remove or fix. |
