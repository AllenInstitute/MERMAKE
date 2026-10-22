# Changelog

All notable changes to MERMAKE. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Spots are written as a Parquet dataset partitioned by set and field of view (`spots.parquet`), together with per-image statistics (`diagnostics.parquet`). The new `[output] format` setting selects `parquet`, `npz` or `both`.
- `mermake-report`, which plots the spot score distributions of each channel across rounds and heatmaps of spot counts and deconvolved/raw signal. Needs the new `report` extra.
- `mermake --suggest-threshold` samples a few images, counts the spots found at every threshold (and in the negated image as a noise estimate) and suggests values for `threshold`. See the configuration page.
- `[hybs] threshold` can be a list with one value per color channel (all channels but the last, which is dapi). `--suggest-threshold` prints the list. `[dapi] threshold` stays one number.
- `pyarrow` is now a dependency.
- Flat fields can be `.npy` 2D arrays as well as `.npz` files with the array `im`.
- Images are read from the `.zarr` entry itself when that is a zarr array, which is how current acquisition software writes them. The older `<fov>/data` layout is still read.
- `[paths] sum_channels = [i, j]` sums two channels into one when an image is read, for acquisitions where one color was imaged twice to double its exposure. 8-bit data is squared before the sum, and the sum is clipped at 65535.
- `[paths] dapi_channel` gives the index of the dapi channel (default `-1`, the last one) for acquisitions where dapi is stored first. The remaining channels are the hyb colors in stored order, and the flat fields follow the image channels.
- `mermake --shard I/N` processes one share of the fields of view, with all the rounds of a field of view in the same share, and `mermake --gpus all` (or `--gpus 0,1,2`) runs one worker per GPU of the machine on its own share. See [several GPUs and nodes](docs/scaling.md).
- `--prefetch N` sets how many images are read ahead (default 5), the main setting for the RAM use.
- `scripts/run_slurm_parallel.sbatch` (a job array, one shard and GPU per task, or several GPUs per task) and `scripts/run_slurm_report.sbatch`.
- `scripts/bench_pipeline.py`, `scripts/bench_patch.py` and `scripts/bench_read.py` to find where the time and the GPU memory go and how fast the input can be read.
- `[hybs] real_fft` and `[dapi] real_fft` (default `true`) select real-to-complex FFTs in the deconvolution. They take half the time and half the memory of the PSF spectrum and the FFT buffer and give the same result up to rounding. `false` uses complex FFTs as before.

### Changed (affects results)

- **`patch_overlap` now defaults to 110 in `[hybs]` and `[dapi]`. It was 89.** The FFT of the deconvolution is several times slower for a padded patch size with a large prime factor: `patch_size + 2 * patch_overlap` was 678 = 2 x 3 x 113 and is now 720 = 2^4 x 3^2 x 5. A run takes about half the time (87 s instead of 185 s for the nine rounds of a field of view on a T4, with `real_fft`). The spots change a little: on a field of view of 3.2 million spots, 0.4 % of them (those within about 1 % of the threshold) were found in only one of the two versions, and `h` of the others changed by 1 % or less for 99 % of them. **To reproduce the results of 0.1.0, set `patch_overlap = 89` and `real_fft = false` in `[hybs]` and `[dapi]`**: that gave exactly the same spots and diagnostics as before this change. If you change `patch_size`, choose `patch_overlap` so that their combination is again a product of 2, 3, 5 and 7.
- **`real_fft` is on by default.** 18 of 3.2 million spots changed against complex FFTs, all at the threshold.

### Changed

- **Faster, with identical results:** 26.4 s to 18.8 s per image on a T4 (40 planes of 2800 x 2800, 4 channels), and 8.1 s with the new defaults above. `box_1d` no longer runs an exact overlap test of its buffers on every call (about 20 ms of host time with the GPU idle), `reflect` copies the slices it changes instead of visiting the whole stack, and `find_local_maxima` sizes its outputs by the number of maxima, reads a uint16 image directly and no longer flushes the GPU memory pool after every call. The peak GPU memory fell from 9.2 to 7.0 GiB. See [where the time goes](docs/scaling.md).
- `--gpu` no longer defaults to 0 and no longer overwrites `CUDA_VISIBLE_DEVICES` when it is not given, so the GPU a scheduler assigned is used. `--gpu N` counts among the visible devices.
- Images are processed in sorted (set, field of view) order, instead of the order of the directory listing.
- `.npz` and drift files are written under a temporary name and renamed when complete, so a killed run leaves no truncated result. Errors while saving now stop the run instead of being lost.
- A run with no images to process (for example an empty shard) exits instead of waiting forever, and the drift of the last field of view is done when the images are over.
- **The default output format is now `parquet`.** Set `[output] format = "npz"` (or `"both"`) to get the `.npz` files. A run resumes from whichever selected format it finds.

## [0.1.0] - 2026-10-14

The first release of the Allen Institute fork. It has breaking changes to the settings files and, for stacks with an odd number of planes, to the results; see below.

### Changed (affects results)

- **Fixed a +1 voxel shift in z** for stacks with an odd number of planes. `Deconvolver` rolled the padded PSF by `-N // 2`, which is one voxel too far when the padded length `N` is odd. Reported z coordinates are now one lower than in earlier versions for such data, and the `habs` and `a` columns, which sample the raw image at the maximum, now use the right plane. x and y were not affected with the default settings.
- `delta_fit` is now limited to 1-5. Larger values silently truncated the neighborhood in the GPU kernel and biased the positions, and 0 gave NaN scores.

### Changed (settings files)

- **Settings are validated and mostly optional.** Only `psf_file`, `flat_field_tag`, `hyb_folders`, `hyb_range` and `output_folder` are required, everything else has a default. Misspelled or unknown keys, wrong types, bad ranges and missing files are reported together before the GPU is used.
- Relative paths are resolved against the folder of the settings file instead of the current directory.
- `tile_size` is now `patch_size` and `overlap` is now `patch_overlap`, since they describe the patches the image is cut into for deconvolution, not the raw image tiles. The old names give an error.
- Removed the unused `codebook` key and the non-functional `background_range` key.
- The default `drift_save` is `drift_{fov}--_set{iset}.pkl`.

### Added

- `mermake --print-config` (every setting with its description and value), `--print-example` and `--dry-run`, none of which needs a GPU.
- Documentation site (mkdocs), with a configuration reference generated from the code, and NumPy-style docstrings for the public API.
- `[project.optional-dependencies]` extras `cuda12`, `cuda11` and `viz`, a `uv.lock`, and `uv` based development commands.
- Tests that run without a GPU, a `gpu` pytest marker that skips the others automatically, a synthetic end-to-end test, and a GitHub Actions workflow.

### Removed

- Background subtraction. It never ran (the aligner was reset for every image), so removing it does not change any result.
- Dead code: `gui.py`, `scripts/worker.py`, `scripts/test.py`, the `other/` directory and the `mermake.py` symlink.

### Internal

- `src/` layout with `cli`, `config`, `pipeline`, `viz`, `io/` and `gpu/` modules instead of one large `__main__`, `io` and `utils`. `math.py` is now `gpu/elementwise.py`.
- Importing `mermake` no longer checks for `nvcc` and no longer needs a GPU. CuPy compiles kernels with NVRTC, so `nvcc` is not required at all.
- Fixed out-of-bounds reads in the maxima kernels when `delta` exceeds an image dimension.

## [0.0.74]

The version of [deprekate/mermake](https://github.com/deprekate/mermake) this fork started from. See the git history.

[0.1.0]: https://github.com/AllenInstitute/MERMAKE/compare/9ef1138...v0.1.0
[0.0.74]: https://github.com/deprekate/mermake
