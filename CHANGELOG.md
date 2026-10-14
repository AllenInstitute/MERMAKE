# Changelog

All notable changes to MERMAKE. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Spots are written as a Parquet dataset partitioned by set and field of view (`spots.parquet`), together with per-image statistics (`diagnostics.parquet`). The new `[output] format` setting selects `parquet`, `npz` or `both`.
- `mermake-report`, which plots the spot score distributions of each channel across rounds and heatmaps of spot counts and deconvolved/raw signal. Needs the new `report` extra.
- `mermake --suggest-threshold` samples a few images, counts the spots found at every threshold (and in the negated image as a noise estimate) and suggests values for `threshold`. See the configuration page.
- `pyarrow` is now a dependency.

### Changed

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
