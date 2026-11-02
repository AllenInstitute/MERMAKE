# Known issues

Bugs and rough edges found during the package restructuring. Entries are kept until fixed.

| # | Area | Issue | Status |
|---|------|-------|--------|
| 2 | Test fixtures | `tests/examples/single/out/*.npz` do not match the output of the current code (nor of the code before the restructuring), so they are from an older version and cannot be used as a regression reference. The `deconv.300.*.npz` planes there were made before the z-shift fix (#10) and are one plane off in z; `test_full_deconv_full` compensates. | Open. Regenerate them or replace them with a small fixture. |
| 3 | Test data size | `tests/examples/` is about 287 MB tracked in git. | Open. Plan: a small fixture in git, the full data as a Code Ocean data asset. |
| 4 | Tests | `test_with_mocked_blur` in `tests/test_deconvolver.py` is never collected, so it does not run. | Open. |
| 6 | Linting | `ruff check` reported 26 findings under the configured rule set (E4, E7, E9, F), including unused variables and a bare `except`. | Fixed in `30f6cb9`. CI now enforces it. |
| 7 | Maxima kernels | Neighbor lookups in `local_maxima_count` and `local_maxima` read out of bounds when `delta` is larger than an image dimension, which crashed the test suite with `cudaErrorIllegalAddress`. Real images are far larger than `delta`, so normal runs were not affected. | Fixed in `5bd0737`. |
| 8 | Point helpers | `points.estimate_step_size` raised `NameError` because `KDTree` was never imported. Nothing calls it. | Fixed in the `io`/`utils` split. |
| 9 | Compare script | `scripts/compare_xh.py` used undefined `ref_data`/`temp_data` in its `Xh_minus` check. | Fixed in the `__main__` split. |
| 10 | Deconvolution | Reported positions were shifted by +1 voxel along any axis whose padded length is odd (`Deconvolver` rolled the padded PSF by `-N // 2`, which floors to `-ceil(N / 2)`). For z, `N = 3 * depth - 2`, so every odd z depth (for example 35 planes) was affected. The `habs` and `a` columns, which sample the raw image at the maximum, were therefore taken one plane off. | Fixed. Results from earlier versions have z one higher than now for odd depths. `tests/test_deconvolver.py::TestPositionPreserved` covers it. |
| 11 | Deconvolution | `Deconvolver.apply` leaves the last `2 * patch_overlap` columns (and rows) of its output uninitialized when the image is no wider than `patch_size`, because the first patch skips the right-edge reflection. Real images are larger than `patch_size` (for example 1200 against 500), so normal runs are not affected. | Open. Found while writing the test for #10. |
| 12 | Dead code | `points.find_two_means` is unused and fits 3 clusters although its name says two. | Open. Remove or fix. |
| 13 | Drift file | `drift_*.pkl` stores CuPy arrays, so it can only be loaded where CuPy and a GPU are available. | Open. Convert to NumPy (`.get()`) when saving. |
| 14 | Reader thread | `ImageQueue` raised `UnboundLocalError` in its reader thread when it had no image to read, so the thread ended without telling the pipeline and the run waited forever. It also marked the last image only after queueing it, and a block of a last field of view with one image never got its drift. | Fixed. The reader always sends the end marker, and the pipeline does the drift of the last block when the images are over. |
| 15 | Result files | `.npz` results and drift files were written straight to their final name. A run killed halfway (a batch job's time limit) left a truncated file that the next run took for a finished result. | Fixed. Files are written under a temporary name and renamed. Parquet results already were. |
| 16 | Saving | An error while saving a result (a full disk) was lost with the future of the save thread, and the run ended as a success without the result. | Fixed. The error is raised in the main thread at the next save or at the end. |
| 17 | GPU choice | `--gpu N` wrote `N` into `CUDA_VISIBLE_DEVICES` whatever the scheduler had set, so SLURM's choice was overridden and every task could end up on GPU 0. | Fixed. Without `--gpu` the variable is left alone, and `--gpu N` counts among the visible devices. |
