# Configuration

MERMAKE reads a [TOML](https://toml.io) settings file:

```bash
mermake settings.toml
```

Only `[paths]` is required, and only five of its keys have no default. Everything else has a default in the code, so a settings file lists just what differs. A complete minimal file:

```toml
--8<-- "src/mermake/example.toml"
```

## Rules

- **Relative paths** are resolved against the folder that contains the settings file, not the current directory.
- **Overrides**: in `[hybs]` and `[dapi]` list only the keys you want to change. The other keys keep their defaults.
- **Validation** happens before the GPU is used. Unknown or misspelled keys (with a suggestion), values of the wrong type or range, a `hyb_range` that does not match `regex`, and missing files are all reported together.
- **Renamed or removed keys** give an error that says what to do: `tile_size` is now `patch_size`, `overlap` is now `patch_overlap`, and `codebook` and `background_range` no longer exist.
- The settings that were used are saved in every result file, as the `args` array.

## Looking at the settings

| Command | What it does |
|---|---|
| `mermake --print-example` | print the minimal example above |
| `mermake --print-config settings.toml` | print every setting with its description and the value that will be used, defaults filled in |
| `mermake --dry-run settings.toml` | validate, and list the images that would be processed |
| `mermake --suggest-threshold settings.toml` | sample a few images and suggest values for `threshold` (needs a GPU, see below) |
| `mermake --dry-run --shard 1/4 settings.toml` | the same for one share of the fields of view (see [several GPUs and nodes](scaling.md)) |

### Choosing `threshold`

`mermake --suggest-threshold settings.toml` deconvolves a few random images (`--sample-images`, default 3) like the pipeline does and finds the maxima once with a very low threshold. A maximum is kept at threshold `T` exactly when its `h` is above `T`, so one run gives the number of spots at every threshold. The same is done on the negated image, where all maxima are noise. The count of these relative to the bright maxima estimates the false positive fraction at each threshold. It prints a table per channel, saves `threshold_suggestion.png` in the output folder (needs matplotlib), and suggests the lowest threshold whose estimated false positive fraction is at most `--max-false-positives` (default 0.3; larger values are greedier and keep more spots).

`[hybs] threshold` is either one number for all color channels or a list with one number per color channel (all channels but the last, which is dapi), for example `threshold = [18810, 9051, 8180]`. When every channel has a suggestion, `--suggest-threshold` prints that list ready to paste. `[dapi] threshold` is always one number. The estimate is rough because deconvolution ringing is not symmetric in sign. The lowest threshold tried for the color channels is 4 robust standard deviations above the median, so a suggestion marked as the lowest tried may be too high.

None of them needs a GPU.

## Running

| Option | What it does |
|---|---|
| `-g N`, `--gpu N` | use GPU `N`, counted among the visible ones. Without it the environment is left alone and the first visible GPU is used, so a scheduler's `CUDA_VISIBLE_DEVICES` is respected |
| `--gpus all` or `--gpus 0,1,2` | run one worker per GPU on this machine, each on its own share of the fields of view |
| `--shard I/N` | process share `I` (from 0) of `N` of the fields of view, every round of a field of view together |
| `--prefetch N` | images read ahead of the GPU (default 5). Each is held in memory whole, so this sets most of the RAM use |

See [several GPUs and nodes](scaling.md).

## Reference

This section is generated from the code (`mermake.config`).

<!-- config-reference -->

## Tuning notes

- `patch_size` is **not** the size of your images. It is the edge of the square patches each z-stack is cut into for deconvolution, and it sets the GPU memory use.
- With a pickled dict of several PSFs (sampled on a grid over the field of view), set `patch_size` to the spacing of that grid so every patch gets the PSF measured near it.
- `patch_overlap` should be larger than the radius of the PSF. Raising it costs memory and time.
- **`patch_size + 2 * patch_overlap` should be a product of 2, 3, 5 and 7.** The FFT of the deconvolution is several times slower for a size with a larger prime factor. The default, 500 + 2 x 110 = 720, is fine. It was 89 up to version 0.1.0 (678 = 2 x 3 x 113, which is slow). Going back to 89 moves about 0.4 % of the spots, those at the threshold. To reproduce the results of version 0.1.0, set `patch_overlap = 89` and `real_fft = false` in `[hybs]` and `[dapi]`. If you change `patch_size`, change `patch_overlap` with it. See [where the time goes](scaling.md).
- `real_fft` (default `true`) uses real-to-complex FFTs in the deconvolution, which take half the time and memory and give the same result up to rounding. Set it to `false` for complex FFTs.
- `threshold` is the setting to adjust when you find too many or too few maxima.
