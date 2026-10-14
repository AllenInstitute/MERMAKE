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

### Choosing `threshold`

`mermake --suggest-threshold settings.toml` deconvolves a few random images (`--sample-images`, default 3) like the pipeline does and finds the maxima once with a very low threshold. A maximum is kept at threshold `T` exactly when its `h` is above `T`, so one run gives the number of spots at every threshold. The same is done on the negated image, where all maxima are noise. The count of these relative to the bright maxima estimates the false positive fraction at each threshold. It prints a table per channel, saves `threshold_suggestion.png` in the output folder (needs matplotlib), and suggests the lowest threshold whose estimated false positive fraction is at most `--max-false-positives` (default 0.3; larger values are greedier and keep more spots).

`[hybs]` has one threshold for all color channels, so choose from the per channel suggestions. The estimate is rough because deconvolution ringing is not symmetric in sign. The lowest threshold tried for the color channels is 4 robust standard deviations above the median, so a suggestion marked as the lowest tried may be too high.

None of them needs a GPU.

## Reference

This section is generated from the code (`mermake.config`).

<!-- config-reference -->

## Tuning notes

- `patch_size` is **not** the size of your images. It is the edge of the square patches each z-stack is cut into for deconvolution, and it sets the GPU memory use.
- With a pickled dict of several PSFs (sampled on a grid over the field of view), set `patch_size` to the spacing of that grid so every patch gets the PSF measured near it.
- `patch_overlap` should be larger than the radius of the PSF. Raising it costs memory and time.
- `threshold` is the setting to adjust when you find too many or too few maxima.
