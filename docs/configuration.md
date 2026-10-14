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

None of them needs a GPU.

## Reference

This section is generated from the code (`mermake.config`).

<!-- config-reference -->

## Tuning notes

- `patch_size` is **not** the size of your images. It is the edge of the square patches each z-stack is cut into for deconvolution, and it sets the GPU memory use.
- With a pickled dict of several PSFs (sampled on a grid over the field of view), set `patch_size` to the spacing of that grid so every patch gets the PSF measured near it.
- `patch_overlap` should be larger than the radius of the PSF. Raising it costs memory and time.
- `threshold` is the setting to adjust when you find too many or too few maxima.
