# MERMAKE

MERMAKE finds the puncta (local intensity maxima) in 3D fluorescence image stacks from MERFISH, smFISH and IF experiments, and measures the drift of each field of view between imaging rounds. Deconvolution and peak finding run on an NVIDIA GPU with [CuPy](https://cupy.dev). Decoding the puncta against a codebook is not part of MERMAKE yet; it is meant to be done on the output.

## Install

You need an NVIDIA GPU. Pick the extra for your CUDA version, `cuda12` or `cuda11`. No CUDA Toolkit or `nvcc` is needed.

```bash
uv sync --extra cuda12          # or: python3 -m pip install "mermake[cuda12]"
```

## Quickstart

Write a settings file with the five required paths:

```toml
[paths]
psf_file = 'psfs/psf_scope3.npy'
flat_field_tag = 'flat_field/Scope3_'
hyb_folders = ['experiment_folder']
hyb_range = 'H1_BICAN_set1:H16_BICAN_set1'
output_folder = 'output'
```

Check it without a GPU, then run it:

```bash
uv run mermake --dry-run settings.toml    # validate and list the images
uv run mermake settings.toml              # process; existing results are skipped
```

`mermake --print-config settings.toml` prints every setting with its description and the value that will be used.

## Documentation

- [Installation](docs/installation.md) and [quickstart on the example data](docs/quickstart.md)
- [Pipeline overview](docs/pipeline.md)
- [Configuration reference](docs/configuration.md)
- [Input and output formats](docs/io-formats.md)
- [Developer guide](docs/developer.md), [known issues](docs/known-issues.md) and the [changelog](CHANGELOG.md)

To build the site: `uv run --group docs mkdocs serve`. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute.

## License

GNU General Public License v3, see [LICENSE](LICENSE).
