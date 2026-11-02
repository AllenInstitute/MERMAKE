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

## Several GPUs and nodes

One `mermake` process drives one GPU. To use more, run one process per GPU, each on its own share of the fields of view (every round of a field of view stays in the same share, because its drift needs them all):

```bash
uv run mermake --gpus all settings.toml        # one worker per GPU of this machine (or --gpus 0,2)
uv run mermake --shard 2/8 settings.toml       # one process: share 2 (from 0) of 8
sbatch --array=0-7 scripts/run_slurm_parallel.sbatch   # SLURM: 8 shards, one GPU each
```

Results are written one field of view at a time, each file under a temporary name until it is complete, so a run that is killed or hits a time limit continues where it stopped when you run it again. Images are read ahead of the GPU, whole, so a process needs RAM for about `--prefetch` + 3 images (13-21 GiB for 4 x 40 x 2800 x 2800 data); `--prefetch 2` is enough when the GPU is the bottleneck. Check that your storage can feed several GPUs with `scripts/bench_read.py`. See [several GPUs and nodes](docs/scaling.md), which also has the measured memory and timings. The SLURM scripts have not been run on a real cluster yet.

## Speed and results

On a Tesla T4 a 4 x 40 x 2800 x 2800 image takes about 8 s with the default settings (26 s in version 0.1.0). Two defaults changed to get there, and they change the results a little:

- `patch_overlap` is now **110** in `[hybs]` and `[dapi]` (it was 89). `patch_size + 2 * patch_overlap` should be a product of 2, 3, 5 and 7 for a fast FFT: 500 + 2 x 110 = 720 is, 500 + 2 x 89 = 678 = 2 x 3 x 113 was not. If you change `patch_size`, change `patch_overlap` to keep that.
- `real_fft = true` (new) uses real-to-complex FFTs, which take half the time and memory.

About 0.4 % of the spots, those within about 1 % of the threshold, differ from version 0.1.0. To reproduce its results exactly, set this in your settings file:

```toml
[hybs]
patch_overlap = 89
real_fft = false

[dapi]
patch_overlap = 89
real_fft = false
```

`scripts/bench_pipeline.py` shows where the time and the GPU memory of your own run go. See the [changelog](CHANGELOG.md) and [where the time goes](docs/scaling.md).

## Documentation

- [Installation](docs/installation.md) and [quickstart on the example data](docs/quickstart.md)
- [Pipeline overview](docs/pipeline.md)
- [Configuration reference](docs/configuration.md)
- [Several GPUs and nodes](docs/scaling.md): sharding, SLURM, memory per process and where the time goes
- [Input and output formats](docs/io-formats.md)
- [Developer guide](docs/developer.md), [known issues](docs/known-issues.md) and the [changelog](CHANGELOG.md)

To build the site: `uv run --group docs mkdocs serve`. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute.

## About this fork

This is the Allen Institute fork of [deprekate/mermake](https://github.com/deprekate/mermake), which it started from at version 0.0.74. It was restructured (a `src/` layout, validated settings, tests, documentation) and has diverged: the settings files, the default `patch_overlap` and, for stacks with an odd number of planes, the z coordinates differ from upstream. See the [changelog](CHANGELOG.md).

## License

GNU General Public License v3, see [LICENSE](LICENSE).
