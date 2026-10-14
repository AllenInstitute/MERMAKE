# Installation

## Requirements

- An **NVIDIA GPU** with a recent driver. The deconvolution buffers are large. The memory use is set by [`patch_size`](configuration.md), which can be lowered.
- Python 3.10 or newer.
- [CuPy](https://docs.cupy.dev/en/stable/install.html) built for your CUDA version. MERMAKE installs it for you through an extra.

You do **not** need the CUDA Toolkit or `nvcc`. CuPy compiles the kernels at run time with NVRTC, and the extras install the CUDA headers it needs (`cupy-cuda12x[ctk]`).

## Install

Pick the extra that matches your CUDA version: `cuda12` for CUDA 12.x, `cuda11` for CUDA 11.x. They exclude each other.

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync --extra cuda12
uv run mermake settings.toml
```

With pip:

```bash
python3 -m pip install "mermake[cuda12]"
```

Add the `viz` extra (`mermake[cuda12,viz]`) for the napari viewer behind `mermake --check`.

## Without a GPU

`pip install mermake` or `uv sync` without an extra works on any machine, and `import mermake` does not need CuPy. You can then use the parts that need no GPU: `mermake --print-example`, `mermake --print-config settings.toml` and `mermake --dry-run settings.toml` validate a settings file and list the images that would be processed. Processing itself stops at startup with an error if CuPy cannot find a GPU.

## Troubleshooting

- *CuPy could not find a usable CUDA device*: check `nvidia-smi`, then see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
- *CuPy cannot find CUDA libraries or headers*: use the `cuda12` or `cuda11` extra rather than a bare `cupy-cuda12x`, since the extras include `[ctk]`.
- *Out of GPU memory*: lower `patch_size` in `[hybs]` and `[dapi]`.
