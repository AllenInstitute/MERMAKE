# MERMAKE

MERMAKE finds the puncta (local intensity maxima) in 3D fluorescence image stacks from MERFISH, smFISH and IF experiments, and measures how each field of view drifts between imaging rounds. The heavy lifting, deconvolution and peak finding, runs on an NVIDIA GPU through [CuPy](https://cupy.dev).

For every field of view and every hybridization round MERMAKE:

1. flat-field corrects and deconvolves each color channel patch by patch,
2. detects local maxima and refines them to sub-voxel positions, with a few quality scores each,
3. does the same for the dapi channel, and uses those points to measure the drift between rounds.

The results are plain `.npz` and `.pkl` files, see [Input and output formats](io-formats.md). Decoding the puncta against a codebook is not part of MERMAKE yet, it is meant to be done on the output.

## Where to start

| I want to... | Go to |
|---|---|
| install it | [Installation](installation.md) |
| run it on the example data | [Quickstart](quickstart.md) |
| understand what it does to my images | [Pipeline overview](pipeline.md) |
| know what a setting means | [Configuration](configuration.md) |
| read the output files | [Input and output formats](io-formats.md) |
| change the code | [Developer guide](developer.md) |

MERMAKE is released under the GNU General Public License v3.
