# Quickstart

This runs MERMAKE on the example field of view in `tests/examples/single` of the repository. You need a GPU and an [installation](installation.md).

Create `quickstart.toml` in the root of the repository. Relative paths are resolved against the folder of the settings file.

```toml
[paths]
psf_file = 'tests/examples/psf_750_Scope0_final.npy'
flat_field_tag = 'tests/examples/Scope0_med_col_raw'
hyb_folders = ['tests/examples/single']
hyb_range = 'H1_TEST_set1:H1_TEST_set1'
output_folder = 'quickstart_out'
```

Check the settings and see what would be processed. This needs no GPU:

```console
$ uv run mermake --dry-run quickstart.toml
----SUMMARY-----
set      H1_TEST
1              1
----------------
1 images in 1 (set, fov) groups
output folder: /path/to/quickstart_out
```

Run it:

```bash
uv run mermake quickstart.toml
```

`quickstart_out/` then holds one file per color channel with the detected maxima, one for the dapi channel and one for the drift:

```
Conv_zscan__020--H1_TEST_set1--col0__Xhfits.npz
Conv_zscan__020--H1_TEST_set1--col1__Xhfits.npz
Conv_zscan__020--H1_TEST_set1--dapiFeatures.npz
drift_Conv_zscan__020--_set1.pkl
```

Look at the maxima of the first color:

```python
import numpy as np

xh = np.load("quickstart_out/Conv_zscan__020--H1_TEST_set1--col0__Xhfits.npz")["Xh"]
print(xh.shape)    # (n_maxima, 8)
print(xh[0, :3])   # z, x, y of the first maximum, in voxels
```

Running the command again does nothing: results that exist are not recomputed. Delete the files to start over. See [Input and output formats](io-formats.md) for the columns and [Configuration](configuration.md) for the settings.
