# Input and output formats

## Input

### Image folders

```
<hyb_folders entry>/
  H1_BICAN_set1/                  one folder per hybridization round ("tag")
    Conv_zscan__020.zarr/         one entry per field of view
    Conv_zscan__020.xml           metadata written by the microscope
    020/data/                     the zarr array with the images
  H2_BICAN_set1/
    ...
```

- Round folders are selected with `hyb_range` and `regex`. The default `regex` splits a name like `H1_BICAN_set1` into prefix (`H`), round number (`1`), middle part (`BICAN`), set number (`1`) and suffix.
- The field of view number is the last number before `.zarr` in the entry name (`020`). The set number is the number after `_set` in the round folder name.
- MERMAKE uses the *name* of the `Conv_zscan__020.zarr` entry to find the field of view, and reads the images from the zarr array `020/data` next to it.

### The zarr array and the XML

The first plane of the array is dropped. The remaining planes are ordered z-major with the color channels interleaved. `Conv_zscan__020.xml` gives the number of channels, as the last number of its `z_offsets` field (`0:-15:-0.4:3` means 3 channels), and the stage position (`stage_position`). The image is reshaped to `(channels, z, y, x)`. The **last channel is the dapi channel**, the others are the hyb colors. 8-bit data is squared and converted to uint16.

All images are assumed to have the shape of the first one.

### PSF

`psf_file` is read with `numpy.load`. It is either

- a `(z, y, x)` array with the PSF peak at index `n // 2` along each axis (the usual convention for an even or odd length), or
- a pickled `dict` of such arrays, sampled on a grid over the field of view, which are used one per patch, in order.

The PSF is cropped or zero padded to `(depth, patch_size, patch_size)` and normalized to sum 1.

### Flat fields

Every file that matches `<flat_field_tag>*` is a flat field, in sorted order, one per channel with the dapi channel last. Each is an `.npz` with the 2D array `im`, at least as large as the image in y and x. It is center cropped, blurred over 20 x 20 pixels and divided by its median.

## Output

Everything goes to `output_folder`. The names come from the templates in `[paths]`, with `{fov}` the name of the zarr entry without the extension (`Conv_zscan__020`) and `{tag}` the round folder (`H1_BICAN_set1`).

| File | Default name | Contents |
|---|---|---|
| maxima of color channel `icol` | `{fov}--{tag}--col{icol}__Xhfits.npz` | `Xh` |
| dapi maxima | `{fov}--{tag}--dapiFeatures.npz` | `Xh_plus`, `Xh_minus` |
| drift of a field of view | `drift_{fov}--_set{iset}.pkl` | see below |

All `.npz` files also hold `version` (the MERMAKE version) and `args`, the settings of the run as rows of `(section, key, value)` strings.

### Maxima: `Xh`, `Xh_plus`, `Xh_minus`

float32 arrays with one row per maximum and 8 columns, in voxels of the full image.

| Column | Name | Meaning |
|---|---|---|
| 0, 1, 2 | z, x, y | refined position: the intensity-weighted centroid of the deconvolved image in the `delta_fit` neighborhood. `x` indexes the second image axis and `y` the third |
| 3 | background | lowest deconvolved value in the `delta_fit` neighborhood, which is subtracted before the centroid is taken |
| 4 | `a` | normalized cross-correlation of the Gaussian template (`sigmaZ`, `sigmaXY`) with the raw image in the neighborhood, at most about 1 |
| 5 | `habs` | raw image value at the maximum voxel |
| 6 | `hn` | the same correlation with the deconvolved image |
| 7 | `h` | deconvolved image value at the maximum voxel |

`Xh_plus` holds the maxima of the dapi image and `Xh_minus` those of the negated dapi image. For the dapi channel `h` is in units of standard deviations of the image.

### Drift file

A pickle of the list `[drifts, files, ifov, ref_path]`:

- `drifts`: one tuple `(tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus)` per round, in processing order. `tzxyf` is the `(z, x, y)` shift in voxels that corrects that round, from the plus and minus alignments together.
- `files`: the round folder of each entry of `drifts`.
- `ifov`: the field of view number.
- `ref_path`: the image that was used as the reference, the middle round.

!!! warning
    The arrays inside are CuPy arrays, so unpickling needs CuPy and a GPU (see [known issues](known-issues.md)).

## Tips

**Only process the dapi channels.** Results that exist are not recomputed. If `hyb_save` is a fixed name such as `'Xhfits.npz'` and a valid `.npz` with an `Xh` array of that name is in the output folder, every color channel counts as done, and only the dapi channel is processed (`dapi_save` stays dynamic).
