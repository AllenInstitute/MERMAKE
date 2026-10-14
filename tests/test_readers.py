"""Tests for reading images and flat fields."""
import numpy as np
import pytest
import zarr

pytest.importorskip("cupy")  # mermake.io.readers imports cupy

from mermake.io.readers import array_path, load_flats, read_im

NCH, NZ, NY, NX = 3, 4, 8, 8
XML = ('<root><z_offsets type="string">0:-15:-0.4:{n}</z_offsets>'
       '<stage_position type="custom">1.0,2.0</stage_position></root>')


def make_stack(seed=0):
	"""A (NCH, NZ, NY, NX) uint8 stack of random pixels, the values of each channel in its own range."""
	rng = np.random.default_rng(seed)
	return np.stack([rng.integers(10 * (i + 1), 10 * (i + 1) + 5, (NZ, NY, NX)) for i in range(NCH)]).astype(np.uint8)


def write_planes(stack):
	"""Planes in the stored order: a dummy plane, then z-major with the channels interleaved."""
	nch, nz, ny, nx = stack.shape
	planes = np.zeros((1 + nz * nch, ny, nx), dtype=stack.dtype)
	for ich in range(nch):
		planes[1 + ich::nch] = stack[ich]
	return planes


def write_fov(root, stack, legacy):
	"""Write a field of view. Current acquisitions store the array in the ``.zarr`` entry itself,
	older ones a group there and the array in ``<fov>/data``."""
	planes = write_planes(stack)
	entry = root / "Conv_zscan__000.zarr"
	if legacy:
		entry.mkdir(parents=True)
		target = root / "000" / "data"
	else:
		target = entry
	z = zarr.open(str(target), mode="w", shape=planes.shape, chunks=(1, *planes.shape[1:]),
	              dtype=planes.dtype, zarr_format=2)
	z[:] = planes
	(root / "Conv_zscan__000.xml").write_text(XML.format(n=stack.shape[0]))
	return str(entry)


class TestLayouts:
	@pytest.mark.parametrize("legacy", [False, True])
	def test_read_im(self, tmp_path, legacy):
		stack = make_stack()
		image = read_im(write_fov(tmp_path, stack, legacy)).compute()
		assert image.shape == stack.shape
		assert image.dtype == np.uint16
		np.testing.assert_array_equal(image, stack.astype(np.uint16) ** 2)

	def test_array_path(self, tmp_path):
		new = write_fov(tmp_path / "new", make_stack(), legacy=False)
		old = write_fov(tmp_path / "old", make_stack(), legacy=True)
		assert array_path(new) == new
		assert array_path(old) == str(tmp_path / "old" / "000" / "data")


@pytest.mark.gpu
class TestFlats:
	def test_npy_and_npz(self, tmp_path):
		rng = np.random.default_rng(0)
		a = rng.uniform(5, 10, (NY, NX)).astype(np.float32)
		b = rng.uniform(5, 10, (NY, NX)).astype(np.float32)
		np.save(tmp_path / "flat_0.npy", a)
		np.savez(tmp_path / "flat_1.npz", im=b)
		shape = (2, NZ, NY, NX)
		mixed = load_flats(str(tmp_path / "flat_"), shape=shape).get()
		only_npz = tmp_path / "npz"
		only_npz.mkdir()
		np.savez(only_npz / "flat_0.npz", im=a)
		np.savez(only_npz / "flat_1.npz", im=b)
		expected = load_flats(str(only_npz / "flat_"), shape=shape).get()
		assert mixed.shape == (2, NY, NX)
		np.testing.assert_allclose(mixed, expected)
