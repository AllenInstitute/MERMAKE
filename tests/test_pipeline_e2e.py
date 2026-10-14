"""End-to-end test: run the ``mermake`` command on a small synthetic fov.

The fov has 3 channels (2 hyb colors and a dapi channel) with bright gaussian
spots planted at known positions. The test checks that the pipeline writes
its outputs and that every planted spot is reported as a local maximum.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

NZ, NY, NX = 12, 96, 96   # z planes per channel, image height and width
NCOL = 3                  # hyb colors plus the trailing dapi channel
OVERLAP = 8

# (z, y, x) of the planted spots, well away from the patch borders
SPOTS = {
	0: [(5, 30, 30), (5, 60, 70), (3, 45, 20)],
	1: [(5, 25, 65), (7, 70, 40), (4, 50, 50)],
	2: [(5, 30, 60), (5, 65, 35)],
}


def plant(spots, amp):
	"""A (NZ, NY, NX) float image with a gaussian at each (z, y, x) spot."""
	zz, yy, xx = np.meshgrid(np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij")
	im = np.zeros((NZ, NY, NX), dtype=np.float32)
	for z, y, x in spots:
		im += amp * np.exp(-((zz - z) ** 2 / (2 * 1.0**2) + ((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5**2)))
	return im


def write_fov(root, fov=0):
	"""Write one fov in the layout read_im expects. Returns the hyb folder."""
	import zarr

	hyb = root / "exp" / "H1_TEST_set1"
	(hyb / f"Conv_zscan__{fov:03d}.zarr").mkdir(parents=True)
	rng = np.random.default_rng(0)
	# plane 0 is dropped by read_im; then z-major with the channels interleaved
	planes = np.zeros((1 + NZ * NCOL, NY, NX), dtype=np.uint8)
	for icol in range(NCOL):
		im = plant(SPOTS[icol], amp=120) + rng.uniform(0, 4, (NZ, NY, NX))
		planes[1 + icol::NCOL] = np.clip(im, 0, 255).astype(np.uint8)
	z = zarr.open(str(hyb / f"{fov:03d}" / "data"), mode="w", shape=planes.shape,
	              chunks=(1, NY, NX), dtype=planes.dtype, zarr_format=2)
	z[:] = planes
	(hyb / f"Conv_zscan__{fov:03d}.xml").write_text(
		'<root><z_offsets type="string">0:-15:-0.4:3</z_offsets>'
		'<stage_position type="custom">-4351.10,5862.40</stage_position></root>'
	)
	return hyb


def write_psf_and_flats(root):
	zz, yy, xx = np.meshgrid(np.arange(-3, 3), np.arange(-5, 5), np.arange(-5, 5), indexing="ij")
	psf = np.exp(-(zz**2 / (2 * 1.0**2) + (yy**2 + xx**2) / (2 * 1.5**2))).astype(np.float32)
	psf /= psf.sum()
	np.save(root / "psf.npy", psf)
	for i in range(NCOL):
		np.savez(root / f"flat_{i}.npz", im=np.full((NY + 8, NX + 8), 100, dtype=np.uint8))


def write_settings(root, out):
	section = """
		patch_size = {patch}
		patch_overlap = {overlap}
		beta = {beta}
		threshold = {threshold}
		blur_radius = {blur}
		delta = {delta}
		delta_fit = {delta_fit}
		sigmaZ = 1
		sigmaXY = 1.5
	"""
	hybs = section.format(patch=NY, overlap=OVERLAP, beta=0.0001, threshold=1000, blur=20, delta=1, delta_fit=3)
	dapi = section.format(patch=NY, overlap=OVERLAP, beta=0.01, threshold=3.0, blur=20, delta=2, delta_fit=3)
	text = textwrap.dedent(f"""\
		[output]
		format = 'both'

		[paths]
		psf_file = '{root / "psf.npy"}'
		flat_field_tag = '{root / "flat_"}'
		hyb_range = 'H1_TEST_set1:H1_TEST_set1'
		hyb_folders = ['{root / "exp"}']
		output_folder = '{out}'

		[hybs]""") + textwrap.dedent(hybs) + "\n[dapi]" + textwrap.dedent(dapi)
	path = root / "settings.toml"
	path.write_text(text)
	return path


def assert_spots_found(xh, spots, tol=1.5):
	"""Every planted (z, y, x) spot has a detected maximum within ``tol`` voxels.

	Xh columns 0-2 are z, x, y of each maximum (x indexes the 2nd image axis).
	"""
	found = xh[:, :3]
	for z, y, x in spots:
		d = np.linalg.norm(found - np.array([z, y, x]), axis=1)
		assert d.min() < tol, f"no maximum within {tol} voxels of planted spot {(z, y, x)}"


@pytest.fixture(scope="module")
def run_output(tmp_path_factory):
	root = tmp_path_factory.mktemp("e2e")
	write_fov(root)
	write_psf_and_flats(root)
	out = root / "out"
	settings = write_settings(root, out)
	proc = subprocess.run([sys.executable, "-m", "mermake", str(settings)],
	                      capture_output=True, text=True, timeout=600)
	return proc, out


def test_exits_cleanly(run_output):
	proc, _ = run_output
	assert proc.returncode == 0, proc.stderr[-2000:]


def test_writes_expected_files(run_output):
	_, out = run_output
	names = {p.name for p in out.iterdir()}
	assert "Conv_zscan__000--H1_TEST_set1--col0__Xhfits.npz" in names
	assert "Conv_zscan__000--H1_TEST_set1--col1__Xhfits.npz" in names
	assert "Conv_zscan__000--H1_TEST_set1--dapiFeatures.npz" in names


@pytest.mark.parametrize("icol", [0, 1])
def test_hyb_spots_detected(run_output, icol):
	_, out = run_output
	data = np.load(out / f"Conv_zscan__000--H1_TEST_set1--col{icol}__Xhfits.npz")
	assert data["Xh"].shape[1] == 8
	assert_spots_found(data["Xh"], SPOTS[icol])


def test_dapi_spots_detected(run_output):
	_, out = run_output
	data = np.load(out / "Conv_zscan__000--H1_TEST_set1--dapiFeatures.npz")
	assert_spots_found(data["Xh_plus"], SPOTS[2])


def test_parquet_spots_and_diagnostics(run_output):
	_, out = run_output
	from mermake.io import spots
	table = spots.open_dataset(str(out)).to_table()
	assert {"iset", "ifov", "ih", "icol", "kind", "h"} <= set(table.column_names)
	assert set(table.column("kind").to_pylist()) >= {"hyb", "dapi_plus"}
	import pyarrow.compute as pc
	hyb0 = table.filter(pc.equal(table.column("icol"), 0)).column("z").to_numpy()
	data = np.load(out / "Conv_zscan__000--H1_TEST_set1--col0__Xhfits.npz")
	assert len(hyb0) == len(data["Xh"])
	diagnostics = spots.open_dataset(str(out), spots.DIAGNOSTICS_DIR).to_table()
	assert diagnostics.num_rows >= 3


def test_suggest_threshold(tmp_path):
	write_fov(tmp_path)
	write_psf_and_flats(tmp_path)
	settings = write_settings(tmp_path, tmp_path / "out")
	proc = subprocess.run([sys.executable, "-m", "mermake", "--suggest-threshold",
	                       "--sample-images", "1", str(settings)],
	                      capture_output=True, text=True, timeout=600)
	assert proc.returncode == 0, proc.stderr[-2000:]
	assert "channel 0" in proc.stdout and "channel dapi" in proc.stdout
	assert "Suggested thresholds" in proc.stdout
