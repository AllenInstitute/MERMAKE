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


# ---- sharding, several workers and robust writes ------------------------------------------------

def mermake(settings, *flags, timeout=600):
	"""Run the ``mermake`` command and return the finished process."""
	return subprocess.run([sys.executable, "-m", "mermake", *flags, str(settings)],
	                      capture_output=True, text=True, timeout=timeout)


def spot_rows(out):
	"""All spots of an output folder as a table sorted so that the order of the maxima does not matter."""
	from mermake.io import spots
	import pyarrow as pa
	table = spots.open_dataset(str(out)).to_table()
	table = table.set_column(table.column_names.index("kind"), "kind", table.column("kind").cast(pa.string()))
	keys = ["iset", "ifov", "ih", "icol", "kind", "z", "x", "y"]
	return table.sort_by([(k, "ascending") for k in keys])


def output_files(out):
	"""Names of every result file below an output folder, relative to it."""
	return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


@pytest.fixture(scope="module")
def three_fovs(tmp_path_factory):
	"""Three fovs, the data they are run on and the output of one plain process, the reference."""
	root = tmp_path_factory.mktemp("shards")
	for fov in range(3):
		write_fov(root, fov)
	write_psf_and_flats(root)
	settings = write_settings(root, root / "out_ref")
	proc = mermake(settings)
	assert proc.returncode == 0, proc.stderr[-2000:]
	return root, settings, root / "out_ref"


def settings_for(root, out_name):
	"""A copy of the settings that writes to ``out_name``."""
	text = (root / "settings.toml").read_text().replace(str(root / "out_ref"), str(root / out_name))
	path = root / f"{out_name}.toml"
	path.write_text(text)
	return path


def test_reference_run_wrote_a_drift_file_per_fov(three_fovs):
	_, _, ref = three_fovs
	assert len(list(ref.glob("drift_*.pkl"))) == 3
	assert not [f for f in output_files(ref) if ".tmp" in f]


def test_shards_together_equal_one_run(three_fovs):
	root, _, ref = three_fovs
	settings = settings_for(root, "out_shards")
	for shard in ("0/2", "1/2"):
		proc = mermake(settings, "--shard", shard)
		assert proc.returncode == 0, proc.stderr[-2000:]
	out = root / "out_shards"
	assert output_files(out) == output_files(ref)
	assert spot_rows(out).equals(spot_rows(ref))


def test_a_shard_only_does_its_own_fovs(three_fovs):
	root, _, ref = three_fovs
	settings = settings_for(root, "out_one_shard")
	proc = mermake(settings, "--shard", "1/2")
	assert proc.returncode == 0, proc.stderr[-2000:]
	assert "shard 1/2: 1 of 3 fov groups" in proc.stdout
	names = output_files(root / "out_one_shard")
	assert any("Conv_zscan__001" in n for n in names)
	assert not any("Conv_zscan__000" in n or "Conv_zscan__002" in n for n in names)
	assert len(list((root / "out_one_shard").glob("drift_*.pkl"))) == 1


def test_a_shard_without_images_exits(three_fovs):
	"""The reader used to die before it could say it was done, so the run never ended."""
	root, _, _ = three_fovs
	settings = settings_for(root, "out_empty_shard")
	proc = mermake(settings, "--shard", "4/5", timeout=120)
	assert proc.returncode == 0, proc.stderr[-2000:]
	assert "No images to process" in proc.stdout
	assert not [f for f in output_files(root / "out_empty_shard") if f.endswith((".npz", ".pkl"))]


def test_workers_on_one_gpu(three_fovs):
	"""``--gpus 0,0`` starts two workers on the same GPU, standing in for two GPUs."""
	root, _, ref = three_fovs
	settings = settings_for(root, "out_workers")
	proc = mermake(settings, "--gpus", "0,0")
	assert proc.returncode == 0, proc.stderr[-2000:]
	out = root / "out_workers"
	assert output_files(out) == output_files(ref)
	assert spot_rows(out).equals(spot_rows(ref))


def test_a_failing_worker_fails_the_run(three_fovs):
	root, _, _ = three_fovs
	settings = settings_for(root, "out_failing")
	text = settings.read_text().replace("psf.npy", "no_such_psf.npy")
	bad = root / "bad_psf.toml"
	bad.write_text(text)
	proc = mermake(bad, "--gpus", "0,0")
	assert proc.returncode != 0


def test_finished_results_are_not_computed_again(three_fovs):
	root, _, ref = three_fovs
	settings = settings_for(root, "out_resume")
	assert mermake(settings, "--shard", "0/3").returncode == 0
	before = {p: p.stat().st_mtime_ns for p in (root / "out_resume").rglob("*") if p.is_file()}
	assert mermake(settings, "--shard", "0/3").returncode == 0
	after = {p: p.stat().st_mtime_ns for p in (root / "out_resume").rglob("*") if p.is_file()}
	assert before == after


class TestAtomicWrites:
	def test_drift_save_leaves_no_partial_file(self, tmp_path):
		from mermake.align import drift_save
		path = tmp_path / "drift_000.pkl"
		drift_save([1, 2, 3], str(path))
		assert path.exists() and [p.name for p in tmp_path.iterdir()] == ["drift_000.pkl"]
		bad = tmp_path / "drift_001.pkl"
		with pytest.raises(Exception):
			drift_save(lambda: None, str(bad))  # cannot be pickled, and fails after the file was opened
		assert not bad.exists()
		assert [p.name for p in tmp_path.iterdir()] == ["drift_000.pkl"]

	def test_save_xfits_leaves_no_partial_file(self, tmp_path, monkeypatch):
		import cupy as cp
		from types import SimpleNamespace
		from mermake.io.queue import ImageQueue

		queue = ImageQueue.__new__(ImageQueue)
		queue.output_folder = str(tmp_path)
		queue.args_array = []
		queue.hyb_save = "{fov}--{tag}--col{icol}__Xhfits.npz"
		queue.dapi_save = "{fov}--{tag}--dapiFeatures.npz"
		image = SimpleNamespace(path=str(tmp_path / "H1_TEST_set1" / "Conv_zscan__000.zarr"),
		                        col0=cp.ones((4, 8), dtype=cp.float32), col1=cp.ones((4, 8), dtype=cp.float32))
		name = "Conv_zscan__000--H1_TEST_set1--col0__Xhfits.npz"
		queue.save_xfits(image, 0)
		assert [p.name for p in tmp_path.iterdir()] == [name]

		def die_halfway(file, **arrays):
			file.write(b"half a file")
			raise OSError("disk full")
		monkeypatch.setattr(cp, "savez_compressed", die_halfway)
		with pytest.raises(OSError, match="disk full"):
			queue.save_xfits(image, 1)
		assert [p.name for p in tmp_path.iterdir()] == [name]


def test_real_fft_off_gives_the_same_spots(tmp_path):
	"""``real_fft = false`` uses complex FFTs, which must find the same spots as the real ones."""
	write_fov(tmp_path)
	write_psf_and_flats(tmp_path)
	settings = write_settings(tmp_path, tmp_path / "out_real")
	assert "real_fft =" not in settings.read_text()  # the default is true
	complex_settings = tmp_path / "complex.toml"
	complex_settings.write_text(settings.read_text().replace(str(tmp_path / "out_real"), str(tmp_path / "out_complex"))
	                            .replace("[hybs]", "[hybs]\nreal_fft = false").replace("[dapi]", "[dapi]\nreal_fft = false"))
	for path in (settings, complex_settings):
		proc = mermake(path)
		assert proc.returncode == 0, proc.stderr[-2000:]
	from scipy.spatial import cKDTree
	real, other = spot_rows(tmp_path / "out_real"), spot_rows(tmp_path / "out_complex")
	assert real.num_rows == other.num_rows > 5
	# match the spots by position: rounding makes the order of equal-looking rows differ
	for kind in ("hyb", "dapi_plus", "dapi_minus"):
		import pyarrow.compute as pc
		pick = {name: t.filter(pc.equal(t.column("kind").cast("string"), kind)) for name, t in (("a", real), ("b", other))}
		a, b = pick["a"], pick["b"]
		assert a.num_rows == b.num_rows
		for icol in sorted(set(a.column("icol").to_pylist())):
			ia, ib = [np.flatnonzero(t.column("icol").to_numpy() == icol) for t in (a, b)]
			xyz = lambda t, i: np.column_stack([t.column(c).to_numpy()[i] for c in ("ih", "z", "x", "y")])  # noqa: E731
			distance, match = cKDTree(xyz(b, ib)).query(xyz(a, ia))
			assert distance.max() < 1e-3
			assert len(set(match)) == len(ia)  # one to one
			np.testing.assert_allclose(a.column("h").to_numpy()[ia], b.column("h").to_numpy()[ib][match], rtol=1e-3)
