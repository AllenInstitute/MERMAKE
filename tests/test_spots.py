import numpy as np
import pytest

pytest.importorskip("pyarrow")
import pyarrow.dataset as ds

from mermake.io import spots


def xh(n, seed=0):
	return np.random.default_rng(seed).random((n, 8)).astype(np.float32)


def test_roundtrip_hyb(tmp_path):
	X = xh(5)
	spots.write_spots(str(tmp_path), 1, 3, 7, 0, {"hyb": X})
	assert spots.spots_exist(str(tmp_path), 1, 3, 7, 0)
	assert not spots.spots_exist(str(tmp_path), 1, 3, 8, 0)
	np.testing.assert_array_equal(spots.read_spots(str(tmp_path), 1, 3, 7, 0)["hyb"], X)


def test_roundtrip_dapi_and_empty(tmp_path):
	plus, minus = xh(4), np.zeros((0, 8), np.float32)
	spots.write_spots(str(tmp_path), 1, 3, 7, -1, {"dapi_plus": plus, "dapi_minus": minus})
	back = spots.read_spots(str(tmp_path), 1, 3, 7, -1)
	np.testing.assert_array_equal(back["dapi_plus"], plus)
	assert back["dapi_minus"].shape == (0, 8)


def test_existing_part_is_not_overwritten(tmp_path):
	spots.write_spots(str(tmp_path), 1, 3, 7, 0, {"hyb": xh(5)})
	spots.write_spots(str(tmp_path), 1, 3, 7, 0, {"hyb": xh(2, seed=1)})
	assert len(spots.read_spots(str(tmp_path), 1, 3, 7, 0)["hyb"]) == 5


def test_dataset_filters_by_fov(tmp_path):
	for ifov in (0, 1, 2):
		for ih in (1, 2):
			spots.write_spots(str(tmp_path), 1, ifov, ih, 0, {"hyb": xh(3 + ifov)})
	dataset = spots.open_dataset(str(tmp_path))
	table = dataset.to_table(filter=ds.field("ifov") == 1)
	assert table.num_rows == 2 * 4
	assert set(table.column("ifov").to_pylist()) == {1}
	assert set(table.column("ih").to_pylist()) == {1, 2}


def test_no_partial_files_left(tmp_path):
	spots.write_spots(str(tmp_path), 1, 0, 1, 0, {"hyb": xh(2)})
	assert not list(tmp_path.rglob("*.tmp*"))


def test_stats(tmp_path):
	X = xh(100)
	X[:, 4] = np.nan  # scores are NaN with delta_fit 0
	stats = spots.spot_stats(X, raw_sum=2.0, deconv_sum=1.0)
	assert stats["n_spots"] == 100
	assert np.isnan(stats["a_p50"])
	assert stats["h_p05"] < stats["h_p50"] < stats["h_p95"]
	assert spots.spot_stats(np.zeros((0, 8)))["n_spots"] == 0
	with pytest.raises(KeyError):
		spots.spot_stats(X, bogus=1)
	spots.write_stats(str(tmp_path), 1, 0, 1, 0, {"hyb": stats})
	table = spots.open_dataset(str(tmp_path), spots.DIAGNOSTICS_DIR).to_table()
	assert table.num_rows == 1 and table.column("raw_sum").to_pylist() == [2.0]
