import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")

from mermake import report
from mermake.io import spots


def test_make_report(tmp_path):
	rng = np.random.default_rng(0)
	root = str(tmp_path)
	for ih in (1, 2, 3):
		for icol in (0, 1):
			X = rng.random((50, 8)).astype(np.float32) * ih
			spots.write_spots(root, 1, 0, ih, icol, {"hyb": X})
			stats = spots.spot_stats(X, raw_sum=10.0, deconv_sum=9.0)
			spots.write_stats(root, 1, 0, ih, icol, {"hyb": stats})
		spots.write_spots(root, 1, 0, ih, -1, {"dapi_plus": rng.random((5, 8)), "dapi_minus": rng.random((5, 8))})
		spots.write_stats(root, 1, 0, ih, -1, {"dapi_plus": spots.spot_stats(rng.random((5, 8)), std_val=1.0)})
	written = report.make_report(root)
	names = {p.rsplit("/", 1)[-1] for p in written}
	assert {"hist_h_col0.png", "hist_h_col1.png", "n_spots.png", "deconv_raw_ratio.png"} <= names
	assert all((tmp_path / "report" / n).stat().st_size > 0 for n in names)


def test_histograms_share_bins_per_round():
	import pyarrow as pa
	table = pa.table({"iset": [1] * 6, "ih": [1, 1, 1, 2, 2, 2], "h": [1., 2., 3., 4., 5., 6.]})
	edges, hists = report.score_histograms(table, "h", bins=10)
	assert list(hists) == [(1, 1), (1, 2)] and len(edges) == 11
