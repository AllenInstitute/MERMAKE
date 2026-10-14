import numpy as np

from mermake import threshold


def test_count_above():
	counts = threshold.count_above([1, 2, 3, 4], np.array([0, 2, 4, 5]))
	assert list(counts) == [4, 2, 0, 0]


def test_suggests_where_noise_ends():
	rng = np.random.default_rng(0)
	noise = rng.exponential(1.0, 20000) + 1       # the same distribution in both signs
	signal = rng.normal(20, 3, 500)               # only in the positive maxima
	pos, neg = np.concatenate([noise, signal]), noise
	a = threshold.analyze(pos, neg, n_images=2, floor=1.0, max_fp=0.3)
	i = list(a["thresholds"]).index(a["suggested"])
	assert 2 < a["suggested"] < 14 and a["fp"][i] <= 0.3 < a["fp"][i - 1]
	assert not a["at_floor"]
	# a stricter limit does not give a lower threshold
	strict = threshold.analyze(pos, neg, n_images=2, floor=1.0, max_fp=0.05)
	assert 6 < strict["suggested"] < 14           # between the noise tail and the signal
	# counts are per image and decrease
	assert a["pos"][0] == len(pos) / 2
	assert np.all(np.diff(a["pos"]) <= 0)


def test_clean_data_suggests_floor():
	a = threshold.analyze(np.linspace(5, 50, 100), [], n_images=1, floor=5.0)
	assert a["suggested"] == 5.0 and a["at_floor"]


def test_noise_only_has_no_suggestion():
	noise = np.random.default_rng(1).exponential(1, 5000) + 1
	assert threshold.analyze(noise, noise, n_images=1, floor=1.0)["suggested"] is None


def test_no_maxima():
	a = threshold.analyze([], [], n_images=1, floor=1.0)
	assert a["suggested"] is None


def test_table_and_plot(tmp_path):
	a = threshold.analyze(np.linspace(5, 50, 100), np.linspace(5, 8, 10), 1, 5.0)
	assert "channel 0" in threshold.format_table(0, a)
	import pytest
	pytest.importorskip("matplotlib")
	assert threshold.plot({0: a, "dapi": a}, str(tmp_path / "t.png"))
	assert (tmp_path / "t.png").stat().st_size > 0
