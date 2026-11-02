"""CPU-only tests for filename parsing and FolderFilter."""
import pytest

from mermake.io.naming import FolderFilter, get_ifov, get_ih, get_iset, shard_keys, shard_matches

REGEX = r"""([A-z]+)(\d+)_(.+)_set(\d+)(.*)"""


class TestGetIh:
	def test_first_number(self):
		assert get_ih("H12_MER_set1") == 12

	def test_uses_basename_only(self):
		# digits in parent directories must not leak in
		assert get_ih("/data/exp2024/H33_MER_set1") == 33

	def test_no_number_sorts_last(self):
		assert get_ih("background") == 10**100

	def test_sorts_numerically(self):
		names = ["H10_a_set1", "H2_a_set1", "H1_a_set1"]
		assert sorted(names, key=get_ih) == ["H1_a_set1", "H2_a_set1", "H10_a_set1"]


class TestGetIfov:
	def test_zarr_name(self):
		assert get_ifov("Conv_zscan__020.zarr") == 20

	def test_full_path(self):
		assert get_ifov("/data/H1_A_set1/Conv_zscan__007.zarr") == 7

	def test_missing_digits_raises(self):
		with pytest.raises(ValueError):
			get_ifov("Conv_zscan.zarr")

	def test_not_zarr_raises(self):
		with pytest.raises(ValueError):
			get_ifov("Conv_zscan__020.tif")


class TestGetIset:
	def test_from_filename(self):
		assert get_iset("H1_MER_set2") == 2

	def test_from_parent_directory(self):
		assert get_iset("/data/H1_MER_set3/Conv_zscan__001.zarr") == 3

	def test_missing_raises(self):
		with pytest.raises(ValueError):
			get_iset("/data/no/such/thing")


class TestFolderFilterIsin:
	@pytest.fixture
	def ff(self):
		return FolderFilter("H1_*_set1:H16_*_set2", REGEX, float("-inf"), float("inf"))

	@pytest.mark.parametrize("name", ["H1_MER_set1", "H8_MER_set2", "H16_anything_set1"])
	def test_inside(self, ff, name):
		assert ff.isin(name)

	@pytest.mark.parametrize("name", [
		"H17_MER_set1",   # hyb above range
		"H0_MER_set1",    # hyb below range
		"H5_MER_set3",    # set above range
		"X5_MER_set1",    # wrong prefix
		"not_a_hyb",      # does not match the regex
	])
	def test_outside(self, ff, name):
		assert not ff.isin(name)

	def test_explicit_middle_must_match(self):
		ff = FolderFilter("H1_BICAN_set1:H16_BICAN_set1", REGEX, float("-inf"), float("inf"))
		assert ff.isin("H4_BICAN_set1")
		assert not ff.isin("H4_OTHER_set1")

	def test_single_round(self):
		ff = FolderFilter("H1_TEST_set1:H1_TEST_set1", REGEX, float("-inf"), float("inf"))
		assert ff.isin("H1_TEST_set1")
		assert not ff.isin("H2_TEST_set1")


class TestUndoRegex:
	def test_roundtrip(self):
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, 0, 1)
		parts = ff._parse_pattern("H1_MER_set2")
		assert ff._undo_regex(parts) == "H1_MER_set2"

	def test_none(self):
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, 0, 1)
		assert ff._undo_regex(None) is None

	def test_partial_groups_stop_after_last(self):
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, 0, 1)
		parts = ff._parse_pattern("H1_MER_set2")[:3]
		assert ff._undo_regex(parts) == "H1_MER"


class TestGetMatches:
	@pytest.fixture
	def tree(self, tmp_path):
		for hyb in ["H1_MER_set1", "H2_MER_set1", "H9_MER_set1", "H1_MER_set2"]:
			for fov in [0, 1, 5]:
				(tmp_path / hyb / f"Conv_zscan__{fov:03d}.zarr").mkdir(parents=True)
		(tmp_path / "H1_MER_set1" / "notes.txt").write_text("not a zarr")
		return tmp_path

	def test_groups_by_set_and_fov(self, tree):
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, float("-inf"), float("inf"))
		matches = ff.get_matches([str(tree)])
		assert set(matches) == {(1, 0), (1, 1), (1, 5)}
		assert all(len(paths) == 2 for paths in matches.values())  # H1 and H2
		assert all(p.endswith(".zarr") for paths in matches.values() for p in paths)

	def test_fov_range(self, tree):
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, 0, 1)
		assert set(ff.get_matches([str(tree)])) == {(1, 0), (1, 1)}

	def test_set_range(self, tree):
		ff = FolderFilter("H1_*_set1:H1_*_set2", REGEX, float("-inf"), float("inf"))
		isets = {iset for iset, _ in ff.get_matches([str(tree)])}
		assert isets == {1, 2}

	def test_missing_folder_is_ignored(self, tree):
		ff = FolderFilter("H1_*_set1:H1_*_set1", REGEX, float("-inf"), float("inf"))
		assert ff.get_matches([str(tree / "does_not_exist")]) == {}

	def test_no_matches(self, tree):
		ff = FolderFilter("H20_*_set1:H30_*_set1", REGEX, float("-inf"), float("inf"))
		assert ff.get_matches([str(tree)]) == {}


class TestShardKeys:
	KEYS = [(1, 5), (1, 0), (2, 3), (1, 2), (2, 1), (1, 9), (1, 7)]

	def test_disjoint_and_complete(self):
		shards = [shard_keys(self.KEYS, i, 3) for i in range(3)]
		flat = [k for shard in shards for k in shard]
		assert sorted(flat) == sorted(self.KEYS)
		assert len(flat) == len(set(flat))

	def test_does_not_depend_on_the_input_order(self):
		a = shard_keys(self.KEYS, 1, 3)
		assert shard_keys(reversed(self.KEYS), 1, 3) == a
		assert shard_keys(set(self.KEYS), 1, 3) == a

	def test_every_nth_of_the_sorted_keys(self):
		# sorted: (1,0) (1,2) (1,5) (1,7) (1,9) (2,1) (2,3)
		assert shard_keys(self.KEYS, 0, 3) == [(1, 0), (1, 7), (2, 3)]
		assert shard_keys(self.KEYS, 1, 3) == [(1, 2), (1, 9)]
		assert shard_keys(self.KEYS, 2, 3) == [(1, 5), (2, 1)]
		assert shard_keys(self.KEYS, 0, 1) == sorted(self.KEYS)

	def test_more_shards_than_keys(self):
		assert shard_keys([(1, 0), (1, 1)], 2, 5) == []
		assert shard_keys([], 0, 2) == []

	@pytest.mark.parametrize("index, count", [(3, 3), (-1, 3), (0, 0), (1, 1)])
	def test_bad_shard(self, index, count):
		with pytest.raises(ValueError, match="index"):
			shard_keys(self.KEYS, index, count)

	def test_matches_are_restricted_and_sorted(self):
		matches = {(1, 3): ["c"], (1, 1): ["a1", "a2"], (1, 2): ["b"]}
		assert list(shard_matches(matches)) == [(1, 1), (1, 2), (1, 3)]
		assert shard_matches(matches, (1, 2)) == {(1, 2): ["b"]}
