"""Tests for reading the images from a URL (``s3://...``) instead of a local path.

The remote branch is tested on fsspec's in-memory file system (``memory://``), which behaves like
S3 for listing and reading. Set ``MERMAKE_S3_TEST_URL`` to a dataset folder (the parent of the round
folders) to also run a test against real object storage.
"""
import os
import uuid

import fsspec
import numpy as np
import pytest

pytest.importorskip("cupy")  # mermake.io.readers imports cupy

from mermake.config import Config, ConfigError, namespace_to_array
from mermake.io import storage
from mermake.io.naming import FolderFilter
from mermake.io.readers import array_path, read_im

from test_config import REQUIRED
from test_readers import make_stack, write_fov

REGEX = r'([A-z]+)(\d+)_(.+)_set(\d+)(.*)'
ALL = (float("-inf"), float("inf"))


@pytest.fixture(autouse=True)
def reset_storage():
	storage.configure()
	yield
	storage.configure()


@pytest.fixture
def memory_root():
	"""A fresh ``memory://`` folder, removed afterwards."""
	fs = fsspec.filesystem("memory")
	root = f"memory://mermake-test-{uuid.uuid4().hex}"
	yield root
	if fs.exists(root):
		fs.rm(root, recursive=True)


def upload(local, url):
	fsspec.filesystem("memory").put(str(local), url, recursive=True)


class TestHelpers:
	def test_is_remote(self):
		assert storage.is_remote("s3://bucket/prefix")
		assert storage.is_remote("memory://x")
		assert not storage.is_remote("/data/x")
		assert not storage.is_remote("relative/x")
		assert not storage.is_remote(None)

	def test_s3_pool_follows_read_threads(self):
		storage.configure({"profile": "lab"}, read_threads=64)
		opts = storage.storage_options("s3://bucket/x")
		assert opts["profile"] == "lab"
		assert opts["config_kwargs"]["max_pool_connections"] == 64
		storage.configure(read_threads=4)
		assert storage.storage_options("s3://bucket/x")["config_kwargs"]["max_pool_connections"] == 10
		assert storage.storage_options("memory://x") == {}

	def test_listdir_local_and_remote_agree(self, tmp_path, memory_root):
		(tmp_path / "a.zarr").mkdir()
		(tmp_path / "a.zarr" / ".zarray").write_text("{}")
		(tmp_path / "b.txt").write_text("x")
		upload(tmp_path, memory_root)
		local = sorted((n, d) for n, _, d in storage.listdir(str(tmp_path)))
		remote = storage.listdir(memory_root)
		assert sorted((n, d) for n, _, d in remote) == local == [("a.zarr", True), ("b.txt", False)]
		assert all(p == f"{memory_root}/{n}" for n, p, _ in remote)

	def test_read_text(self, tmp_path, memory_root):
		(tmp_path / "f.xml").write_text("<root/>")
		upload(tmp_path, memory_root)
		assert storage.read_text(str(tmp_path / "f.xml")) == storage.read_text(f"{memory_root}/f.xml") == "<root/>"
		assert storage.exists(f"{memory_root}/f.xml") and not storage.exists(f"{memory_root}/g.xml")

	def test_missing_package_is_explained(self, monkeypatch):
		def no_s3fs(protocol, **kwargs):
			raise ImportError("Install s3fs to access S3")
		monkeypatch.setattr(fsspec, "filesystem", no_s3fs)
		with pytest.raises(ImportError, match=r"mermake\[s3\]"):
			storage.listdir("s3://bucket/prefix")


class TestRemoteMatches:
	@pytest.fixture
	def tree(self, tmp_path):
		for hyb in ["H1_MER_set1", "H2_MER_set1", "H9_MER_set1", "H1_MER_set2"]:
			for fov in [0, 1, 5]:
				entry = tmp_path / hyb / f"Conv_zscan__{fov:03d}.zarr"
				entry.mkdir(parents=True)
				(entry / ".zarray").write_text("{}")   # object storage has no empty folders
		(tmp_path / "H1_MER_set1" / "notes.txt").write_text("not a zarr")
		return tmp_path

	@pytest.mark.parametrize("hyb_range,fovs", [("H1_*_set1:H2_*_set1", ALL), ("H1_*_set1:H1_*_set2", ALL),
	                                            ("H1_*_set1:H9_*_set1", (0, 1))])
	def test_same_as_local(self, tree, memory_root, hyb_range, fovs):
		upload(tree, memory_root)
		ff = FolderFilter(hyb_range, REGEX, *fovs)
		local = ff.get_matches([str(tree)])
		remote = ff.get_matches([memory_root])
		assert local and set(remote) == set(local)
		for key in local:
			rel = sorted(os.path.relpath(p, tree) for p in local[key])
			assert sorted(p[len(memory_root) + 1:] for p in remote[key]) == rel
			assert all(p.startswith(memory_root + "/") for p in remote[key])

	def test_missing_folder_is_ignored(self, memory_root):
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, *ALL)
		assert ff.get_matches([f"{memory_root}/does_not_exist"]) == {}

	def test_unreadable_folder_is_an_error(self, monkeypatch):
		def denied(path):
			raise PermissionError("Access Denied")
		monkeypatch.setattr(storage, "listdir", denied)
		ff = FolderFilter("H1_*_set1:H2_*_set1", REGEX, *ALL)
		with pytest.raises(PermissionError, match="credentials"):
			ff.get_matches(["s3://bucket/prefix"])


class TestRemoteRead:
	@pytest.mark.parametrize("legacy", [False, True])
	def test_read_im_same_as_local(self, tmp_path, memory_root, legacy):
		stack = make_stack()
		local = write_fov(tmp_path, stack, legacy)
		upload(tmp_path, memory_root)
		remote = f"{memory_root}/{os.path.basename(local)}"
		np.testing.assert_array_equal(read_im(remote).compute(), read_im(local).compute())

	def test_array_path(self, tmp_path, memory_root):
		write_fov(tmp_path / "new", make_stack(), legacy=False)
		write_fov(tmp_path / "old", make_stack(), legacy=True)
		upload(tmp_path, memory_root)
		new, old = f"{memory_root}/new/Conv_zscan__000.zarr", f"{memory_root}/old/Conv_zscan__000.zarr"
		assert array_path(new) == new
		assert array_path(old) == f"{memory_root}/old/000/data"

	def test_without_xml_the_first_plane_is_dropped(self, tmp_path, memory_root):
		local = write_fov(tmp_path, make_stack(), legacy=False)
		os.remove(tmp_path / "Conv_zscan__000.xml")
		upload(tmp_path, memory_root)
		remote = f"{memory_root}/Conv_zscan__000.zarr"
		assert read_im(remote).shape == read_im(local).shape


class TestConfig:
	def settings(self, **paths):
		return {"paths": {**REQUIRED, **paths}}

	def test_url_is_not_resolved(self, tmp_path):
		cfg = Config.from_dict(self.settings(hyb_folders=["s3://bucket/data", "rel"]), base_dir=str(tmp_path),
		                       check_files=False)
		assert cfg.paths.hyb_folders == ["s3://bucket/data", str(tmp_path / "rel")]

	def test_only_hyb_folders_can_be_urls(self):
		with pytest.raises(ConfigError, match="psf_file must be a local path"):
			Config.from_dict(self.settings(psf_file="s3://bucket/psf.npy"), check_files=False)

	def test_storage_options_must_be_a_table(self):
		with pytest.raises(ConfigError, match="storage_options must be a table"):
			Config.from_dict(self.settings(storage_options="lab"), check_files=False)

	def test_missing_remote_folder_is_reported(self, tmp_path, memory_root):
		(tmp_path / "psf.npy").write_bytes(b"")
		(tmp_path / "flat_0.npy").write_bytes(b"")
		data = self.settings(psf_file=str(tmp_path / "psf.npy"), flat_field_tag=str(tmp_path / "flat_"),
		                     hyb_folders=[f"{memory_root}/nothing"])
		with pytest.raises(ConfigError, match="does not exist"):
			Config.from_dict(data, check_files=True)

	def test_option_values_are_not_saved(self):
		cfg = Config.from_dict(self.settings(storage_options={"key": "AKIA-SECRET", "secret": "s3cr3t"}),
		                       check_files=False)
		rows = namespace_to_array(cfg)
		text = repr(rows)
		assert "AKIA-SECRET" not in text and "s3cr3t" not in text
		assert ("paths", "storage_options", "['key', 'secret']") in rows

	def test_to_toml_round_trip(self):
		cfg = Config.from_dict(self.settings(storage_options={"profile": "lab", "anon": False}), check_files=False)
		assert "storage_options = {profile = 'lab', anon = false}" in cfg.to_toml()


@pytest.mark.skipif(not os.environ.get("MERMAKE_S3_TEST_URL"), reason="set MERMAKE_S3_TEST_URL to test real object storage")
class TestLive:
	def test_list_and_read_one_image(self):
		url = os.environ["MERMAKE_S3_TEST_URL"].rstrip("/")
		storage.configure(read_threads=16)
		rounds = [n for n, _, d in storage.listdir(url) if d]
		assert rounds
		ff = FolderFilter(f"{rounds[0]}:{rounds[0]}", REGEX, 0, 0)
		matches = ff.get_matches([url])
		path = next(iter(matches.values()))[0]
		image = read_im(path)
		assert image.ndim == 4
		plane = image[0, 0].compute()
		assert plane.shape == image.shape[-2:]
