"""CPU-only tests for config helpers, XML metadata and point helpers."""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from mermake import cli
from mermake.config import Config, ConfigError, DEFAULTS, Params, Paths, namespace_to_array, reference_rows, tomllib
from mermake.io.metadata import get_xml_field, read_xml
from mermake.points import estimate_step_size, points_to_coords


class TestNamespaceToArray:
	def test_flattens_with_dotted_prefix(self):
		rows = namespace_to_array({"paths": {"psf": "a.npy"}, "gpu": 0})
		assert ("paths", "psf", "a.npy") in rows
		assert ("", "gpu", "0") in rows

	def test_accepts_namespaces(self):
		ns = SimpleNamespace(hybs=SimpleNamespace(patch_size=500))
		assert namespace_to_array(ns) == [("hybs", "patch_size", "500")]

	def test_deeper_nesting(self):
		rows = namespace_to_array({"a": {"b": {"c": 1}}})
		assert rows == [("a.b", "c", "1")]


class TestXml:
	XML = '<?xml version="1.0"?><root><z_offsets type="string">0:-15:-0.4:3</z_offsets></root>'

	def test_get_field(self, tmp_path):
		path = tmp_path / "scan.xml"
		path.write_text(self.XML, encoding="ISO-8859-1")
		assert get_xml_field(str(path), "z_offsets") == "0:-15:-0.4:3"

	def test_read_xml_root(self, tmp_path):
		path = tmp_path / "scan.xml"
		path.write_text(self.XML, encoding="ISO-8859-1")
		assert read_xml(str(path)).tag == "root"


class TestPoints:
	def grid(self, step=10.0, n=4):
		xs, ys = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
		return np.column_stack([xs.ravel(), ys.ravel()])

	def test_estimate_step_size(self):
		assert estimate_step_size(self.grid(10.0)) == 10.0

	def test_points_to_coords(self):
		pts = self.grid(10.0) + 100.0
		coords = points_to_coords(pts)
		assert coords.min() == 0 and coords.max() == 3
		assert len({tuple(c) for c in coords}) == 16

	def test_points_to_coords_does_not_modify_input(self):
		pts = self.grid(10.0) + 5.0
		before = pts.copy()
		points_to_coords(pts)
		np.testing.assert_array_equal(pts, before)


# ---- Config validation -------------------------------------------------------------

REQUIRED = dict(psf_file="psf.npy", flat_field_tag="flat_", hyb_folders=["exp"],
                hyb_range="H1_X_set1:H2_X_set1", output_folder="out")


def load(paths=None, **sections):
	return Config.from_dict({"paths": paths or REQUIRED, **sections}, check_files=False)


class TestDefaults:
	def test_only_required_keys_needed(self):
		cfg = load()
		assert cfg.hybs.threshold == 3600 and cfg.dapi.threshold == 3.0
		assert cfg.paths.fov_range is None
		assert cfg.paths.drift_save == "drift_{fov}--_set{iset}.pkl"

	def test_override_one_value(self):
		cfg = load(hybs={"threshold": 4000})
		assert cfg.hybs.threshold == 4000
		assert cfg.hybs.beta == DEFAULTS["hybs"]["beta"]
		assert cfg.dapi.threshold == 3.0

	def test_numbers_are_coerced(self):
		cfg = load(hybs={"sigmaZ": 2})
		assert isinstance(cfg.hybs.sigmaZ, float) and isinstance(cfg.dapi.sigmaZ, float)

	def test_vars_gives_kwargs_for_gpu_functions(self):
		assert set(vars(load().hybs)) == {f.name for f in fields(Params)}


class TestValidation:
	def error(self, **kwargs):
		with pytest.raises(ConfigError) as exc:
			load(**kwargs)
		return str(exc.value)

	def test_missing_required(self):
		with pytest.raises(ConfigError, match="missing required key 'psf_file'"):
			Config.from_dict({"paths": {}}, check_files=False)

	def test_misspelled_key_suggests_fix(self):
		assert "did you mean 'threshold'" in self.error(hybs={"threshhold": 5})

	def test_unknown_section(self):
		assert "unknown section 'hyb'" in self.error(hyb={})

	def test_retired_keys(self):
		assert "renamed to 'patch_size'" in self.error(hybs={"tile_size": 5})
		assert "renamed to 'patch_overlap'" in self.error(dapi={"overlap": 5})
		assert "no longer used" in self.error(paths=dict(REQUIRED, codebook="x"))

	def test_wrong_type(self):
		assert "patch_size must be an integer" in self.error(hybs={"patch_size": "big"})
		assert "patch_size must be an integer" in self.error(hybs={"patch_size": 1.5})
		assert "hyb_folders must be a list of strings" in self.error(paths=dict(REQUIRED, hyb_folders="exp"))

	def test_value_ranges(self):
		assert "delta must be between 1 and 5" in self.error(hybs={"delta": 6})
		assert "beta must be positive" in self.error(dapi={"beta": 0})

	def test_ranges_and_regex(self):
		assert "hyb_range" in self.error(paths=dict(REQUIRED, hyb_range="H1_X_set1"))
		assert "does not match regex" in self.error(paths=dict(REQUIRED, hyb_range="foo:bar"))
		assert "fov_range" in self.error(paths=dict(REQUIRED, fov_range="5:1"))
		assert "regex" in self.error(paths=dict(REQUIRED, regex="(("))
		assert "5 capture groups" in self.error(paths=dict(REQUIRED, regex="(a)"))

	def test_all_errors_reported_together(self):
		msg = self.error(hybs={"threshhold": 5}, dapi={"beta": -1, "delta": 9})
		assert msg.count("\n") == 2

	def test_missing_files_reported(self, tmp_path):
		with pytest.raises(ConfigError) as exc:
			Config.from_dict({"paths": dict(REQUIRED)}, base_dir=str(tmp_path))
		msg = str(exc.value)
		assert "psf_file" in msg and "flat_field_tag" in msg and "hyb_folders" in msg


class TestPaths:
	def test_relative_paths_resolve_against_settings_dir(self, tmp_path):
		(tmp_path / "settings.toml").write_text(
			"[paths]\npsf_file='p.npy'\nflat_field_tag='f_'\nhyb_folders=['e']\n"
			"hyb_range='H1_X_set1:H2_X_set1'\noutput_folder='o'\n")
		cfg = Config.from_file(tmp_path / "settings.toml", check_files=False)
		assert cfg.paths.psf_file == str(tmp_path / "p.npy")
		assert cfg.paths.hyb_folders == [str(tmp_path / "e")]
		assert cfg.paths.output_folder == str(tmp_path / "o")

	def test_absolute_paths_kept(self):
		cfg = Config.from_dict({"paths": dict(REQUIRED, psf_file="/data/psf.npy")},
		                       base_dir="/somewhere", check_files=False)
		assert cfg.paths.psf_file == "/data/psf.npy"


class TestRoundTrip:
	def test_to_toml_reloads_equal(self):
		cfg = load(hybs={"threshold": 4000}, paths=dict(REQUIRED, fov_range="0:3"))
		again = Config.from_dict(tomllib.loads(cfg.to_toml()), check_files=False)
		assert again == cfg

	def test_default_regex_survives(self):
		again = Config.from_dict(tomllib.loads(load().to_toml()), check_files=False)
		assert again.paths.regex == load().paths.regex == r'([A-z]+)(\d+)_(.+)_set(\d+)(.*)'


class TestDocs:
	def test_every_field_is_documented(self):
		for cls in (Paths, Params):
			for f in fields(cls):
				assert f.metadata.get("doc", "").strip(), f"{cls.__name__}.{f.name} has no doc"

	def test_every_params_field_has_defaults_for_both_sections(self):
		names = {f.name for f in fields(Params)}
		assert set(DEFAULTS["hybs"]) == names and set(DEFAULTS["dapi"]) == names

	def test_reference_rows_cover_all_settings(self):
		keys = {(r["section"], r["key"]) for r in reference_rows()}
		assert ("paths", "psf_file") in keys and ("dapi", "sigmaXY") in keys
		required = {r["key"] for r in reference_rows() if r["required"] and r["section"] == "paths"}
		assert required == {"psf_file", "flat_field_tag", "hyb_folders", "hyb_range", "output_folder"}

	def test_packaged_example_is_valid(self):
		cfg = Config.from_dict(tomllib.loads(cli.example_text()), check_files=False)
		assert cfg.paths.hyb_range == "H1_BICAN_set1:H16_BICAN_set1"
