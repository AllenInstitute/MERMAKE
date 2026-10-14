"""CPU-only tests for command line parsing and settings loading."""
import argparse
import sys

import pytest

from mermake import cli

SETTINGS = """
[paths]
psf_file = 'psf.npy'
flat_field_tag = 'flat_'
hyb_folders = ['exp']
hyb_range = 'H1_X_set1:H2_X_set1'
output_folder = 'out'

[hybs]
patch_size = 400
"""


@pytest.fixture
def toml_file(tmp_path):
	(tmp_path / "psf.npy").write_bytes(b"")
	(tmp_path / "flat_0.npz").write_bytes(b"")
	(tmp_path / "exp").mkdir()
	path = tmp_path / "settings.toml"
	path.write_text(SETTINGS)
	return str(path)


class TestIsValidFile:
	def test_loads_toml(self, toml_file):
		settings = cli.is_valid_file(toml_file)
		assert settings.paths.psf_file.endswith("psf.npy")
		assert settings.hybs.patch_size == 400
		assert settings.dapi.patch_size == 500  # default

	def test_missing_file(self, tmp_path):
		with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
			cli.is_valid_file(str(tmp_path / "nope.toml"))

	def test_invalid_toml(self, tmp_path):
		bad = tmp_path / "bad.toml"
		bad.write_text("this is = = not toml")
		with pytest.raises(argparse.ArgumentTypeError, match="not valid TOML"):
			cli.is_valid_file(str(bad))


class TestParser:
	def test_defaults(self, toml_file):
		args = cli.build_parser("mermake").parse_args([toml_file])
		assert args.gpu == 0
		assert args.check is False
		assert args.settings.paths.hyb_folders[0].endswith("exp")

	def test_options(self, toml_file):
		args = cli.build_parser("mermake").parse_args(["-g", "2", "-c", toml_file])
		assert args.gpu == 2
		assert args.check is True

	def test_missing_settings_prints_example(self, capsys):
		with pytest.raises(SystemExit) as exc:
			cli.main([])
		assert exc.value.code == 2
		err = capsys.readouterr().err
		assert "settings.toml" in err
		assert "[paths]" in err and "psf_file" in err

	def test_bad_settings_path(self, tmp_path, capsys):
		with pytest.raises(SystemExit):
			cli.build_parser("mermake").parse_args([str(tmp_path / "nope.toml")])
		assert "does not exist" in capsys.readouterr().err

	def test_usage_uses_prog_name(self):
		assert cli.build_parser("mermake").usage.startswith("mermake ")


def test_invalid_settings_reported(tmp_path):
	bad = tmp_path / "bad.toml"
	bad.write_text("[paths]\npsf_file = 'x'\n")
	with pytest.raises(argparse.ArgumentTypeError, match="missing required key"):
		cli.is_valid_file(str(bad))


def test_print_example(capsys):
	cli.main(["--print-example"])
	assert "[paths]" in capsys.readouterr().out


def test_print_config_needs_no_gpu(toml_file, capsys):
	cli.main(["--print-config", toml_file])
	out = capsys.readouterr().out
	assert "patch_size = 400" in out and "[dapi]" in out


def test_dry_run_lists_images(toml_file, tmp_path, capsys):
	zarr = tmp_path / "exp" / "H1_X_set1" / "Conv_zscan__003.zarr"
	zarr.mkdir(parents=True)
	cli.main(["--dry-run", toml_file])
	out = capsys.readouterr().out
	assert "1 images" in out


def test_importing_cli_does_not_import_cupy():
	# CUDA_VISIBLE_DEVICES has to be set before cupy is imported, so the
	# cli module must stay free of it.
	import subprocess
	code = "import sys, mermake.cli; sys.exit('cupy' in sys.modules)"
	assert subprocess.run([sys.executable, "-c", code]).returncode == 0
