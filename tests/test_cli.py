"""CPU-only tests for command line parsing and settings loading."""
import argparse
import sys

import pytest

from mermake import cli

SETTINGS = """
[paths]
psf_file = 'psf.npy'
hyb_folders = ['exp']

[hybs]
patch_size = 500
"""


@pytest.fixture
def toml_file(tmp_path):
	path = tmp_path / "settings.toml"
	path.write_text(SETTINGS)
	return str(path)


class TestIsValidFile:
	def test_loads_toml(self, toml_file):
		settings = cli.is_valid_file(toml_file)
		assert settings["paths"]["psf_file"] == "psf.npy"
		assert settings["hybs"]["patch_size"] == 500

	def test_missing_file(self, tmp_path):
		with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
			cli.is_valid_file(str(tmp_path / "nope.toml"))

	def test_invalid_toml(self, tmp_path):
		bad = tmp_path / "bad.toml"
		bad.write_text("this is = = not toml")
		with pytest.raises(argparse.ArgumentTypeError, match="Error loading TOML"):
			cli.is_valid_file(str(bad))


class TestParser:
	def test_defaults(self, toml_file):
		args = cli.build_parser("mermake").parse_args([toml_file])
		assert args.gpu == 0
		assert args.check is False
		assert args.settings["paths"]["hyb_folders"] == ["exp"]

	def test_options(self, toml_file):
		args = cli.build_parser("mermake").parse_args(["-g", "2", "-c", toml_file])
		assert args.gpu == 2
		assert args.check is True

	def test_missing_settings_prints_example(self, capsys):
		with pytest.raises(SystemExit) as exc:
			cli.build_parser("mermake").parse_args([])
		assert exc.value.code == 2
		err = capsys.readouterr().err
		assert "settings.toml" in err
		assert "[paths]" in err and "[hybs]" in err and "[dapi]" in err

	def test_bad_settings_path(self, tmp_path, capsys):
		with pytest.raises(SystemExit):
			cli.build_parser("mermake").parse_args([str(tmp_path / "nope.toml")])
		assert "does not exist" in capsys.readouterr().err

	def test_usage_uses_prog_name(self):
		assert cli.build_parser("mermake").usage.startswith("mermake ")


def test_example_config_text_is_valid_toml():
	"""The example printed to users must itself be loadable."""
	tomllib = cli.tomllib
	settings = tomllib.loads(cli.toml_text)
	assert {"paths", "hybs", "dapi"} <= set(settings)


def test_importing_cli_does_not_import_cupy():
	# CUDA_VISIBLE_DEVICES has to be set before cupy is imported, so the
	# cli module must stay free of it.
	import subprocess
	code = "import sys, mermake.cli; sys.exit('cupy' in sys.modules)"
	assert subprocess.run([sys.executable, "-c", code]).returncode == 0
