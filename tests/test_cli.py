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
		assert args.gpu is None  # the environment is left alone unless a gpu is asked for
		assert args.gpus is None and args.shard is None
		assert args.prefetch == 5
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


class TestShardOption:
	@pytest.mark.parametrize("text, expected", [("0/1", (0, 1)), ("1/4", (1, 4)), ("3/4", (3, 4))])
	def test_parses(self, toml_file, text, expected):
		args = cli.build_parser("mermake").parse_args(["--shard", text, toml_file])
		assert args.shard == expected

	@pytest.mark.parametrize("text", ["4/4", "5/4", "0/0", "-1/2", "1", "a/b", "1/2/3", ""])
	def test_rejects(self, toml_file, text, capsys):
		with pytest.raises(SystemExit):
			cli.build_parser("mermake").parse_args([f"--shard={text}", toml_file])
		assert "I/N" in capsys.readouterr().err

	@pytest.mark.parametrize("text", ["0", "-2", "x"])
	def test_prefetch_must_be_positive(self, toml_file, text):
		with pytest.raises(SystemExit):
			cli.build_parser("mermake").parse_args(["--prefetch", text, toml_file])


def test_dry_run_reports_the_shard(toml_file, tmp_path, capsys):
	for fov in range(5):
		(tmp_path / "exp" / "H1_X_set1" / f"Conv_zscan__{fov:03d}.zarr").mkdir(parents=True)
	cli.main(["--dry-run", "--shard", "1/2", toml_file])
	out = capsys.readouterr().out
	assert "shard 1/2: 2 of 5 fov groups" in out  # fovs 1 and 3
	assert "2 images" in out


class TestDeviceSelection:
	@pytest.fixture
	def stubs(self, monkeypatch):
		"""Stand-ins for the modules that need cupy, and the arguments the pipeline receives."""
		import types
		seen = []
		runtime = types.ModuleType("mermake.gpu.runtime")
		runtime.check_cuda = lambda: None
		pipeline = types.ModuleType("mermake.pipeline")
		pipeline.run = seen.append
		monkeypatch.setitem(sys.modules, "mermake.gpu.runtime", runtime)
		monkeypatch.setitem(sys.modules, "mermake.pipeline", pipeline)
		return seen

	def test_environment_is_kept_without_gpu_option(self, toml_file, monkeypatch, stubs):
		import os
		monkeypatch.setattr(os, "environ", {"CUDA_VISIBLE_DEVICES": "2,3"})
		cli.main([toml_file])
		assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
		assert len(stubs) == 1

	def test_gpu_counts_among_the_visible_devices(self, toml_file, monkeypatch, stubs):
		import os
		monkeypatch.setattr(os, "environ", {"CUDA_VISIBLE_DEVICES": "2,3"})
		cli.main(["--gpu", "1", toml_file])
		assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

	def test_gpu_is_the_physical_number_without_restriction(self, toml_file, monkeypatch, stubs):
		import os
		monkeypatch.setattr(os, "environ", {})
		cli.main(["--gpu", "1", toml_file])
		assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

	def test_gpu_beyond_the_visible_devices(self, toml_file, monkeypatch, stubs, capsys):
		import os
		monkeypatch.setattr(os, "environ", {"CUDA_VISIBLE_DEVICES": "2"})
		with pytest.raises(SystemExit):
			cli.main(["--gpu", "1", toml_file])
		assert "only 1 are visible" in capsys.readouterr().err
		assert not stubs


class TestGpusOption:
	def test_cannot_be_combined_with_gpu(self, toml_file, capsys):
		with pytest.raises(SystemExit):
			cli.main(["--gpus", "0,1", "--gpu", "0", toml_file])
		assert "cannot be combined" in capsys.readouterr().err

	def test_bad_value(self, toml_file, capsys):
		with pytest.raises(SystemExit):
			cli.main(["--gpus", "zero", toml_file])
		assert "--gpus" in capsys.readouterr().err

	def test_starts_one_worker_per_gpu(self, toml_file, monkeypatch):
		import os
		from mermake import launch
		monkeypatch.setattr(os, "environ", {"CUDA_VISIBLE_DEVICES": "4,5,6"})
		got = []
		monkeypatch.setattr(launch, "run_workers", lambda jobs: got.extend(jobs) or 7)
		assert cli.main(["--gpus", "all", "--prefetch", "2", toml_file]) == 7  # the exit status
		assert [env["CUDA_VISIBLE_DEVICES"] for _, env in got] == ["4", "5", "6"]
		for k, (cmd, _) in enumerate(got):
			assert cmd[cmd.index("--shard") + 1] == f"{k}/3"
			assert "--gpus" not in cmd and toml_file in cmd and "--prefetch" in cmd


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
