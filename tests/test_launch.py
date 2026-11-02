"""CPU-only tests for choosing GPUs and running one worker per GPU."""
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from mermake import launch
from mermake.io.naming import shard_keys


class TestResolveDevice:
	def test_physical_number_without_restriction(self):
		assert launch.resolve_device(2, {}) == "2"

	def test_counts_among_the_visible_devices(self):
		assert launch.resolve_device(0, {"CUDA_VISIBLE_DEVICES": "2,3"}) == "2"
		assert launch.resolve_device(1, {"CUDA_VISIBLE_DEVICES": "2,3"}) == "3"

	def test_uuids_are_kept(self):
		assert launch.resolve_device(1, {"CUDA_VISIBLE_DEVICES": "GPU-aaa, GPU-bbb"}) == "GPU-bbb"

	@pytest.mark.parametrize("index", [2, -1])
	def test_beyond_the_visible_devices(self, index):
		with pytest.raises(ValueError, match="only 2 are visible"):
			launch.resolve_device(index, {"CUDA_VISIBLE_DEVICES": "2,3"})

	def test_nothing_visible(self):
		with pytest.raises(ValueError, match="only 0 are visible"):
			launch.resolve_device(0, {"CUDA_VISIBLE_DEVICES": ""})


class TestParseGpus:
	def test_all_uses_the_visible_devices(self):
		assert launch.parse_gpus("all", {"CUDA_VISIBLE_DEVICES": "4,5"}) == ["4", "5"]

	def test_all_asks_the_machine_when_nothing_is_set(self):
		assert launch.parse_gpus("all", {}, query=lambda: ["0", "1", "2"]) == ["0", "1", "2"]

	def test_all_without_any_gpu(self):
		with pytest.raises(ValueError, match="no GPU"):
			launch.parse_gpus("all", {}, query=lambda: [])

	def test_list(self):
		assert launch.parse_gpus("0,2", {}) == ["0", "2"]
		assert launch.parse_gpus("1,0", {"CUDA_VISIBLE_DEVICES": "6,7"}) == ["7", "6"]

	def test_a_gpu_can_be_repeated(self):
		assert launch.parse_gpus("0,0", {}) == ["0", "0"]

	@pytest.mark.parametrize("spec", ["", "a", "0,,1", "0;1"])
	def test_malformed(self, spec):
		with pytest.raises(ValueError, match="--gpus"):
			launch.parse_gpus(spec, {})

	def test_beyond_the_visible_devices(self):
		with pytest.raises(ValueError, match="only 1 are visible"):
			launch.parse_gpus("0,1", {"CUDA_VISIBLE_DEVICES": "3"})


def test_strip_option():
	argv = ["--gpus", "0,1", "s.toml", "--prefetch=2", "--shard=1/2"]
	assert launch.strip_option(argv, "--gpus") == ["s.toml", "--prefetch=2", "--shard=1/2"]
	assert launch.strip_option(argv, "--shard") == ["--gpus", "0,1", "s.toml", "--prefetch=2"]
	assert launch.strip_option(["--gpu", "1"], "--gpus") == ["--gpu", "1"]


class TestWorkerJobs:
	def test_commands_and_environment(self):
		jobs = launch.worker_jobs(["--gpus", "all", "s.toml"], ["4", "5"], environ={"HOME": "/h"}, cpus=16)
		assert len(jobs) == 2
		for k, (cmd, env) in enumerate(jobs):
			assert cmd[:3] == [sys.executable, "-m", "mermake"]
			assert cmd[3:] == ["s.toml", "--shard", f"{k}/2"]
			assert env["CUDA_VISIBLE_DEVICES"] == str(4 + k)
			assert env["DASK_NUM_WORKERS"] == "8" and env["HOME"] == "/h"

	def test_the_cores_are_split_between_the_workers(self):
		jobs = launch.worker_jobs(["s.toml"], ["0", "1", "2"], environ={}, cpus=2)
		assert jobs[0][1]["DASK_NUM_WORKERS"] == "1"

	def test_a_set_thread_count_is_kept(self):
		jobs = launch.worker_jobs(["s.toml"], ["0"], environ={"DASK_NUM_WORKERS": "3"}, cpus=16)
		assert jobs[0][1]["DASK_NUM_WORKERS"] == "3"

	def test_the_machines_share_is_split_between_its_workers(self):
		# machine 1 of 3 with 2 GPUs takes shares 2 and 3 of 6
		jobs = launch.worker_jobs(["--shard", "1/3", "s.toml"], ["0", "1"], shard=(1, 3), environ={}, cpus=4)
		assert [cmd[cmd.index("--shard") + 1] for cmd, _ in jobs] == ["2/6", "3/6"]
		assert sum(cmd.count("--shard") for cmd, _ in jobs) == 2  # the original option was removed

	def test_machines_and_workers_cover_every_field_of_view_once(self):
		keys = [(1, i) for i in range(23)]
		seen = []
		for machine in range(3):
			for cmd, _ in launch.worker_jobs(["s.toml"], ["0", "1"], shard=(machine, 3), environ={}, cpus=4):
				index, count = map(int, cmd[cmd.index("--shard") + 1].split("/"))
				seen.extend(shard_keys(keys, index, count))
		assert sorted(seen) == keys


def run_launcher(codes):
	"""Run workers that exit with ``codes`` through ``run_workers``, in this process."""
	jobs = [([sys.executable, "-c", f"import sys; sys.exit({c})"], dict(os.environ)) for c in codes]
	return launch.run_workers(jobs)


class TestRunWorkers:
	def test_all_succeed(self):
		assert run_launcher([0, 0, 0]) == 0

	def test_a_failure_gives_its_status(self):
		assert run_launcher([0, 3, 4]) == 3

	def test_a_failure_does_not_stop_the_others(self, tmp_path):
		marker = tmp_path / "done"
		slow = f"import time, pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('x')"
		jobs = [([sys.executable, "-c", "import sys; sys.exit(2)"], dict(os.environ)),
		        ([sys.executable, "-c", slow], dict(os.environ))]
		assert launch.run_workers(jobs) == 2
		assert marker.exists()

	def test_a_killed_worker_is_a_failure(self):
		jobs = [([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"], dict(os.environ))]
		assert launch.run_workers(jobs) == 128 + signal.SIGKILL

	def test_sigterm_is_passed_to_the_workers(self, tmp_path):
		"""The launcher gets SIGTERM (scancel, the time limit) and its workers must hear about it."""
		ready, got = tmp_path / "ready", tmp_path / "got_sigterm"
		worker = textwrap.dedent(f"""
			import pathlib, signal, sys, time
			def handler(signum, frame):
				pathlib.Path({str(got)!r}).write_text('x')
				sys.exit(0)
			signal.signal(signal.SIGTERM, handler)
			pathlib.Path({str(ready)!r}).write_text('x')
			time.sleep(60)
		""")
		code = textwrap.dedent(f"""
			import sys
			from mermake import launch
			sys.exit(launch.run_workers([([sys.executable, '-c', {worker!r}], {{}})]))
		""")
		proc = subprocess.Popen([sys.executable, "-c", code])
		try:
			deadline = time.time() + 30
			while not ready.exists() and time.time() < deadline:
				time.sleep(0.05)
			assert ready.exists(), "the worker did not start"
			proc.send_signal(signal.SIGTERM)
			assert proc.wait(timeout=30) == 0
			assert got.exists()
		finally:
			proc.kill()
