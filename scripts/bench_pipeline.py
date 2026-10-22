"""Where does the time and the GPU memory of a mermake run go?

Runs the real pipeline functions on the first few images of a settings file, with timers around the
stages and around the GPU functions they call, and reports the times, the wait for the reader thread
and the peak GPU memory. Nothing is saved, so the output folder of the settings can be anything,
but it must not hold results (finished channels are skipped, not timed).

    uv run --no-sync python scripts/bench_pipeline.py settings.toml --images 9 --skip 1

The first ``--skip`` images are run but not counted, since they pay for compiling the kernels and
planning the FFTs. Use ``--fov-range`` of the settings to choose the images. The ``gpu ms`` columns
come from CUDA events, the ``host ms`` columns from the clock: for a function that only launches
kernels they differ by the time the host spent waiting or launching.
"""
import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from types import SimpleNamespace


class Stats:
	"""Calls, host seconds and GPU milliseconds per name."""
	def __init__(self):
		self.calls = defaultdict(int)
		self.host = defaultdict(float)
		self.events = defaultdict(list)
		self.gpu = defaultdict(float)

	def add(self, name, host, start=None, end=None):
		self.calls[name] += 1
		self.host[name] += host
		if start is not None:
			self.events[name].append((start, end))

	def resolve(self):
		"""Turn the recorded events into milliseconds. Call after the GPU has finished."""
		import cupy as cp
		for name, pairs in self.events.items():
			self.gpu[name] += sum(cp.cuda.get_elapsed_time(a, b) for a, b in pairs)
		self.events.clear()

	def reset(self):
		self.resolve()
		self.calls.clear()
		self.host.clear()
		self.gpu.clear()


def instrument(stats, module, attr, name, label=None):
	"""Replace ``module.attr`` with a function that times the original under ``name``.

	``label`` may map the arguments to a different name, to tell calls of one function apart.
	"""
	import cupy as cp
	original = getattr(module, attr)

	def timed(*args, **kwargs):
		key = label(*args, **kwargs) if label else name
		start, end = cp.cuda.Event(), cp.cuda.Event()
		t0 = time.perf_counter()
		start.record()
		try:
			return original(*args, **kwargs)
		finally:
			end.record()
			stats.add(key, time.perf_counter() - t0, start, end)
	setattr(module, attr, timed)
	return original


class MemorySampler(threading.Thread):
	"""Polls the free memory of the device to find the peak of what is in use, the memory pool,
	the FFT workspaces and the CUDA context included. Run it on a device nothing else uses."""
	def __init__(self, interval=0.002):
		super().__init__(daemon=True)
		import cupy as cp
		self.interval = interval
		self.stop = threading.Event()
		free, self.total = cp.cuda.runtime.memGetInfo()
		self.min_free = free

	def run(self):
		import cupy as cp
		while not self.stop.is_set():
			free, _ = cp.cuda.runtime.memGetInfo()
			self.min_free = min(self.min_free, free)
			time.sleep(self.interval)

	def peak_gib(self):
		"""Most device memory that was in use since the start or the last reset, in GiB."""
		return (self.total - self.min_free) / 1024**3

	def reset(self):
		import cupy as cp
		self.min_free, _ = cp.cuda.runtime.memGetInfo()


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("settings", help="settings TOML, with an output folder that holds no results")
	parser.add_argument("--images", type=int, default=9, help="images to run, all rounds of the first fov by default [9]")
	parser.add_argument("--skip", type=int, default=1, help="leading images that are run but not counted [1]")
	parser.add_argument("--prefetch", type=int, default=5)
	parser.add_argument("--gpu", type=int, default=None)
	args = parser.parse_args(argv)

	if args.gpu is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	from mermake.cli import is_valid_file
	args.settings = is_valid_file(args.settings)
	args.shard = None

	import cupy as cp
	import numpy as np
	import mermake.deconvolver
	import mermake.gpu.blur
	import mermake.pipeline as pl
	from mermake.io.queue import ImageQueue, Block
	from mermake.io.readers import load_flats, color_indices, dapi_index

	stats = Stats()
	# the functions of the hot path. The reflect is called by name from the deconvolver module.
	instrument(stats, mermake.deconvolver, "reflect", "reflect (padding)")
	instrument(stats, mermake.gpu.blur, "box_1d", "box blur")
	instrument(stats, cp.fft, "fftn", "fftn")
	instrument(stats, cp.fft, "ifftn", "ifftn")
	instrument(stats, pl, "find_local_maxima", "maxima (patch)",
	           label=lambda image, *a, **k: "maxima (whole image, dapi)"
	           if image.shape[1] * image.shape[2] > 700 * 700 else "maxima (patch)")
	instrument(stats, pl, "drift", "drift")

	pl.build_config(args)
	psfs = np.load(args.paths.psf_file, allow_pickle=True)
	sampler = None
	with ImageQueue(args, prefetch_count=args.prefetch) as queue:
		ncol, *chan_shape = queue.shape
		print(f"image {tuple(queue.shape)} {queue.dtype}, {queue.n_images} images, "
		      f"device {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
		buffer = cp.empty(chan_shape, dtype=cp.float32)
		chan = cp.empty(chan_shape, dtype=cp.uint16)
		flats = load_flats(shape=queue.shape, **vars(args.paths))
		deconvolver = pl.build_deconvolvers(psfs, chan_shape, args)
		saver = SimpleNamespace(submit=lambda *a, **k: None)
		ctx = SimpleNamespace(
			args=args, ncol=ncol, dapi=dapi_index(ncol, args.paths.dapi_channel),
			colors=color_indices(ncol, args.paths.dapi_channel), buffer=buffer, chan=chan, flats=flats,
			deconvolver=deconvolver, executor=saver, block=Block())

		sampler = MemorySampler()
		sampler.start()
		stage = defaultdict(float)
		n = 0
		wall0 = None
		images = iter(queue)
		for i in range(args.images):
			t0 = time.perf_counter()
			image = next(images, None)
			if image is None:
				break
			wait = time.perf_counter() - t0
			counted = i >= args.skip
			if i == args.skip:
				stats.reset()
				sampler.reset()
				wall0 = time.perf_counter()
				stage.clear()
			t = time.perf_counter()
			pl.process_dapi(image, queue, ctx)
			cp.cuda.Stream.null.synchronize()
			stage["dapi"] += time.perf_counter() - t
			t = time.perf_counter()
			for icol in range(len(ctx.colors)):
				pl.process_hyb_channel(image, icol, queue, ctx)
			cp.cuda.Stream.null.synchronize()
			stage["colors"] += time.perf_counter() - t
			t = time.perf_counter()
			pl.finish_block(image, ctx)
			cp.cuda.Stream.null.synchronize()
			stage["block/drift"] += time.perf_counter() - t
			if counted:
				stage["wait for the reader"] += wait
				n += 1
			print(f"image {i}{'' if counted else ' (not counted)'}: wait {wait:.2f} s, queue {queue.queue.qsize()} ready, "
			      f"pool {cp.get_default_memory_pool().total_bytes() / 1024**3:.2f} GiB", flush=True)
			del image
		total = time.perf_counter() - wall0 if wall0 else float("nan")
		stats.resolve()
		peak = sampler.peak_gib()
		sampler.stop.set()

	if n == 0:
		print("no image was counted, raise --images or lower --skip")
		return 1
	print(f"\n{n} images counted, {total / n:.2f} s per image (wall, including waiting for the reader)")
	print("\nper image, by stage:")
	for name, seconds in stage.items():
		print(f"  {name:28s}{seconds / n:8.2f} s  {100 * seconds / total:5.1f} %")
	print("\nper image, by function (a function may call another):")
	print(f"  {'':28s}{'calls':>7s}{'host ms':>10s}{'gpu ms':>10s}")
	for name in sorted(stats.calls, key=lambda k: -stats.host[k]):
		print(f"  {name:28s}{stats.calls[name] / n:7.0f}{1000 * stats.host[name] / n:10.0f}{stats.gpu[name] / n:10.0f}")
	print(f"\npeak GPU memory in use over the counted images: {peak:.2f} GiB of {sampler.total / 1024**3:.1f} GiB "
	      f"(includes the CUDA context, about 0.3 GiB)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
