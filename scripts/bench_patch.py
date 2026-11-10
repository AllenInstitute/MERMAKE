"""Time the operations of one deconvolved patch and some alternatives, on the shapes of a real run.

    uv run --no-sync python scripts/bench_patch.py [--depth 40] [--patch-size 500] [--patch-overlap 89]

Everything is measured on synthetic data of the shapes the pipeline uses, so no images are needed.
Each line is the median time of the operation in milliseconds (CUDA events, after warm-up). A call that
makes the host wait with the GPU idle shows up here too, as with ``cp.shares_memory``.
The alternatives that change numbers are compared with the current code, and the size of the
difference is printed next to the time.
"""
import argparse
import sys

import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

from mermake.gpu import blur
from mermake.gpu.fill import reflect


def ms(fn, *args, repeat=15):
	"""Median GPU time of ``fn(*args)`` in ms."""
	result = benchmark(fn, args, n_repeat=repeat, n_warmup=3)
	return float(np.median(result.gpu_times)) * 1000


def row(name, t, note=""):
	print(f"  {name:58s}{t:9.2f} ms  {note}")


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--depth", type=int, default=40, help="z planes of a channel [40]")
	parser.add_argument("--patch-size", type=int, default=500)
	parser.add_argument("--patch-overlap", type=int, default=89)
	parser.add_argument("--blur-radius", type=int, default=30, help="[hybs] blur_radius, dapi uses 50 [30]")
	args = parser.parse_args(argv)

	D, P = args.depth, args.patch_size + 2 * args.patch_overlap
	zpad = D - 1
	Z = D + 2 * zpad
	print(f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, depth {D}, patch {P} x {P}, "
	      f"padded stack {Z} x {P} x {P} (hyb)\n")

	rng = cp.random.default_rng(0)
	raw = rng.integers(0, 4000, (D, P, P), dtype=cp.uint16)
	flat = rng.uniform(0.9, 1.1, (P, P)).astype(cp.float32)
	pad = cp.zeros((Z, P, P), dtype=cp.float32)
	fft = cp.empty(pad.shape, dtype=cp.complex64)
	psf = (rng.standard_normal(pad.shape) + 1j * rng.standard_normal(pad.shape)).astype(cp.complex64)
	res = cp.empty((D, P, P), dtype=cp.float32)
	buf, tem = cp.empty_like(res), cp.empty_like(res)
	mask = (slice(zpad, -zpad), slice(None), slice(None))

	print("the operations of one interior patch, as in Deconvolver.patch_wise:")
	row("patch_pad[:] = 0", ms(lambda: pad.__setitem__(slice(None), 0.0)))
	row("copy the patch in (uint16 -> float32)", ms(lambda: pad.__setitem__((slice(zpad, zpad + D),), raw)))
	row("divide by the flat field", ms(lambda: cp.divide(pad[mask], flat, out=pad[mask])))
	row("reflect in z, both sides (2 calls)", ms(lambda: (reflect(pad, zpad, mode='in', axis=0, out=pad),
	                                                       reflect(pad, zpad + D - 1, mode='out', axis=0, out=pad))))
	row("reflect in x (an edge patch has 2 more of these)", ms(lambda: reflect(pad, 89, mode='in', axis=1, out=pad)))
	t_fwd = ms(lambda: fft.__setitem__(slice(None), cp.fft.fftn(pad)))
	row("fftn, copied into patch_fft", t_fwd)
	row("multiply by the PSF spectrum", ms(lambda: cp.multiply(fft, psf, out=fft)))
	t_inv = ms(lambda: res.__setitem__(slice(None), cp.fft.ifftn(fft)[zpad:-zpad].real))
	row("ifftn, real part of the middle planes", t_inv)
	size = args.blur_radius
	t_blur = ms(lambda: (blur.box_1d(res, size, axis=1, out=buf), blur.box_1d(buf, size, axis=2, out=tem)))
	row(f"box blur x then y, box {size}", t_blur)
	row("subtract the blur", ms(lambda: cp.subtract(res, tem, out=res)))
	row("multiply by the flat field again", ms(lambda: cp.multiply(pad[mask], flat, out=pad[mask])))
	print()

	print("what box_1d does around its kernel (this was 40 ms until the overlap test was made cheap):")
	def kernel(axis):
		blur.box_1d_kernel(((res.size + 255) // 256,), (256,), (res, buf, D, P, P, size // 2, axis))
	row("the kernel alone, x", ms(lambda: kernel(1)))
	row("cp.shares_memory(a, b), host time with the GPU idle", ms(lambda: cp.shares_memory(res, buf)))
	row("cp.may_share_memory(a, b)", ms(lambda: cp.may_share_memory(res, buf)))
	print()

	print("FFT of the padded stack (forward + inverse, c2c, like the pipeline), by shape:")
	shapes = [(Z, P, P), (Z, 672, 672), (120, P, P), (120, 672, 672), (128, 640, 640), (Z, 640, 640),
	          (108, 648, 648), (112, 700, 700), (120, 720, 720)]
	base = None
	for z, y, x in shapes:
		a = rng.standard_normal((z, y, x)).astype(cp.float32)
		t = ms(lambda: cp.fft.ifftn(cp.fft.fftn(a)))
		if base is None:
			base = t / (z * y * x)
		note = f"{t / (z * y * x) / base:.2f}x the time per voxel of the current shape"
		row(f"{z} x {y} x {x}", t, note)
		del a
	print()

	print("real-to-complex FFT (rfftn/irfftn), the same stack:")
	a = pad
	half = cp.fft.rfftn(a)
	t_r = ms(lambda: cp.fft.irfftn(cp.fft.rfftn(a), s=a.shape))
	row(f"{Z} x {P} x {P}  rfftn + irfftn", t_r, f"{ms(lambda: cp.fft.ifftn(cp.fft.fftn(a))) / t_r:.2f}x faster than c2c")
	print(f"  spectrum {half.shape} {half.dtype}, {half.nbytes / 2**20:.0f} MiB against {fft.nbytes / 2**20:.0f} MiB")
	# numerical difference of the whole deconvolution step with a real PSF filter
	x = rng.standard_normal((Z, P, P)).astype(cp.float32)
	psf_real = cp.fft.fftn(cp.asarray(rng.random((Z, P, P)).astype(cp.float32)))
	full = cp.fft.ifftn(cp.fft.fftn(x) * psf_real).real
	half_out = cp.fft.irfftn(cp.fft.rfftn(x) * psf_real[:, :, :P // 2 + 1], s=x.shape)
	err = float(cp.max(cp.abs(full - half_out)) / cp.max(cp.abs(full)))
	print(f"  largest difference of the filtered stack, relative to its largest value: {err:.2e}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
