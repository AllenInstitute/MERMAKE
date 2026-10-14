"""CUDA runtime helpers."""
import os

import cupy as cp


def kernel_source(name):
	"""Return the text of the CUDA source file ``kernels/<name>``."""
	path = os.path.join(os.path.dirname(__file__), "kernels", name)
	with open(path, "r") as f:
		return f.read()


def check_cuda():
	"""Raise a helpful error if CuPy cannot see a CUDA device.

	nvcc is not needed: CuPy compiles kernels with NVRTC, which needs the
	CUDA headers (``pip install cupy-cuda12x[ctk]``).
	"""
	try:
		n = cp.cuda.runtime.getDeviceCount()
	except Exception as e:
		raise RuntimeError(
			"CuPy could not find a usable CUDA device. Install the extra "
			"matching your CUDA version (e.g. `pip install mermake[cuda12]`). "
			"See https://docs.cupy.dev/en/stable/install.html"
		) from e
	if n < 1:
		raise RuntimeError("No CUDA device found.")


def profile():
	"""Print the shape and size of every cupy array that owns GPU memory, and the memory pool usage. For debugging.
	"""
	import gc
	mempool = cp.get_default_memory_pool()
	# Loop through all objects in the garbage collector
	for obj in gc.get_objects():
		if isinstance(obj, cp.ndarray):
			# Check if it's a view (not a direct memory allocation)
			if obj.base is not None:
				# Skip views as they do not allocate new memory
				continue
			print(f"CuPy array with shape {obj.shape} and dtype {obj.dtype}")
			print(f"Memory usage: {obj.nbytes / 1024**2:.2f} MB")  # Convert to MB
	print(f"Used memory after: {mempool.used_bytes() / 1024**2:.2f} MB")
