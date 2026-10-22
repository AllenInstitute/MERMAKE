"""Local maxima detection and sub-voxel refinement (GPU)."""
import cupy as cp
from .gpu.runtime import kernel_source

kernel_code = kernel_source("maxima.cu")

# Define the kernels separately
local_maxima_count_kernel = cp.RawKernel(kernel_code, "local_maxima_count")
local_maxima_kernel = cp.RawKernel(kernel_code, "local_maxima")
#delta_fit_kernel = cp.RawKernel(kernel_code, "delta_fit")
delta_fit_cross_corr_kernel = cp.RawKernel(kernel_code, "delta_fit_cross_corr")
# the same kernel reading a uint16 raw image directly
delta_fit_cross_corr_u16_kernel = cp.RawKernel(kernel_code, "delta_fit_cross_corr", options=("-DRAW_T=unsigned short",))

def find_local_maxima(image, threshold=None, delta=1, delta_fit=0, raw=None, sigmaZ=1, sigmaXY=1.5, **kwargs):
	"""Find local intensity maxima in a 3D image and refine them to sub-voxel positions.

	A voxel is a maximum if it is brighter than ``threshold`` and no voxel within
	``delta`` of it is brighter. Its position is then refined to the intensity-weighted
	centroid of the ``delta_fit`` neighborhood (after subtracting the neighborhood minimum),
	and two correlation scores are computed against a Gaussian template.

	Parameters
	----------
	image : cupy.ndarray
		(z, y, x) float32 image to search, usually the deconvolved image.
	threshold : float
		Minimum intensity of a maximum, in the units of ``image``.
	delta : int
		Radius in voxels of the spherical neighborhood a maximum has to win. At most 5.
	delta_fit : int
		Radius in voxels of the spherical neighborhood used for the centroid and the scores.
		At most 5: the kernel holds 600 points, and a larger neighborhood would be truncated.
		With 0 the position stays at the integer voxel and the scores are NaN.
	raw : cupy.ndarray
		(z, y, x) raw image with the same shape as ``image``, used for the scores ``a`` and
		``habs``. Required when any maximum is found. float32 and uint16 are read as they are,
		anything else is converted to float32 first.
	sigmaZ, sigmaXY : float
		Width in voxels, in z and in xy, of the Gaussian template of the scores ``a`` and ``hn``.
		They do not change the position.
	**kwargs
		Ignored, so a whole settings section can be passed with ``**vars(section)``.

	Returns
	-------
	cupy.ndarray
		float32 array with shape (n, 8) on the GPU. Columns: z, x, y of the refined
		position (x indexes the second image axis), ``background`` (the minimum of ``image`` in
		the neighborhood), ``a`` (normalized cross-correlation of the Gaussian template with
		``raw`` in the neighborhood), ``habs`` (``raw`` at the maximum voxel), ``hn`` (the same
		correlation with ``image``) and ``h`` (``image`` at the maximum voxel).
	"""
	# Ensure the image is in C-contiguous order for the kernel
	#if not image.flags.c_contiguous:
	#	print('not contiguous')
	#	image = cp.ascontiguousarray(image)
	
	# Ensure raw is also contiguous
	#if raw is not None and not raw.flags.c_contiguous:
	#	raw = cp.ascontiguousarray(raw)

	if delta > 5:
		raise TypeError("Delta must be an less than or equal to 5 due to MAX_KERNEL_POINTS")
	if delta_fit > 5:
		raise ValueError("delta_fit must be at most 5 due to MAX_KERNEL_POINTS")
	
	depth, height, width = image.shape
	total_voxels = depth * height * width
	
	# Convert parameters to appropriate types
	threshold = cp.float32(threshold)
	sigmaZ = cp.float32(sigmaZ)
	sigmaXY = cp.float32(sigmaXY)
	
	count = cp.zeros(1, dtype=cp.uint32)
	threads = 256
	blocks = (total_voxels + threads - 1) // threads
	
	# 1st pass: Count only to help limit GPU RAM
	local_maxima_count_kernel((blocks,), (threads,),
		(image.ravel(), threshold, delta, count, depth, height, width, total_voxels))
	# Use stream.synchronize() instead of global synchronize for better performance
	stream = cp.cuda.get_current_stream()
	stream.synchronize()
	#cp.cuda.Device().synchronize()
	
	num = int(count.get()[0])
	
	if num == 0:
		return cp.zeros((0, 8), dtype=cp.float32)
	
	# Allocate output arrays, as large as the maxima that were counted. Every entry is written.
	z_out = cp.empty(num, dtype=cp.uint16)
	x_out = cp.empty_like(z_out)
	y_out = cp.empty_like(z_out)
	count[:] = 0  # reset counter
	
	# Call the kernel to find local maxima and store coordinates
	local_maxima_kernel((blocks,), (threads,), 
		(image.ravel(), threshold, delta, delta_fit,
		 z_out, x_out, y_out, count, depth, height, width, num))
	#cp.cuda.Device().synchronize()
	stream.synchronize()
	
	# Get the actual number found
	num_found = int(count.get()[0])
	
	if num_found == 0:
		return cp.zeros((0, 8), dtype=cp.float32)
	
	# Trim arrays
	z_out = z_out[:num_found]
	x_out = x_out[:num_found]
	y_out = y_out[:num_found]
	
	# Allocate output array for refined coordinates
	output = cp.zeros((num_found, 8), dtype=cp.float32)

	# the kernel reads float32 or uint16 raw as it is, so a whole uint16 image is not copied
	fit_kernel = delta_fit_cross_corr_kernel
	if raw.dtype == cp.uint16:
		fit_kernel = delta_fit_cross_corr_u16_kernel
	elif raw.dtype != cp.float32:
		raw = raw.astype(cp.float32)

	# Call delta fit kernel
	blocks = (num_found + threads - 1) // threads
	fit_kernel((blocks,), (threads,), 
		(image.ravel(), raw.ravel(), z_out, x_out, y_out, output, 
		 num_found, depth, height, width, delta_fit, sigmaZ, sigmaXY))
	
	#cp.cuda.runtime.deviceSynchronize()
	stream.synchronize()
	return output

if __name__ == "__main__":
	import numpy as np
	np.set_printoptions(suppress=True, linewidth=100)
	# Example Usage
	#cim = cp.random.rand(40, 300, 300).astype(cp.float32)
	cim = cp.random.rand(4, 4, 4).astype(cp.float32)
	print(cim)
	local = find_local_maxima(cim, 0.97, 1, 3, raw=cim)
	print('local.shape',local.shape, flush=True)
	print(local)
