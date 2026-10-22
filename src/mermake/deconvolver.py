"""Patch-wise FFT deconvolution on the GPU."""
import gc
from itertools import cycle

import cupy as cp
import numpy as np

from .gpu import blur
from .gpu.fill import reflect

def repeat_last(iterable):
	"""Yield the items of ``iterable``, then keep yielding the last one forever. Yields nothing if it is empty.
	"""
	it = iter(iterable)
	try:
		last = next(it)  # Get the first element
	except StopIteration:
		return  # Empty iterable, nothing to repeat
	yield last
	for item in it:
		yield item
		last = item  # Update last element
	while True:
		yield last  # Repeat last element indefinitely


def laplacian_3d(shape):
	"""Create a 3D Laplacian kernel for a given shape."""
	lap = cp.zeros(shape, dtype=cp.float16)
	z_c, y_c, x_c = shape[0] // 2, shape[1] // 2, shape[2] // 2
	lap[z_c, y_c, x_c] = 6
	lap[z_c - 1, y_c, x_c] = -1
	lap[z_c + 1, y_c, x_c] = -1
	lap[z_c, y_c - 1, x_c] = -1
	lap[z_c, y_c + 1, x_c] = -1
	lap[z_c, y_c, x_c - 1] = -1
	lap[z_c, y_c, x_c - 1] = -1
	#lap[z_c, y_c, x_c + 1] = -1  # Bug fix (previously had two -1s at the same position)
	return lap

def batch_laplacian_fft(batch_size, shape):
	"""Compute the 3D Laplacian in frequency space and prepare for batch processing."""
	lap = laplacian_3d(shape)  # Create a single 3D Laplacian
	#lap_fft = cp.fft.fftn(cp.fft.ifftshift(lap))  # Shift Laplacian to center before FFT
	lap_fft = cp.fft.fftn(lap)
	return lap_fft[None, ...]  # Add batch dimension without copying memory


class Deconvolver:
	"""FFT deconvolution of a z-stack, done patch by patch on the GPU.

	The xy plane is cut into square patches of ``patch_size`` pixels. Each patch gets
	``patch_overlap`` pixels of context on every side (reflected at the image border),
	is deconvolved with a regularized (Laplacian) Wiener-style filter built from the PSF,
	and only its central ``patch_size`` square is kept. The z axis is reflect-padded by
	``zpad`` planes on both sides instead of being cut.

	Parameters
	----------
	psfs : numpy.ndarray or dict of numpy.ndarray
		PSF with shape (z, y, x), or several PSFs sampled on a grid across the field of view,
		which are then used one per patch in order. Cropped or zero-padded to
		(depth, patch_size, patch_size) and normalized to sum 1.
	channel_shape : tuple of int
		(z, y, x) shape of one channel of the images that will be deconvolved.
	patch_size : int
		Edge of the square xy patches, in pixels. Not the raw image size.
	zpad : int
		Planes of reflected padding added above and below in z.
	patch_overlap : int
		Pixels of context around each patch, in xy.
	beta : float
		Weight of the Laplacian regularization term. Larger means smoother, weaker deconvolution.
	xp : module
		Array module, cupy.
	real_fft : bool
		Use real-to-complex FFTs. The patches and the PSF are real, so the result is the same up to
		rounding, in about 60 % of the time and with half the memory for the PSF spectrum and the FFT
		buffer. False uses complex FFTs.
	**kwargs
		Ignored, so a whole settings section can be passed with ``**vars(section)``.

	Notes
	-----
	The patch buffers are allocated once, with shape (patch_size + 2*patch_overlap) squared
	per z-plane, so ``patch_size`` and ``patch_overlap`` set the GPU memory use. The FFTs are much
	faster when the padded sizes, ``patch_size + 2*patch_overlap`` and ``depth + 2*zpad``, only have
	the prime factors 2, 3, 5 and 7.
	"""
	def __init__(self, psfs, channel_shape, patch_size=300, zpad=0, patch_overlap=89, beta=0.001, xp=cp,
	             real_fft=True, **kwargs):
		self.patch_size = patch_size
		self.depth = channel_shape[0]
		self.real_fft = real_fft

		if isinstance(psfs, dict):
			# new method using multiple psfs taken across fov
			psf_stack = np.stack(list(map(self.center_psf, list(psfs.values()))))
		else:
			# old single psf method
			psf_stack = self.center_psf(psfs)
			psf_stack = self.center_psf(psfs)[None, ...]
		psf_stack = np.pad(psf_stack, ((0, 0), (zpad, zpad), (patch_overlap, patch_overlap), (patch_overlap, patch_overlap)), mode='constant')
		# put the PSF peak (at n // 2) at index 0. Note -(n // 2), not -n // 2: the latter floors to
		# -ceil(n / 2), one voxel too far whenever n is odd
		shift = -(np.array(psf_stack.shape[1:]) // 2)
		psf_stack[:] = np.roll(psf_stack, shift=shift, axis=(1,2,3))
		
		# the filter of a real PSF is Hermitian, so the half of the spectrum along the last axis is all
		# that the real-to-complex FFTs need
		transform = xp.fft.rfftn if real_fft else xp.fft.fftn
		spectrum_shape = (*psf_stack.shape[:-1], psf_stack.shape[-1] // 2 + 1) if real_fft else psf_stack.shape
		psf_fft = xp.empty(spectrum_shape, dtype=xp.complex64)
		# have to do this by zslice due to gpu ram ~ 48GB
		for z in range(len(psf_fft)):
			psf_fft[z] = transform(cp.asarray(psf_stack[z]))
			psf_conj = xp.conj(psf_fft[z])
			psf_fft[z] *= psf_conj
			laplacian_fft = transform(laplacian_3d(psf_stack.shape[1:]))
			laplacian_fft *= laplacian_fft.conj()
			laplacian_fft *= beta
			psf_fft[z] += laplacian_fft
			psf_fft[z] = psf_conj / psf_fft[z]
		del laplacian_fft, psf_conj, psf_stack

		self.psf_fft = psf_fft
		
		gc.collect()  # Standard Python garbage collection
		if xp == cp:
			cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
			cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
			cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed

		# Preallocate patch arrays
		if patch_size:
			patch_shape = (patch_size + 2*patch_overlap, patch_size + 2*patch_overlap)
			self.patch_pad = xp.empty((2*zpad + self.depth, *patch_shape), dtype=xp.float32)
			self.patch_res = xp.empty((		 self.depth, *patch_shape), dtype=xp.float32)
			self.patch_fft = xp.empty(self.patch_pad.shape[:-1] + (psf_fft.shape[-1],), dtype=xp.complex64)
			self.patch_buf = xp.empty_like(self.patch_res)
			self.patch_tem = xp.empty_like(self.patch_res)

		self.patch_overlap = patch_overlap
		self.zpad = zpad
		self.xp = xp

	def patch_wise(self, image, flat_field=None, blur_radius=None, **kwargs):
		"""Deconvolve ``image`` one patch at a time.

		Parameters
		----------
		image : cupy.ndarray
			(z, y, x) image, uint16 or float.
		flat_field : cupy.ndarray, optional
			Flat field the patches are divided by before deconvolution.
		blur_radius : int, optional
			Width in pixels of the xy box blur subtracted from each deconvolved patch.

		Yields
		------
		x, y : int
			Position of the patch in the image. The patch buffers are reused, so copy
			what you need before advancing.
		deconvolved : cupy.ndarray
			Deconvolved patch including the overlap margin, float32.
		raw : cupy.ndarray
			The raw patch (flat field removed again) with the same shape.
		"""
		xp = cp.get_array_module(image)
		zpad = self.zpad
		patch_overlap = self.patch_overlap
		patch_size = self.patch_size
		patch_pad = self.patch_pad
		patch_fft = self.patch_fft
		patch_res = self.patch_res
		patch_buf = self.patch_buf
		patch_tem = self.patch_tem
	
		# use cycle to repeat the single PSF or iterate normally if multiple PSFs exist
		psf_ffts = cycle(self.psf_fft) if len(self.psf_fft) == 1 else iter(self.psf_fft)
		patches = self.patched(image)
		flat_field = flat_field if flat_field is None or flat_field.ndim == image.ndim else flat_field[np.newaxis] 
		flats = cycle([(None,None,None)]) if flat_field is None else self.patched(flat_field)
	
		# the big loop
		for (x,y,patch),(_,_,flat),psf_fft in zip(patches, flats, psf_ffts):
			zdim,xdim,ydim = patch.shape
			xstart = patch_overlap if x == 0 else 0
			ystart = patch_overlap if y == 0 else 0
			xend = xdim
			yend = ydim
			
			patch_pad[:] = 0.0
			patch_pad[ zpad:zpad+zdim, xstart:xstart+xdim, ystart:ystart+ydim] = patch      # fill
		
			# flat field correction
			if flat_field is not None:
				mask = (slice(zpad,-zpad), slice(xstart,xstart+xdim), slice(ystart,ystart+ydim))
				cp.divide(patch_pad[mask], flat, out=patch_pad[mask])

			# x reflect at left and right
			if x == 0:
				reflect(patch_pad, patch_overlap, mode='in', axis=1, out=patch_pad)
				xend += patch_overlap
			elif xdim < patch_size + (2 * patch_overlap):
				reflect(patch_pad, xdim-1, mode='out', axis=1, out=patch_pad)
				xend += patch_overlap
			# y reflect at top and bottom
			if y == 0:
				reflect(patch_pad, patch_overlap, mode='in', axis=2, out=patch_pad)
				yend += patch_overlap
			elif ydim < patch_size + (2 * patch_overlap):
				reflect(patch_pad, ydim-1, mode='out', axis=2, out=patch_pad)
				yend += patch_overlap
			# z reflect down
			reflect(patch_pad, zpad, mode='in', axis=0, out=patch_pad)
			# z relfect up
			reflect(patch_pad, zpad+zdim-1, mode='out', axis=0, out=patch_pad)
			#patch_pad[:zpad, :, :] = patch_pad[zpad, :, :]
			#patch_pad[-zpad:, :, :] = patch_pad[zpad+zdim-1, :, :]
			#patch_pad[zpad - zdim:zpad, :, :] = patch_pad[zpad:zpad+zdim, :, :][::-1, :, :]
			#patch_pad[zpad + zdim:zpad + zdim + zpad, :, :] = patch_pad[zdim:zpad+zdim, :, :][::-1, :, :]

			# the fft convolution and deconvolution
			if self.real_fft:
				patch_fft[:] = xp.fft.rfftn(patch_pad)
				xp.multiply(patch_fft, psf_fft, out=patch_fft)
				patch_res[:] = xp.fft.irfftn(patch_fft, s=patch_pad.shape)[zpad:-zpad]
			else:
				patch_fft[:] = xp.fft.fftn(patch_pad)
				xp.multiply(patch_fft, psf_fft, out=patch_fft)
				patch_res[:] = xp.fft.ifftn(patch_fft)[zpad:-zpad].real

			# optional blur subtraction
			if blur_radius is not None:
				blur.box_1d(patch_res, blur_radius, axis=1, out=patch_buf)
				blur.box_1d(patch_buf, blur_radius, axis=2, out=patch_tem)
				xp.subtract(patch_res, patch_tem, out=patch_res)
			
			# flat field uncorrection - put patch_pad back so it matches the raw patch
			if flat_field is not None:
				mask = (slice(zpad,-zpad), slice(xstart,xstart+xdim), slice(ystart,ystart+ydim))
				cp.multiply(patch_pad[mask], flat, out=patch_pad[mask])
			
			yield x,y,patch_res[:,:xend, :yend],patch_pad[zpad:-zpad, :xend, :yend]

	def apply(self, image, flat_field=None, blur_radius=None, output=None, **kwargs):
		"""Deconvolve the whole image and stitch the patches back together.

		Only the central ``patch_size`` square of each patch is used, so the overlap margins never
		end up in the output.

		Parameters
		----------
		image : cupy.ndarray
		    ``(z, y, x)`` image. Should be larger than ``patch_size`` in y and x.
		flat_field : cupy.ndarray, optional
		    Flat field to divide by.
		blur_radius : int, optional
		    Width in pixels of the xy box blur subtracted from each patch.
		output : cupy.ndarray, optional
		    float32 array with the shape of ``image`` to write into. Created if None.

		Returns
		-------
		cupy.ndarray
		    float32 deconvolved image.
		"""
		xp = cp.get_array_module(image)
		if output is None:
			output = xp.empty_like(image, dtype=xp.float32)

		patch_overlap = self.patch_overlap
		sz, sx, sy = image.shape
		for x, y, patch, _ in self.patch_wise(image, flat_field=flat_field, blur_radius=blur_radius):
				zdim,xdim,ydim = patch.shape
				xdim -= (2 * patch_overlap)
				ydim -= (2 * patch_overlap)
				output[:, x:x+xdim, y:y+ydim] = patch[:,patch_overlap:-patch_overlap, patch_overlap:-patch_overlap]
		return output

	def patched(self, image):
		"""
		Yield patches of the image do reflected padding later
		"""
		xp = cp.get_array_module(image)
		sz, sx, sy = image.shape
		self.nx = int(xp.ceil(sx / self.patch_size))
		self.ny = int(xp.ceil(sy / self.patch_size))

		patch_size = self.patch_size
		for x in range(0, sx, patch_size):
			for y in range(0, sy, patch_size):
				# Compute bounds with patch_overlap
				x_start = max(x - self.patch_overlap, 0)
				y_start = max(y - self.patch_overlap, 0)
				x_end = min(x + self.patch_size + self.patch_overlap, sx)
				y_end = min(y + self.patch_size + self.patch_overlap, sy)
	
				# Extract patch region
				patch = image[:, x_start:x_end, y_start:y_end]
				yield x, y, patch

	def unpatched(self, image):
		"""
		Reconstruct the original image from patched representation.

		Parameters:
		-----------
		patched_image : ndarray
			Stacked patches with shape (num_tiles, sz, patch_size+2*patch_overlap, patch_size+2*patch_overlap)

		Returns:
		--------
		ndarray
			Reconstructed image with shape (sz, sx, sy)
		"""

		# Extract the usable part of each patch (removing the patch_overlap)
		out = image[:, :, self.patch_overlap:-self.patch_overlap, self.patch_overlap:-self.patch_overlap]

		# Reshape to organize patches in a grid
		out = out.reshape(self.ny, self.nx, self.sz, self.patch_size, self.patch_size)
		
		# Transpose and reshape to reconstruct the image
		#out = out.transpose(2, 0, 3, 1, 4).reshape(self.sz, self.ny * self.patch_size, self.nx * self.patch_size)

		out = out.reshape(self.ny, self.nx, self.sz, self.patch_size, self.patch_size)
		out = cp.ascontiguousarray(out.transpose(2, 0, 3, 1, 4))  # Single contiguous conversion
		out = out.reshape(self.sz, self.ny * self.patch_size, self.nx * self.patch_size)

		return out

	def center_psf(self, psf):
		"""
		Inserts `psf` into a zero-padded array of `target_shape`, cropping if necessary.
	
		Parameters:
		- psf (ndarray): The PSF array (NumPy or CuPy).
		- target_shape (tuple): The desired output shape.
	
		Returns:
		- ndarray: The centered PSF inside a zero-padded/cropped array.
		"""
		xp = cp.get_array_module(psf)  # Handle NumPy or CuPy

		target_shape = xp.array([self.depth, self.patch_size, self.patch_size])
		psf_shape = xp.array(psf.shape)
		psff = xp.zeros(tuple(target_shape.tolist()), dtype=psf.dtype)  # Use same dtype for consistency
	
		# Compute start & end indices for both source (psf) and target (psff)
		start_psff = xp.maximum(0, (target_shape - psf_shape) // 2)
		end_psff = start_psff + xp.minimum(target_shape, psf_shape)
	
		start_psf = xp.maximum(0, (psf_shape - target_shape) // 2)
		end_psf = start_psf + xp.minimum(target_shape, psf_shape)
	
		# Assign using slices
		slices_psff = tuple(slice(int(s), int(e)) for s, e in zip(start_psff, end_psff))
		slices_psf = tuple(slice(int(s), int(e)) for s, e in zip(start_psf, end_psf))
		psff[slices_psff] = psf[slices_psf]

		# normalize at the end
		psff /= psff.sum()

		return psff

def full_deconv(image, psfs, flat_field = None, patch_size=300, zpad = None, patch_overlap = 1, beta = 0.001):
	"""Deconvolve one image with a new :class:`Deconvolver` and free its GPU memory afterwards.

	Parameters
	----------
	image : cupy.ndarray
	    ``(z, y, x)`` image.
	psfs : numpy.ndarray or dict
	    PSF, see :class:`Deconvolver`.
	flat_field : cupy.ndarray, optional
	    Flat field to divide by.
	patch_size, patch_overlap, beta : see :class:`Deconvolver`
	zpad : int, optional
	    Default: the depth of the image.

	Returns
	-------
	cupy.ndarray
	    float32 deconvolved image with the shape of ``image``.
	"""
	xp = cp.get_array_module(image)

	shape = image.shape
	if zpad is None:
		zpad = shape[0]
	deconvolver = Deconvolver(psfs, channel_shape=shape, zpad=zpad, patch_size=patch_size, patch_overlap=patch_overlap, beta=beta)
	deconv = deconvolver.apply(image, flat_field)

	del deconvolver
	gc.collect()  # Standard Python garbage collection
	if xp == cp:
		cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
		cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
		cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed
	return deconv



