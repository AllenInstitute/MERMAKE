"""FFT based convolution on the GPU."""
import cupy as cp

def fftconvolve(a, b, mode='full'):
	"""N-dimensional convolution by FFT, like ``scipy.signal.fftconvolve``.

	Parameters
	----------
	a, b : cupy.ndarray
	    Arrays with the same number of dimensions.
	mode : {'full', 'same', 'valid'}
	    ``full`` returns the whole convolution, ``same`` the part with the shape of ``a`` and
	    ``valid`` the part computed without zero padding.

	Returns
	-------
	cupy.ndarray
	    Real-valued result.
	"""
	# Compute the size for FFT in each dimension
	shape = [sa + sb - 1 for sa, sb in zip(a.shape, b.shape)]

	# Perform FFTs (n-dimensional)
	fa = cp.fft.fftn(a, shape)
	fb = cp.fft.fftn(b, shape)

	# Multiply in frequency domain and inverse FFT
	out = cp.fft.ifftn(fa * fb).real

	# Handle cropping modes
	if mode == 'full':
		return out
	elif mode == 'same':
		start = [(o - s) // 2 for o, s in zip(out.shape, a.shape)]
		end = [start[i] + a.shape[i] for i in range(len(start))]
		slices = tuple(slice(start[i], end[i]) for i in range(len(start)))
		return out[slices]
	elif mode == 'valid':
		valid_shape = [sa - sb + 1 for sa, sb in zip(a.shape, b.shape)]
		start = [sb - 1 for sb in b.shape]
		end = [start[i] + valid_shape[i] for i in range(len(start))]
		slices = tuple(slice(start[i], end[i]) for i in range(len(start)))
		return out[slices]
	else:
		raise ValueError("mode must be 'full', 'same', or 'valid'")
