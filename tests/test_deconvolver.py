import pytest

pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu
import numpy as np

try:
	import cupy as cp
	CUPY_AVAILABLE = True
except ImportError:
	import numpy as cp
	CUPY_AVAILABLE = False

from mermake.io.readers import read_im

from mermake.deconvolver import (
	batch_laplacian_fft, 
	repeat_last, 
	Deconvolver, 
	full_deconv
)


class TestUtilityFunctions:
	"""Test utility functions."""
	
	def test_repeat_last_basic(self):
		"""Test repeat_last with basic iterable."""
		result = list(zip(repeat_last([1, 2, 3]), range(6)))
		expected = [(1, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
		assert result == expected
	
	def test_repeat_last_empty(self):
		"""Test repeat_last with empty iterable."""
		result = list(repeat_last([]))
		assert result == []
	
	def test_repeat_last_single_element(self):
		"""Test repeat_last with single element."""
		result = list(zip(repeat_last([42]), range(5)))
		expected = [(42, 0), (42, 1), (42, 2), (42, 3), (42, 4)]
		assert result == expected


class TestLaplacianFunctions:
	"""Test Laplacian-related functions."""
	
	def test_batch_laplacian_fft(self):
		"""Test batch_laplacian_fft function."""
		batch_size = 2
		shape = (5, 5, 5)
		
		result = batch_laplacian_fft(batch_size, shape)
		
		# Check output shape (should have batch dimension)
		assert result.shape == (1, *shape)
		
		# Check it's in frequency domain (complex values)
		if CUPY_AVAILABLE:
			assert result.dtype == cp.complex64 or result.dtype == cp.complex128
		else:
			assert result.dtype == np.complex64 or result.dtype == np.complex128


class TestDeconvolverInitialization:
	"""Test Deconvolver class initialization."""
	
	@pytest.fixture
	def sample_psf(self):
		"""Create a sample PSF for testing."""
		# Create a small Gaussian-like PSF
		psf = np.zeros((10, 10, 10))
		psf[5, 5, 5] = 1.0
		# Add some spread
		psf[4:7, 4:7, 4:7] += 0.1
		return psf
	
	@pytest.fixture
	def sample_image_shape(self):
		"""Sample image shape for testing."""
		return (20, 100, 100)  # batch, z, x, y
	
	def test_deconvolver_init_single_psf(self, sample_psf, sample_image_shape):
		"""Test Deconvolver initialization with single PSF."""
		deconv = Deconvolver(
			psfs=sample_psf,
			channel_shape=sample_image_shape,
			patch_size=50,
			zpad=5,
			patch_overlap=10,
			beta=0.001
		)
		
		assert deconv.patch_size == 50
		assert deconv.patch_overlap == 10
		assert deconv.zpad == 5
		assert deconv.psf_fft is not None
	
	def test_deconvolver_init_multiple_psfs(self, sample_psf, sample_image_shape):
		"""Test Deconvolver initialization with multiple PSFs."""
		psfs_dict = {
			'center': sample_psf,
			'corner1': sample_psf * 0.9,
			'corner2': sample_psf * 1.1
		}
		
		deconv = Deconvolver(
			psfs=psfs_dict,
			channel_shape=sample_image_shape,
			patch_size=50,
			patch_overlap=10,
			beta=0.001
		)
		
		assert len(deconv.psf_fft) == 3
	
	def test_deconvolver_memory_allocation(self, sample_psf, sample_image_shape):
		"""Test that Deconvolver properly allocates memory for patches."""
		patch_size = 50
		patch_overlap = 10
		zpad = 5
		
		deconv = Deconvolver(
			psfs=sample_psf,
			channel_shape=sample_image_shape,
			patch_size=patch_size,
			zpad=zpad,
			patch_overlap=patch_overlap
		)
		
		expected_patch_shape = (patch_size + 2*patch_overlap, patch_size + 2*patch_overlap)
		expected_pad_shape = (2*zpad + sample_image_shape[0], *expected_patch_shape)
		expected_res_shape = (sample_image_shape[0], *expected_patch_shape)
		
		assert deconv.patch_pad.shape == expected_pad_shape
		assert deconv.patch_res.shape == expected_res_shape


class TestDeconvolverMethods:
	"""Test Deconvolver methods."""
	
	@pytest.fixture
	def deconv_instance(self):
		"""Create a Deconvolver instance for testing."""
		psf = np.zeros((10, 10, 10))
		psf[5, 5, 5] = 1.0
		channel_shape = (20, 100, 100)
		
		return Deconvolver(
			psfs=psf,
			channel_shape=channel_shape,
			patch_size=50,
			zpad=5,
			patch_overlap=10,
			beta=0.001
		)
	
	def test_center_psf_same_size(self, deconv_instance):
		"""Test center_psf with PSF same size as target."""
		psf = np.ones((20, 50, 50))  # Same as target
		result = deconv_instance.center_psf(psf)
		
		assert result.shape == (20, 50, 50)
		assert np.allclose(result.sum(), 1.0)  # Should be normalized
	
	def test_center_psf_smaller(self, deconv_instance):
		"""Test center_psf with smaller PSF."""
		psf = np.ones((10, 25, 25))  # Smaller than target
		result = deconv_instance.center_psf(psf)
		
		assert result.shape == (20, 50, 50)
		assert np.allclose(result.sum(), 1.0)  # Should be normalized
		# Check that PSF is centered
		assert result[10, 25, 25] > 0  # Should have non-zero values in center
	
	def test_center_psf_larger(self, deconv_instance):
		"""Test center_psf with larger PSF."""
		psf = np.ones((30, 75, 75))  # Larger than target
		result = deconv_instance.center_psf(psf)
		
		assert result.shape == (20, 50, 50)
		assert np.allclose(result.sum(), 1.0)  # Should be normalized
	
	def test_patched_basic(self, deconv_instance):
		"""Test basic tiling functionality."""
		image = np.random.random((20, 100, 100))
		
		patches = list(deconv_instance.patched(image))
		
		# Should have 4 patches (2x2 grid for 100x100 image with patch_size=50)
		assert len(patches) == 4
		
		# Check patch positions
		positions = [(x, y) for x, y, _ in patches]
		expected_positions = [(0, 0), (0, 50), (50, 0), (50, 50)]
		assert positions == expected_positions
	
	def test_patched_with_overlap(self, deconv_instance):
		"""Test that patches include proper patch_overlap."""
		image = np.random.random((20, 100, 100))
		
		patch_overlap = deconv_instance.patch_overlap
		patch_size = deconv_instance.patch_size
		for x, y, patch in deconv_instance.patched(image):
			zdim,xdim,ydim = patch.shape
			xstart = 0 if x == 0 else patch_overlap
			ystart = 0 if y == 0 else patch_overlap
			xend = xdim if xdim < patch_size else patch_size
			yend = ydim if ydim < patch_size else patch_size
			# First patch should include patch_overlap on right and bottom
			if x == 0 and y == 0:
				assert patch.shape == (20, 60, 60)  # 50 + 10 patch_overlap
			# Check that patch data matches original image
			#assert np.array_equal(patch[:,patch_overlap:-patch_overlap,patch_overlap:-patch_overlap], image[:, x:x+patch_size, y:y+patch_size]), f"x={x},y={y}"
			sub = patch[:,xstart:xstart+xend, ystart:ystart+yend]
			ref = image[:, x:x+patch_size, y:y+patch_size]
			assert sub.shape == ref.shape
			assert np.array_equal(sub, ref), f"x={x},y={y}"


class TestDeconvolverProcessing:
	"""Test actual deconvolution processing."""
	
	@pytest.fixture
	def simple_test_data(self):
		"""Create simple test data."""
		# Create a simple test image with a point source
		image = cp.zeros((10, 50, 50))
		image[5, 25, 25] = 100  # Point source in center
		
		# Create a simple PSF
		psf = np.zeros((10, 10, 10))
		psf[5, 5, 5] = 1.0
		
		return image, psf
	
	def test_apply_basic(self, simple_test_data):
		"""Test basic apply functionality."""
		image, psf = simple_test_data
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			patch_size=30,
			patch_overlap=5,
			zpad=2,
			beta=0.001
		)
		result = deconv.apply(image)
		
		assert result.shape == image.shape, 1
		assert result.dtype == np.float32, 2
		# Result should be finite (no NaN or inf)
		assert np.all(np.isfinite(result)), 3
	
	@pytest.mark.parametrize("blur_radius", [None, 2, 5])
	def test_apply_with_blur_subtraction(self, simple_test_data, blur_radius):
		"""Test apply with different blur subtraction settings."""
		image, psf = simple_test_data
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			patch_size=30,
			patch_overlap=5,
			zpad=2
		)
		
		result = deconv.apply(image, blur_radius=blur_radius)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))
	
	def test_apply_with_flat_field(self, simple_test_data):
		"""Test apply with flat field correction."""
		image, psf = simple_test_data
		# Create a simple flat field (slight gradient)
		flat_field = cp.ones(( 50, 50))
		flat_field[:, :] *= cp.linspace(0.8, 1.2, 50)[:, None]
		
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			patch_size=30,
			patch_overlap=5,
			zpad=2
		)
		
		result = deconv.apply(image, flat_field=flat_field)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))


class TestDeconvolverEdgeCases:
	"""Test edge cases and error conditions."""
	
	def test_small_image(self):
		"""Test with very small image."""
		image = cp.random.random((5, 20, 20))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			patch_size=15,  # Smaller than image
			patch_overlap=2,
			zpad=1
		)
		
		result = deconv.apply(image)
		assert result.shape == image.shape
	
	def test_single_patch_image(self):
		"""Test with image that fits in a single patch."""
		image = cp.random.random((10, 30, 30))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			patch_size=50,  # Larger than image
			patch_overlap=5,
			zpad=2
		)
		
		result = deconv.apply(image)
		assert result.shape == image.shape


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestCuPySpecific:
	"""Test CuPy-specific functionality."""
	
	def test_cupy_memory_management(self):
		"""Test that CuPy memory is properly managed."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		# Monitor memory before
		mempool = cp.get_default_memory_pool()
		initial_memory = mempool.used_bytes()
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			patch_size=30,
			patch_overlap=5,
			zpad=2,
			xp=cp
		)
		
		result = deconv.apply(image)
		
		# Clean up
		del deconv, result
		cp._default_memory_pool.free_all_blocks()
		
		# Memory should be cleaned up
		final_memory = mempool.used_bytes()
		assert final_memory <= initial_memory + 1024  # Allow small overhead


class TestFullDeconv:
	"""Test the full_deconv convenience function."""
	
	def test_full_deconv_basic(self):
		"""Test basic full_deconv functionality."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		result = full_deconv(
			image=image,
			psfs=psf,
			patch_size=30,
			patch_overlap=5,
			beta=0.001
		)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))
	
	def test_full_deconv_with_all_options(self):
		"""Test full_deconv with all options."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		flat_field = cp.ones(( 50, 50))
		
		result = full_deconv(
			image=image,
			psfs=psf,
			flat_field=flat_field,
			patch_size=30,
			zpad=5,
			patch_overlap=5,
			beta=0.01
		)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))
	
	def test_full_deconv_memory_cleanup(self):
		"""Test that full_deconv cleans up memory properly."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		# This should not raise memory errors and should clean up
		result = full_deconv(image=image, psfs=psf, patch_size=30)
		
		assert result is not None
		assert result.shape == image.shape

	def test_full_deconv_full(self):
		image = read_im('tests/examples/single/H1_TEST_set1/Conv_zscan__020.zarr')
		cim = cp.asarray(image[0])

		psfs = np.load('tests/examples/psf_750_Scope0_final.npy')
		result = full_deconv(image=cim, psfs=psfs, patch_size=300, patch_overlap=89)

		expected = cp.zeros([35,1200,1200], dtype=cp.float16)
		for z in range(35):
			zslice = cp.load(f'tests/examples/single/out/deconv.300.{z}.npz')
			expected[z] = zslice['arr_0']
	
		# Compare in float16 with a float16-appropriate tolerance. rtol=1e-6 is far
		# tighter than float16 can represent (1 ULP near these magnitudes is ~2), so it
		# effectively demanded bit-identical output and broke whenever the CuPy/CUDA
		# build changed. atol=2 covers last-bit quantization; rtol=1e-2 covers the small
		# proportional drift from FFT/library version differences.
		# The stored planes were made before the odd-size PSF roll was fixed, so they are shifted
		# by one plane in z (expected[z] is the deconvolution of plane z - 1). Compare with that
		# offset; plane 0 of the result has no counterpart. Regenerate the files to drop this.
		cp.testing.assert_allclose(result[:-1].astype(cp.float16), expected[1:], rtol=1e-2, atol=2)


class TestIntegrationAndPerformance:
	"""Integration tests and performance checks."""
	
	def test_deconvolution_preserves_intensity(self):
		"""Test that deconvolution roughly preserves total intensity."""
		# Create image with known total intensity
		image = cp.zeros((10, 50, 50))
		image[5, 25, 25] = 100
		total_input = np.sum(image)
		
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		result = full_deconv(
			image=image,
			psfs=psf,
			patch_size=30,
			beta=0.001  # Low regularization
		)
		
		total_output = np.sum(result)
		
		# Should preserve intensity within reasonable bounds
		# (exact preservation depends on regularization and boundary effects)
		assert abs(total_output - total_input) / total_input < 0.5
	
	def test_different_psf_formats(self):
		"""Test that different PSF input formats work."""
		image = cp.random.random((10, 50, 50))
		
		# Single PSF
		psf_single = np.zeros((5, 5, 5))
		psf_single[2, 2, 2] = 1.0
		
		result1 = full_deconv(image=image, psfs=psf_single, patch_size=30)
		
		# Dictionary of PSFs
		psfs_dict = {
			'center': psf_single,
			'corner': psf_single * 0.9
		}
		
		result2 = full_deconv(image=image, psfs=psfs_dict, patch_size=30)
		
		assert result1.shape == result2.shape == image.shape
		# Results should be different due to different PSF handling
		assert not np.allclose(result1, result2)

'''
# Mock tests for dependencies that might not be available
class TestMockDependencies:
	"""Test with mocked dependencies."""
	
	@patch('mermake.deconvolver.cp', new=np)
	def test_fallback_to_numpy(self):
		"""Test that code works when CuPy is mocked to NumPy."""
		image = np.random.random((5, 20, 20))
		psf = np.zeros((3, 3, 3))
		psf[1, 1, 1] = 1.0
		
		# This should work even with mocked CuPy
		result = full_deconv(image=image, psfs=psf, patch_size=15)
		assert result.shape == image.shape
	
	@patch('mermake.gpu.blur.box_1d')
	def test_with_mocked_blur(self, mock_blur):
		"""Test with mocked blur function."""
		mock_blur.return_value = None  # blur modifies in-place
		
		image = cp.random.random((5, 20, 20))
		psf = np.zeros((3, 3, 3))
		psf[1, 1, 1] = 1.0
		
		result = full_deconv(
			image=image, 
			psfs=psf, 
			patch_size=15,
			blur_radius=2  # This should trigger the blur calls
		)
		
		# Blur should have been called
		assert mock_blur.called
		assert result.shape == image.shape
'''

if __name__ == "__main__":
	pytest.main([__file__, "-v"])


class TestPositionPreserved:
	"""Deconvolution must not move features, whatever the parity of the padded sizes.

	The padded PSF is rolled so its peak (at n // 2) sits at index 0. An off-by-one there
	shifted the result by a voxel along every axis whose padded length is odd.
	"""

	IMAGE_XY = 40  # larger than patch_size, so the image is made of several patches

	@staticmethod
	def psf():
		zz, yy, xx = np.meshgrid(np.arange(-3, 3), np.arange(-5, 5), np.arange(-5, 5), indexing="ij")
		psf = np.exp(-(zz**2 / 2.0 + (yy**2 + xx**2) / (2 * 1.5**2))).astype(np.float32)
		return psf / psf.sum()

	# padded z is depth + 2*zpad = 3*depth - 2, so odd for odd depth; padded xy is
	# patch_size + 2*patch_overlap, so odd for odd patch_size
	@pytest.mark.parametrize("depth", [11, 12])
	@pytest.mark.parametrize("patch_size", [24, 25])
	def test_peak_stays_put(self, depth, patch_size):
		n = self.IMAGE_XY
		spot = (depth // 2, 30, 12)  # second patch in x, first in y
		image = cp.zeros((depth, n, n), dtype=cp.float32)
		image[spot] = 1000.0
		deconv = Deconvolver(self.psf(), image.shape, patch_size=patch_size, zpad=depth - 1,
		                     patch_overlap=8, beta=0.0001)
		out = deconv.apply(image)
		assert np.unravel_index(int(cp.argmax(out)), out.shape) == spot
