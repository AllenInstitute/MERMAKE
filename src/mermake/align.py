"""Drift correction between hybridization rounds by aligning dapi maxima."""
import os
import pickle

from .gpu.fft import fftconvolve

import cupy as cp
from cupyx.scipy.spatial import KDTree
#from scipy.signal import fftconvolve
#from scipy.spatial import KDTree

class Aligner:
	"""Find the translation between two sets of 3D points.

	The reference points are binned on a grid ``resc`` times coarser than the voxels and the
	translation is found by FFT cross-correlation of the two binned images. It is then refined
	by iterating nearest neighbor matches with a shrinking distance cutoff.

	Parameters
	----------
	X_ref : cupy.ndarray
	    Reference points, rows of ``(z, x, y, ...)`` in voxels. Only the first three columns
	    are used for matching.
	resc : float
	    Bin size of the coarse alignment, in voxels, and the starting match distance.
	trim : int
	    Points this many bins from the xy edge of the point cloud are ignored in the coarse step.
	th : float, optional
	    If given, only points whose last column (the intensity ``h``) exceeds it are used.
	"""
	def __init__(self, X_ref, resc=5, trim=10, th=None):
		self.xp = cp.get_array_module(X_ref)
		self.resc = resc
		self.trim = trim
		self.th = th
		
		# do thresholding of the maximas
		if th:
			X_ref = self.threshold(X_ref)

		# unlike scipy the cupy KDTree crashes if given no points, so catch it here
		if X_ref.size == 0:
			raise ValueError(
				"No points left after thresholding; cannot build KDTree. "
				"Check your data and/or threshold value."
			)

		self.X_ref = X_ref[:,:3]
		# Build KDTree once
		self.tree = KDTree(self.X_ref)

		# Precompute reference image for FFT correlation
		self.im_ref, self.Xm_ref = self.get_im_from_Xh(self.X_ref)

	def get_im_from_Xh(self, Xh):
		"""Bin points into a binary image on the coarse grid.

		Returns
		-------
		image : cupy.ndarray
		    float32 image that is 1 in every bin that holds a point.
		offset : cupy.ndarray
		    Integer offset of the image origin in bins (always zero here).
		"""
		# it would help to be able to use a preallocated buffer here so
		# that the imf array isnt created in ram every time, which is slow
		#  ie  get_im_from_Xh(self, Xh, out=buffer)
		xp = self.xp
		resc = self.resc
		trim = self.trim

		X = xp.round(Xh / resc).astype(xp.int32)
		Xm = xp.min(X, axis=0)
		XM = xp.max(X, axis=0)
		n = xp.asarray([0, trim, trim], dtype=xp.int32)
		keep = xp.all((X <= (XM - n)) & (X >= (Xm + n)), axis=-1)
		X = X[keep]
		# if trim is too extreme gracefully exit
		if X.shape[0] == 0:
			return xp.zeros((1, 1, 1), dtype=xp.float32), xp.array([0, 0, 0], dtype=xp.int32)

		Xm = xp.array([0, 0, 0], dtype=xp.int32)
		sz = xp.max(X, axis=0) + 1
		imf = xp.zeros(sz.tolist(), dtype=xp.float32)
		imf[tuple(X.T)] = 1
		return imf, Xm

	def get_Xtzxy(self, X, tzxy0, target=3):
		"""Refine a translation by matching each point to its nearest reference point.

		The match distance shrinks from ``resc`` to ``target`` voxels over five iterations, and the
		translation is updated each time to the mean offset of the matched pairs.

		Returns
		-------
		tzxy : cupy.ndarray
		    Refined ``(z, x, y)`` translation of ``X`` relative to the reference, in voxels.
		n_points : int
		    Number of points matched in the last iteration.
		"""
		xp = self.xp
		tzxy = tzxy0
		Npts = 0

		for dist_th in xp.linspace(self.resc, target, 5):
			XT = X - tzxy
			ds, inds = self.tree.query(XT)
			keep = ds < dist_th
			X_ref_ = self.X_ref[inds[keep]]
			X_ = X[keep]
			if X_.shape[0] == 0:
				# no points kept, skip update
				continue
			tzxy = xp.mean(X_ - X_ref_, axis=0)
		Npts = int(xp.sum(keep))
		return tzxy, Npts

	def get_shifted_slices(self, X, shape):
		"""Compute slices for signal and background images given integer shift."""
		slices_sig = []
		slices_bk = []
		tzxy = self.get_best_translation_points(X)
		tzxy = tzxy.round().astype(int).tolist()
		for t, sh in zip(tzxy, shape):
			if t > 0:
				slices_sig.append(slice(t, sh))
				slices_bk.append(slice(0, sh - t))
			else:
				slices_sig.append(slice(0, sh + t))
				slices_bk.append(slice(-t, sh))
		return tuple(slices_bk) , tuple(slices_sig)

	def get_best_translation_points(self, X, target=3, return_counts=False):
		"""Translation of the points ``X`` relative to the reference points.

		Parameters
		----------
		X : cupy.ndarray
		    Points, rows of ``(z, x, y, ...)`` in voxels.
		target : float
		    Final match distance of the refinement, in voxels.
		return_counts : bool
		    Also return the number of matched points.

		Returns
		-------
		tzxy : cupy.ndarray
		    ``(z, x, y)`` shift in voxels, such that ``X - tzxy`` lines up with the reference.
		n_points : int
		    Only if ``return_counts``.
		"""
		xp = self.xp
	
		# do thresholding of the maximas
		if self.th:
			X = self.threshold(X)

		X = X[:,:3]

		# Image for FFT correlation
		im, Xm = self.get_im_from_Xh(X)

		# Pure-CuPy FFT correlation
		im_cor = fftconvolve(im, self.im_ref[::-1, ::-1, ::-1])
		im_cor_shape = xp.asarray(im_cor.shape)
		im_ref_shape = xp.asarray(self.im_ref.shape)
		tzxy = xp.array(xp.unravel_index(xp.argmax(im_cor), im_cor_shape)) - im_ref_shape + 1 + Xm - self.Xm_ref
		tzxy = tzxy * self.resc

		# Refinement
		tzxy, Npts = self.get_Xtzxy(X, tzxy, target=target)

		if return_counts:
			return tzxy, Npts
		return tzxy


	def threshold(self, X):
		"""Keep the points whose last column exceeds ``th``. Returns an empty ``(0, dim)`` array if none do.
		"""
		xp = cp.get_array_module(X)
		th = self.th
		dim = X.shape[1]
		if X.size:
			X = X[X[:,-1]>th]
			if X.size:
				return X
		return xp.zeros([0,dim])

class DualAligner:
	"""Align using both the positive and the negative dapi maxima of a reference image.

	Parameters
	----------
	ref : object
	    Image with the attributes ``Xh_plus`` and ``Xh_minus``.
	th : float, optional
	    Intensity threshold passed to the two :class:`Aligner` objects.
	"""
	def __init__(self, ref, th=None):
		self.xp = cp.get_array_module(ref)
		self.plus = Aligner(ref.Xh_plus[:,:3], th = th)
		self.minus = Aligner(ref.Xh_minus[:,:3], th = th)
		self.th = th
	
	def get_best_translation_pointsV2(self, obj):
		"""Drift of ``obj`` relative to the reference image.

		The plus and minus alignments are averaged, weighted by their number of matched points, if
		they agree to within 2 voxels. Otherwise the one with more matches is used.

		Parameters
		----------
		obj : object
		    Image with the attributes ``Xh_plus`` and ``Xh_minus``.

		Returns
		-------
		tzxyf : cupy.ndarray
		    Final drift ``(z, x, y)`` in voxels, with the sign that corrects ``obj`` (the negative of
		    the translation).
		tzxy_plus, tzxy_minus : cupy.ndarray
		    The translations found with the positive and negative maxima.
		N_plus, N_minus : int
		    Number of matched points in each.
		"""
		xp = self.xp

		Xh_plus = obj.Xh_plus
		Xh_minus = obj.Xh_minus
		
		tzxy_plus,N_plus = self.plus.get_best_translation_points(Xh_plus, return_counts = True) 
		tzxy_minus,N_minus = self.minus.get_best_translation_points(Xh_minus, return_counts = True) 
	
		if N_plus + N_minus == 0:
			tzxyf = xp.zeros(3)
		elif xp.max(xp.abs(tzxy_minus - tzxy_plus)) <= 2:
			# weighted average
			tzxyf = -(tzxy_plus * N_plus + tzxy_minus * N_minus) / (N_plus + N_minus)
		else:
			# pick the stronger match
			tzxyf = -tzxy_plus if N_plus >= N_minus else -tzxy_minus

		return tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus

def drift_save(data, filepath):
	"""Pickle ``data`` to ``filepath``.
	"""
	# maybe put this inside io.py for consistency
	with open(filepath, 'wb') as f:
		pickle.dump(data, f)

def drift(block):
	"""Compute the drift of every image of a field of view relative to its middle round.

	Nothing is computed if the drift file of the block exists.

	Parameters
	----------
	block : mermake.io.queue.Block
	    The images of one field of view, with ``Xh_plus`` and ``Xh_minus`` set.

	Returns
	-------
	list, str or None
	    ``([drifts, files, ifov, ref_path], filepath)``, ready for :func:`drift_save`, or ``None``
	    if the drift file already exists. ``drifts`` holds one
	    ``(tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus)`` tuple per image and ``files`` the
	    round folder of each.
	"""
	ifov = block.ifov()
	filepath = block.drift_file()
	if not os.path.exists(filepath):
		ref = block[len(block)//2]
		dual = DualAligner(ref, th=4)
		drifts = []
		files = []
		for image in block:
			drift = dual.get_best_translation_pointsV2(image)
			drifts.append(drift)
			files.append(os.path.dirname(image.path))
		return [drifts, files, ifov, ref.path] , filepath

