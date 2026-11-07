"""Drift correction between hybridization rounds by aligning dapi maxima.

The alignment runs on numpy arrays (scipy, on the CPU) or on cupy arrays (on the GPU), whichever it
is given. The pipeline and ``mermake --drift`` use the CPU: the dapi maxima of an image are a few
thousand points, so the work is small, and on the CPU it neither waits for nor holds up the GPU.
cupy is only imported for cupy input, so the CPU path also runs on a machine without a GPU.
"""
import os
import pickle
from pathlib import Path

import numpy as np


def get_array_module(x):
	"""``numpy`` for numpy arrays (and anything that is not a cupy array), ``cupy`` for cupy arrays."""
	if type(x).__module__.startswith("cupy"):
		import cupy
		return cupy
	return np


def to_numpy(x):
	"""``x`` as a numpy array, copied from the GPU if it is a cupy array."""
	return x.get() if hasattr(x, "get") and not isinstance(x, np.ndarray) else np.asarray(x)


def _backend(xp):
	"""``(KDTree, fftconvolve)`` for the array module ``xp``."""
	if xp is np:
		from scipy.signal import fftconvolve
		from scipy.spatial import cKDTree
		return cKDTree, fftconvolve
	from cupyx.scipy.spatial import KDTree
	from .gpu.fft import fftconvolve
	return KDTree, fftconvolve

class Aligner:
	"""Find the translation between two sets of 3D points.

	The reference points are binned on a grid ``resc`` times coarser than the voxels and the
	translation is found by FFT cross-correlation of the two binned images. It is then refined
	by iterating nearest neighbor matches with a shrinking distance cutoff.

	Parameters
	----------
	X_ref : numpy.ndarray or cupy.ndarray
	    Reference points, rows of ``(z, x, y, ...)`` in voxels. Only the first three columns
	    are used for matching. The type chooses the backend: scipy on the CPU or cupy on the GPU.
	resc : float
	    Bin size of the coarse alignment, in voxels, and the starting match distance.
	trim : int
	    Points this many bins from the xy edge of the point cloud are ignored in the coarse step.
	th : float, optional
	    If given, only points whose last column (the intensity ``h``) exceeds it are used.
	"""
	def __init__(self, X_ref, resc=5, trim=10, th=None):
		self.xp = get_array_module(X_ref)
		KDTree, self.fftconvolve = _backend(self.xp)
		self.resc = resc
		self.trim = trim
		self.th = th
		
		# do thresholding of the maximas
		if th:
			X_ref = self.threshold(X_ref)

		# the cupy KDTree crashes if given no points, so refuse them here for both backends
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

		# FFT correlation
		im_cor = self.fftconvolve(im, self.im_ref[::-1, ::-1, ::-1])
		im_ref_shape = xp.asarray(self.im_ref.shape)
		tzxy = xp.array(xp.unravel_index(xp.argmax(im_cor), tuple(im_cor.shape))) - im_ref_shape + 1 + Xm - self.Xm_ref
		tzxy = tzxy * self.resc

		# Refinement
		tzxy, Npts = self.get_Xtzxy(X, tzxy, target=target)

		if return_counts:
			return tzxy, Npts
		return tzxy


	def threshold(self, X):
		"""Keep the points whose last column exceeds ``th``. Returns an empty ``(0, dim)`` array if none do.
		"""
		xp = get_array_module(X)
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
		self.xp = get_array_module(ref.Xh_plus)
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
	"""Pickle ``data`` to ``filepath``, which only exists once it is complete.
	"""
	# maybe put this inside io.py for consistency
	tmp = f"{filepath}.tmp{os.getpid()}"
	try:
		with open(tmp, 'wb') as f:
			pickle.dump(data, f)
		os.replace(tmp, filepath)
	finally:
		if os.path.exists(tmp):
			os.remove(tmp)

def snapshot(image):
	"""What the drift needs of an image, as numpy: ``path``, ``ih``, ``Xh_plus`` and ``Xh_minus``.

	Taken in the main thread so the drift can run in the background after the image is gone.
	"""
	from types import SimpleNamespace
	return SimpleNamespace(path=image.path, ih=getattr(image, "ih", None),
	                       Xh_plus=to_numpy(image.Xh_plus), Xh_minus=to_numpy(image.Xh_minus))


def drift_rounds(ref, images, th=4):
	"""Drift of each of ``images`` relative to ``ref``, all with ``Xh_plus`` and ``Xh_minus``.

	Returns one ``(tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus)`` tuple per image, see
	:meth:`DualAligner.get_best_translation_pointsV2`. ``ref`` itself gives a drift of zero.
	"""
	dual = DualAligner(ref, th=th)
	return [dual.get_best_translation_pointsV2(image) for image in images]


def drift(block):
	"""Compute the drift of every image of a field of view relative to its first round.

	The reference is the first round, so the drift of a round only needs that round and the
	first one, and can be computed as soon as both are fitted. Runs on the CPU.
	Nothing is computed if the drift file of the block exists.

	Parameters
	----------
	block : mermake.io.queue.Block
	    The images of one field of view in round order, with ``Xh_plus`` and ``Xh_minus`` set.

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
		images = [snapshot(image) for image in block]
		drifts = drift_rounds(images[0], images)
		return [drifts, [os.path.dirname(image.path) for image in images], ifov, images[0].path], filepath


def save_drift(images, iset, ifov, drift_file, output_folder, output_format, complete=True):
	"""Compute and write the drift of a field of view that is not on disk yet.

	Parameters
	----------
	images : list
	    :func:`snapshot` of the fitted rounds in round order. The first is the reference, so it
	    must be the first round of the field of view.
	complete : bool
	    Whether ``images`` holds every round. The pickle ``drift_file`` is only written then; with
	    the parquet format each round also gets its own row in ``drift.parquet`` as soon as it is
	    fitted.

	Returns
	-------
	int
	    The number of rounds whose drift was computed.
	"""
	from .io import spots
	ref = images[0]
	parquet = output_format in ("parquet", "both")
	todo = [im for im in images if not (parquet and spots.drift_exists(output_folder, iset, ifov, im.ih))]
	write_pickle = complete and not os.path.exists(drift_file)
	if write_pickle:
		todo = images
	if not todo:
		return 0
	results = drift_rounds(ref, todo)
	if parquet:
		for image, result in zip(todo, results):
			spots.write_drift(output_folder, iset, ifov, image.ih, ref.ih, result)
	if write_pickle:
		drift_save([results, [os.path.dirname(im.path) for im in todo], ifov, ref.path], drift_file)
	return len(todo)


def load_dapi_maxima(path, iset, ifov, paths, output_format):
	"""The saved dapi maxima of the image at ``path`` as a :func:`snapshot`, or ``None`` if not fitted yet."""
	from types import SimpleNamespace
	from .io import spots
	from .io.naming import get_ih
	ih = get_ih(os.path.dirname(path))
	if output_format in ("parquet", "both"):
		if not spots.spots_exist(paths.output_folder, iset, ifov, ih, -1):
			return None
		parts = spots.read_spots(paths.output_folder, iset, ifov, ih, -1)
		plus, minus = parts[spots.KIND_DAPI_PLUS], parts[spots.KIND_DAPI_MINUS]
	else:
		name = paths.dapi_save.format(fov=Path(path).stem, tag=Path(path).parent.name)
		npz = os.path.join(paths.output_folder, name)
		if not os.path.exists(npz):
			return None
		with np.load(npz) as data:
			plus, minus = data["Xh_plus"], data["Xh_minus"]
	return SimpleNamespace(path=path, ih=ih, Xh_plus=plus, Xh_minus=minus)


def _drift_fov(iset, ifov, files, paths, output_format):
	"""``mermake --drift`` for one field of view: ``(rounds computed, whether it waits for its first round)``."""
	from .io import spots
	from .io.naming import get_ih
	files = sorted(files, key=lambda f: get_ih(os.path.dirname(f)))
	drift_file = os.path.join(paths.output_folder, paths.drift_save.format(ifov=ifov, fov=Path(files[0]).stem, iset=iset))
	if os.path.exists(drift_file) and (output_format == "npz" or all(
			spots.drift_exists(paths.output_folder, iset, ifov, get_ih(os.path.dirname(f))) for f in files)):
		return 0, False   # finished earlier: do not read its maxima again
	images = [load_dapi_maxima(f, iset, ifov, paths, output_format) for f in files]
	if images[0] is None:
		return 0, True
	fitted = [im for im in images if im is not None]
	return save_drift(fitted, iset, ifov, drift_file, paths.output_folder, output_format,
	                  complete=len(fitted) == len(images)), False


def run_drift(config, shard=None, jobs=None):
	"""``mermake --drift``: compute the drift of every fitted round from the saved dapi maxima.

	Reads only the results in the output folder (and lists the image folders to know the rounds),
	never the images, and needs no GPU. Rounds whose drift is on disk are skipped, so it can be run
	again while an acquisition or a run is still going: a round is aligned as soon as it and the
	first round of its field of view are fitted. The fields of view are shared out over ``jobs``
	processes (default: one per core); each takes about 1-2 s of CPU per round.
	"""
	from concurrent.futures import ProcessPoolExecutor
	from .io.naming import FolderFilter, shard_matches
	paths, output_format = config.paths, config.output.format
	fov_min, fov_max = (-float("inf"), float("inf"))
	if paths.fov_range:
		fov_min, fov_max = map(float, paths.fov_range.split(":"))
	matches = shard_matches(FolderFilter(paths.hyb_range, paths.regex, fov_min, fov_max).get_matches(paths.hyb_folders), shard)
	keys = sorted(matches)
	jobs = jobs or os.cpu_count() or 1
	with ProcessPoolExecutor(max_workers=min(jobs, max(1, len(keys)))) as pool:
		results = list(pool.map(_drift_fov, *zip(*[(iset, ifov, matches[(iset, ifov)], paths, output_format)
		                                             for iset, ifov in keys]))) if keys else []
	rounds = sum(n for n, _ in results)
	done = sum(1 for n, _ in results if n)
	waiting = sum(1 for _, w in results if w)
	print(f"drift: {rounds} rounds of {done} fields of view computed; {waiting} of {len(matches)} fields of view "
	      f"wait for their first round", flush=True)
	return rounds
