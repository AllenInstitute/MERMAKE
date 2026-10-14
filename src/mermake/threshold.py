"""Help choosing ``threshold`` for spot finding: ``mermake --suggest-threshold settings.toml``.

A few randomly chosen images are deconvolved the way the pipeline does it, and the maxima are
found once with a very low threshold. A voxel is only a candidate if it is brighter than the
threshold, and nothing else depends on the threshold, so the spots a higher threshold ``T``
would keep are exactly the ones with ``h > T``. One low run therefore gives the number of spots
at every threshold.

The same is done on the negated image. A real spot is bright, so the maxima of the negated image
are noise (or ringing of the deconvolution), and their count above ``T`` relative to the count of
the bright maxima is a rough estimate of the fraction of false positives at ``T``.

The sampling needs a GPU, the analysis (:func:`analyze`) does not.
"""
import os
import random
from types import SimpleNamespace

import numpy as np

H = 7  # column of the deconvolved value at the maximum in an Xh array
DAPI = "dapi"


def count_above(values, thresholds):
	"""Number of ``values`` greater than each of ``thresholds``."""
	values = np.sort(np.asarray(values, dtype=np.float64))
	return len(values) - np.searchsorted(values, thresholds, side="right")


def analyze(pos, neg, n_images, floor, max_fp=0.3, points=40):
	"""Spot counts as a function of the threshold, and a suggested threshold.

	Parameters
	----------
	pos, neg : array_like
	    ``h`` of the maxima of the image and of the negated image, found with threshold ``floor``.
	n_images : int
	    Number of images the maxima came from. Counts are reported per image.
	floor : float
	    The threshold the maxima were found with. Nothing below it is known.
	max_fp : float
	    Largest estimated fraction of false positives (negative over positive maxima) to accept.
	    Larger values give a lower, greedier threshold.
	points : int
	    Number of thresholds, log-spaced from ``floor`` to the 99.9th percentile of ``pos``.

	Returns
	-------
	dict
	    ``thresholds``, ``pos`` and ``neg`` (counts per image), ``fp`` (``neg / pos``, NaN where
	    there are no maxima), ``suggested`` (the lowest threshold with ``fp <= max_fp``, or None)
	    and ``at_floor`` (the suggestion is the lowest threshold tried, so the floor was too high
	    to see where the noise ends).
	"""
	pos = np.asarray(pos, dtype=np.float64)
	neg = np.asarray(neg, dtype=np.float64)
	pos = pos[np.isfinite(pos)]
	neg = neg[np.isfinite(neg)]
	if len(pos) == 0:
		return dict(thresholds=np.array([floor]), pos=np.zeros(1), neg=np.zeros(1),
		            fp=np.full(1, np.nan), suggested=None, at_floor=False)
	top = max(np.percentile(pos, 99.9), floor * 1.0001)
	thresholds = np.geomspace(floor, top, points) if floor > 0 else np.linspace(floor, top, points)
	n_pos = count_above(pos, thresholds)
	n_neg = count_above(neg, thresholds)
	with np.errstate(divide="ignore", invalid="ignore"):
		fp = np.where(n_pos > 0, n_neg / n_pos, np.nan)
	ok = np.flatnonzero(fp <= max_fp)
	suggested = float(thresholds[ok[0]]) if len(ok) else None
	return dict(thresholds=thresholds, pos=n_pos / n_images, neg=n_neg / n_images, fp=fp,
	            suggested=suggested, at_floor=bool(len(ok) and ok[0] == 0))


def _noise_floor(chan, flat, ctx, sigmas):
	"""``sigmas`` robust standard deviations above the median of the first deconvolved patch."""
	import cupy as cp
	for _, _, patch, _ in ctx.deconvolver.hybs.patch_wise(chan, flat, **vars(ctx.args.hybs)):
		values = patch[:, ::2, ::2]
		median = cp.median(values)
		mad = cp.median(cp.abs(values - median))
		return float(median + sigmas * 1.4826 * mad)


def sample(args, n_images=3, seed=0, floor_sigmas=4.0, dapi_floor=1.5, log=print):
	"""Find maxima with a low threshold in randomly chosen images, positive and negative.

	Parameters
	----------
	args : argparse.Namespace
	    The parsed command line (needs ``settings``).
	n_images : int
	    Number of images to sample, from any round and field of view.
	seed : int
	    Seed of the random choice of images.
	floor_sigmas : float
	    The lowest threshold for the color channels, in robust standard deviations of the
	    deconvolved first patch above its median. Lower values find many more candidates, which
	    takes time and GPU memory.
	dapi_floor : float
	    The lowest threshold for dapi, in standard deviations of the image.

	Returns
	-------
	dict
	    ``{channel: dict(pos, neg, floor)}`` with ``channel`` the color channel index, or
	    ``"dapi"``. ``pos`` and ``neg`` are numpy arrays of ``h``. Also ``n_images`` under the key
	    ``"n_images"``.
	"""
	import cupy as cp
	from .io.naming import FolderFilter
	from .io.readers import Container, load_flats, read_im
	from .pipeline import build_config, build_deconvolvers, find_hyb_maxima
	from .maxima import find_local_maxima

	build_config(args)
	paths = args.paths
	fov_min, fov_max = (-float("inf"), float("inf"))
	if paths.fov_range:
		fov_min, fov_max = map(float, paths.fov_range.split(":"))
	matches = FolderFilter(paths.hyb_range, paths.regex, fov_min, fov_max).get_matches(paths.hyb_folders)
	files = sorted(f for group in matches.values() for f in group)
	if not files:
		raise RuntimeError("No images found.")
	chosen = random.Random(seed).sample(files, min(n_images, len(files)))

	ncol, *chan_shape = read_im(chosen[0]).shape
	psfs = np.load(paths.psf_file, allow_pickle=True)
	flats = load_flats(shape=(ncol, *chan_shape), **vars(paths))
	buffer = cp.empty(chan_shape, dtype=cp.float32)
	chan = cp.empty(chan_shape, dtype=cp.uint16)
	ctx = SimpleNamespace(args=args, deconvolver=build_deconvolvers(psfs, chan_shape, args))

	found = {icol: dict(pos=[], neg=[], floor=None) for icol in (*range(ncol - 1), DAPI)}
	for path in chosen:
		log(f"sampling {path}")
		container = Container(path)
		for icol in range(ncol - 1):
			container[icol].compute()
			chan.set(container[icol].data)
			entry = found[icol]
			if entry["floor"] is None:
				entry["floor"] = _noise_floor(chan, flats[icol], ctx, floor_sigmas)
			floor = entry["floor"]
			for key, negate in (("pos", False), ("neg", True)):
				Xh, _ = find_hyb_maxima(chan, flats[icol], ctx, negate=negate, threshold=floor)
				entry[key].append(cp.asnumpy(Xh[:, H]))
		container[-1].compute()
		chan.set(container[-1].data)
		params = {**vars(args.dapi), "threshold": dapi_floor}
		ctx.deconvolver.dapi.apply(chan, flat_field=flats[-1], output=buffer, **vars(args.dapi))
		std = float(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size))
		cp.divide(buffer, std, out=buffer)
		entry = found[DAPI]
		entry["floor"] = dapi_floor
		entry["pos"].append(cp.asnumpy(find_local_maxima(buffer, raw=chan, **params)[:, H]))
		cp.multiply(buffer, -1, out=buffer)
		entry["neg"].append(cp.asnumpy(find_local_maxima(buffer, raw=chan, **params)[:, H]))
		del container

	results = {}
	for key, entry in found.items():
		results[key] = dict(pos=np.concatenate(entry["pos"]), neg=np.concatenate(entry["neg"]),
		                    floor=entry["floor"])
	results["n_images"] = len(chosen)
	return results


def format_table(label, result, max_rows=12):
	"""One channel of :func:`analyze`'s result as text, with about ``max_rows`` rows."""
	t = result["thresholds"]
	step = max(1, len(t) // max_rows)
	lines = [f"channel {label}", f"  {'threshold':>12} {'spots/image':>12} {'noise/image':>12} {'false pos.':>10}"]
	for i in range(0, len(t), step):
		fp = "" if np.isnan(result["fp"][i]) else f"{100 * result['fp'][i]:.0f}%"
		lines.append(f"  {t[i]:12.4g} {result['pos'][i]:12.1f} {result['neg'][i]:12.1f} {fp:>10}")
	return "\n".join(lines)


def plot(analyses, path):
	"""Draw spot counts against threshold for every channel to ``path``. Return whether matplotlib was found."""
	try:
		import matplotlib
	except ImportError:
		return False
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	n = len(analyses)
	cols = min(n, 3)
	rows = -(-n // cols)
	fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.3 * rows), squeeze=False)
	for ax, (label, a) in zip(axes.ravel(), analyses.items()):
		ax.plot(a["thresholds"], np.maximum(a["pos"], 0.5), label="maxima")
		ax.plot(a["thresholds"], np.maximum(a["neg"], 0.5), label="maxima of negated image")
		if a["suggested"] is not None:
			ax.axvline(a["suggested"], color="k", ls="--", lw=1, label=f"suggested {a['suggested']:.4g}")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_title(f"channel {label}")
		ax.set_xlabel("threshold")
		ax.set_ylabel("spots per image")
		ax.legend(fontsize=7)
	for ax in axes.ravel()[n:]:
		ax.set_visible(False)
	fig.tight_layout()
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return True


def suggest(args, n_images=3, max_fp=0.3):
	"""Run the sampling and print and plot the result. Entry point of ``--suggest-threshold``."""
	results = sample(args, n_images=n_images)
	n = results.pop("n_images")
	analyses = {label: analyze(r["pos"], r["neg"], n, r["floor"], max_fp=max_fp)
	            for label, r in results.items()}
	for label, a in analyses.items():
		print(format_table(label, a))
		print()
	hyb = [a["suggested"] for label, a in analyses.items() if label != DAPI and a["suggested"] is not None]
	print(f"Suggested thresholds for at most {100 * max_fp:.0f}% estimated false positives, from {n} images:")
	for label, a in analyses.items():
		if a["suggested"] is None:
			note = "no threshold reaches that, the noise estimate stays high"
		elif a["at_floor"]:
			note = f"{a['suggested']:.4g} (the lowest tried: the true value may be lower)"
		else:
			note = f"{a['suggested']:.4g}"
		print(f"  channel {label}: {note}")
	if hyb:
		print(f"\n[hybs] takes one threshold for all color channels. The lowest suggestion, {min(hyb):.4g}, "
		      "keeps the most spots.")
	if DAPI in analyses and analyses[DAPI]["suggested"] is not None:
		print(f"[dapi] threshold = {analyses[DAPI]['suggested']:.4g}")
	print("The noise estimate is rough: deconvolution ringing is not symmetric in sign.")
	path = os.path.join(args.settings.paths.output_folder, "threshold_suggestion.png")
	if plot(analyses, path):
		print(f"plot: {path}")
	return analyses
