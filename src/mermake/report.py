"""Diagnostic plots from the parquet results of a run: ``mermake-report <output_folder>``.

For every color channel, the distribution of each spot score is drawn once per round, so the
same channel can be compared across hybridization rounds. Two heatmaps (round by channel) show
the number of spots and the ratio of the deconvolved to the raw signal.

Needs matplotlib (``pip install mermake[report]``). Does not need a GPU.
"""
import argparse
import os
import sys

import numpy as np

from .io import spots

DEFAULT_SCORES = ("h", "habs", "hn")


def _plt():
	try:
		import matplotlib
	except ImportError:
		sys.exit("mermake-report needs matplotlib: pip install 'mermake[report]'")
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	return plt


def _round_label(iset, ih):
	return f"set{iset} r{ih}"


def score_histograms(table, score, bins=100):
	"""Histogram one score per round.

	Parameters
	----------
	table : pyarrow.Table
	    Spots of one channel, with the columns ``iset``, ``ih`` and ``score``.
	score : str
	    One of ``spots.SCORE_COLUMNS``.
	bins : int
	    Number of bins, shared by all rounds. The range covers the 0.5 to 99.5 percentile of all
	    the spots of the channel.

	Returns
	-------
	edges : numpy.ndarray
	    Bin edges.
	histograms : dict
	    ``{(iset, ih): density}``, in the order of the rounds. Rounds without a finite score are left out.
	"""
	values = table.column(score).to_numpy()
	isets = table.column("iset").to_numpy()
	rounds = table.column("ih").to_numpy()
	ok = np.isfinite(values)
	if not ok.any():
		return None, {}
	lo, hi = np.percentile(values[ok], [0.5, 99.5])
	if hi <= lo:
		hi = lo + 1
	edges = np.linspace(lo, hi, bins + 1)
	histograms = {}
	for iset, ih in sorted(set(zip(isets[ok].tolist(), rounds[ok].tolist()))):
		sel = ok & (isets == iset) & (rounds == ih)
		histograms[(iset, ih)] = np.histogram(values[sel], bins=edges, density=True)[0]
	return edges, histograms


def plot_score_histograms(table, icol, score, path):
	"""Draw :func:`score_histograms` of one channel to ``path``. Return whether anything was drawn."""
	plt = _plt()
	edges, histograms = score_histograms(table, score)
	if not histograms:
		return False
	centers = (edges[1:] + edges[:-1]) / 2
	cmap = plt.get_cmap("viridis")
	fig, ax = plt.subplots(figsize=(7, 4))
	for i, (key, density) in enumerate(histograms.items()):
		ax.plot(centers, density, color=cmap(i / max(len(histograms) - 1, 1)), lw=1,
		        label=_round_label(*key))
	ax.set_xlabel(score)
	ax.set_ylabel("density")
	ax.set_title(f"channel {icol}: {score} of the spots by round")
	if len(histograms) <= 20:
		ax.legend(fontsize=6, ncol=2)
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return True


def plot_heatmap(diagnostics, value, title, path, fmt="{:.0f}"):
	"""Draw a round by channel heatmap of the per-image statistic ``value``.

	``diagnostics`` is a pyarrow table of the diagnostics dataset; ``value`` is a column name or
	a function of the table returning an array. Cells combine the images of every field of view by mean.
	"""
	plt = _plt()
	isets = diagnostics.column("iset").to_numpy()
	rounds = diagnostics.column("ih").to_numpy()
	cols = diagnostics.column("icol").to_numpy()
	vals = value(diagnostics) if callable(value) else diagnostics.column(value).to_numpy()
	vals = np.asarray(vals, dtype=float)
	row_keys = sorted(set(zip(isets.tolist(), rounds.tolist())))
	col_keys = sorted(c for c in set(cols.tolist()) if c != -1)
	if not row_keys or not col_keys:
		return False
	grid = np.full((len(row_keys), len(col_keys)), np.nan)
	for r, (iset, ih) in enumerate(row_keys):
		for c, icol in enumerate(col_keys):
			sel = (isets == iset) & (rounds == ih) & (cols == icol) & np.isfinite(vals)
			if sel.any():
				grid[r, c] = vals[sel].mean()
	fig, ax = plt.subplots(figsize=(1 + 0.8 * len(col_keys), 1 + 0.35 * len(row_keys)))
	image = ax.imshow(grid, aspect="auto", cmap="viridis")
	ax.set_xticks(range(len(col_keys)), [str(c) for c in col_keys])
	ax.set_yticks(range(len(row_keys)), [_round_label(*k) for k in row_keys], fontsize=7)
	ax.set_xlabel("channel")
	ax.set_title(title)
	fig.colorbar(image, ax=ax)
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return True


def make_report(output_folder, report_dir=None, scores=DEFAULT_SCORES):
	"""Write the diagnostic plots of the run in ``output_folder``. Return the paths written.

	Parameters
	----------
	output_folder : str
	    The ``output_folder`` of the run, which holds ``spots.parquet`` and ``diagnostics.parquet``.
	report_dir : str, optional
	    Where the PNG files go. Default ``<output_folder>/report``.
	scores : sequence of str
	    The spot scores to draw histograms of.
	"""
	report_dir = report_dir or os.path.join(output_folder, "report")
	os.makedirs(report_dir, exist_ok=True)
	written = []

	dataset = spots.open_dataset(output_folder, spots.SPOTS_DIR)
	import pyarrow.dataset as ds
	channels = sorted(c for c in dataset.to_table(columns=["icol"]).column("icol").unique().to_pylist()
	                  if c != -1)
	for icol in channels:
		table = dataset.to_table(columns=["iset", "ih", *scores], filter=ds.field("icol") == icol)
		for score in scores:
			path = os.path.join(report_dir, f"hist_{score}_col{icol}.png")
			if plot_score_histograms(table, icol, score, path):
				written.append(path)

	diagnostics = spots.open_dataset(output_folder, spots.DIAGNOSTICS_DIR).to_table()
	path = os.path.join(report_dir, "n_spots.png")
	if plot_heatmap(diagnostics, "n_spots", "mean number of spots per image", path):
		written.append(path)
	path = os.path.join(report_dir, "deconv_raw_ratio.png")
	def ratio(t):
		return (t.column("deconv_sum").to_numpy(zero_copy_only=False).astype(float)
		        / t.column("raw_sum").to_numpy(zero_copy_only=False).astype(float))
	if plot_heatmap(diagnostics, ratio, "sum of deconvolved / sum of raw", path):
		written.append(path)
	return written


def main(argv=None):
	"""Command line entry point of ``mermake-report``."""
	parser = argparse.ArgumentParser(
		prog="mermake-report", description="Plot diagnostics from the parquet output of a mermake run.")
	parser.add_argument("output_folder", help="the output_folder of the run")
	parser.add_argument("-o", "--out", help="where to write the plots (default <output_folder>/report)")
	parser.add_argument("--score", action="append", choices=spots.SCORE_COLUMNS,
	                    help="score to draw histograms of, repeatable (default: h, habs, hn)")
	args = parser.parse_args(argv)
	for path in make_report(args.output_folder, args.out, tuple(args.score or DEFAULT_SCORES)):
		print(path)


if __name__ == "__main__":
	main()
