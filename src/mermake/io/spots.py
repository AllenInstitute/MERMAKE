"""Spot results as Parquet datasets, partitioned by set and field of view.

Two datasets live in the output folder::

    spots.parquet/iset=<set>/ifov=<fov>/part-<kind>-r<round>-c<channel>.parquet
    diagnostics.parquet/iset=<set>/ifov=<fov>/part-<kind>-r<round>-c<channel>.parquet

A later step that matches spots across rounds and channels reads one field of view with
``pyarrow.dataset``::

    ds.dataset(root / "spots.parquet", partitioning="hive").to_table(filter=ds.field("ifov") == 3)

or hands one ``ifov`` directory to each worker. ``iset`` and ``ifov`` come from the directory
names, the other identifiers are columns.

This module does not import cupy: callers pass numpy arrays.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SPOTS_DIR = "spots.parquet"
DIAGNOSTICS_DIR = "diagnostics.parquet"

# columns of an Xh array, in order (see mermake.maxima.find_local_maxima)
XH_COLUMNS = ("z", "x", "y", "background", "a", "habs", "hn", "h")
SCORE_COLUMNS = ("background", "a", "habs", "hn", "h")
PERCENTILES = (5, 50, 95)

KIND_HYB = "hyb"
KIND_DAPI_PLUS = "dapi_plus"
KIND_DAPI_MINUS = "dapi_minus"

SPOTS_SCHEMA = pa.schema(
	[("kind", pa.dictionary(pa.int8(), pa.string())), ("ih", pa.int32()), ("icol", pa.int8())]
	+ [(name, pa.float32()) for name in XH_COLUMNS])

EXTRA_STATS = ("std_val", "raw_sum", "deconv_sum", "n_candidates", "n_edge_dropped")
STATS_SCHEMA = pa.schema(
	[("kind", pa.dictionary(pa.int8(), pa.string())), ("ih", pa.int32()), ("icol", pa.int8()),
	 ("n_spots", pa.int64())]
	+ [(f"{name}_p{p:02d}", pa.float64()) for name in SCORE_COLUMNS for p in PERCENTILES]
	+ [(name, pa.float64()) for name in EXTRA_STATS])


def part_dir(root, dataset, iset, ifov):
	"""Directory that holds the parts of one field of view."""
	return os.path.join(root, dataset, f"iset={iset}", f"ifov={ifov}")


def part_path(root, dataset, iset, ifov, kind, ih, icol):
	"""Path of the part of one round and channel. ``kind`` is a ``KIND_*`` value."""
	return os.path.join(part_dir(root, dataset, iset, ifov), f"part-{kind}-r{ih}-c{icol}.parquet")


def _write_atomic(table, path, metadata=None):
	"""Write ``table`` so that ``path`` only exists once it is complete (resume relies on it)."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	if metadata:
		table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
	tmp = f"{path}.tmp{os.getpid()}"
	pq.write_table(table, tmp, compression="zstd")
	os.replace(tmp, path)


def spots_table(parts, ih, icol):
	"""Build the table of one image channel.

	Parameters
	----------
	parts : dict
	    ``{kind: Xh}`` with ``Xh`` an ``(n, 8)`` array with the columns ``XH_COLUMNS``.
	ih : int
	    Round number.
	icol : int
	    Color channel, ``-1`` for dapi.
	"""
	arrays = []
	for kind, Xh in parts.items():
		Xh = np.asarray(Xh, dtype=np.float32).reshape(-1, len(XH_COLUMNS))
		n = len(Xh)
		cols = {
			"kind": pa.array([kind] * n, type=pa.string()).dictionary_encode().cast(SPOTS_SCHEMA.field("kind").type),
			"ih": pa.array(np.full(n, ih, dtype=np.int32)),
			"icol": pa.array(np.full(n, icol, dtype=np.int8)),
		}
		cols.update({name: pa.array(Xh[:, i]) for i, name in enumerate(XH_COLUMNS)})
		arrays.append(pa.table(cols, schema=SPOTS_SCHEMA))
	if not arrays:
		return SPOTS_SCHEMA.empty_table()
	return pa.concat_tables(arrays)


def write_spots(root, iset, ifov, ih, icol, parts, metadata=None):
	"""Write the spots of one image channel unless its part exists. Return the path.

	``parts`` is ``{kind: Xh}``: ``{"hyb": Xh}`` for a color channel and
	``{"dapi_plus": ..., "dapi_minus": ...}`` for dapi. ``metadata`` (str to str) goes in the
	schema metadata, e.g. the mermake version and settings.
	"""
	name = KIND_HYB if icol != -1 else "dapi"
	path = part_path(root, SPOTS_DIR, iset, ifov, name, ih, icol)
	if not os.path.exists(path):
		_write_atomic(spots_table(parts, ih, icol), path, metadata)
	return path


def spots_exist(root, iset, ifov, ih, icol):
	"""Whether the part of one image channel has been written."""
	name = KIND_HYB if icol != -1 else "dapi"
	return os.path.exists(part_path(root, SPOTS_DIR, iset, ifov, name, ih, icol))


def read_spots(root, iset, ifov, ih, icol):
	"""Read the part of one image channel back as ``{kind: (n, 8) float32 array}``."""
	name = KIND_HYB if icol != -1 else "dapi"
	table = pq.ParquetFile(part_path(root, SPOTS_DIR, iset, ifov, name, ih, icol)).read()
	kinds = table.column("kind").to_pylist()
	X = np.stack([table.column(c).to_numpy() for c in XH_COLUMNS], axis=1)
	X = X.reshape(-1, len(XH_COLUMNS)).astype(np.float32, copy=False)
	if icol != -1:
		return {KIND_HYB: X}
	kinds = np.asarray(kinds)
	return {k: X[kinds == k] for k in (KIND_DAPI_PLUS, KIND_DAPI_MINUS)}


def spot_stats(Xh, **extra):
	"""Summarize an ``(n, 8)`` spot array as a dict of the ``STATS_SCHEMA`` fields.

	Percentiles of the scores ignore NaN (scores are NaN when ``delta_fit`` is 0), and are NaN
	when there are no spots. ``extra`` may hold any of ``EXTRA_STATS``.
	"""
	Xh = np.asarray(Xh, dtype=np.float64).reshape(-1, len(XH_COLUMNS))
	stats = {"n_spots": int(len(Xh))}
	for name in SCORE_COLUMNS:
		col = Xh[:, XH_COLUMNS.index(name)]
		col = col[~np.isnan(col)]
		values = np.percentile(col, PERCENTILES) if len(col) else [np.nan] * len(PERCENTILES)
		for p, v in zip(PERCENTILES, values):
			stats[f"{name}_p{p:02d}"] = float(v)
	for key in extra:
		if key not in EXTRA_STATS:
			raise KeyError(f"unknown stat {key!r}")
	stats.update({k: (None if v is None else float(v)) for k, v in extra.items()})
	return stats


def write_stats(root, iset, ifov, ih, icol, rows):
	"""Write the statistics of one image channel unless its part exists.

	``rows`` is ``{kind: stats dict}`` as returned by :func:`spot_stats`.
	"""
	name = KIND_HYB if icol != -1 else "dapi"
	path = part_path(root, DIAGNOSTICS_DIR, iset, ifov, name, ih, icol)
	if os.path.exists(path):
		return path
	records = []
	for kind, stats in rows.items():
		record = {"kind": kind, "ih": ih, "icol": icol}
		record.update({field.name: stats.get(field.name) for field in STATS_SCHEMA
		               if field.name not in record})
		records.append(record)
	table = pa.Table.from_pylist(records, schema=STATS_SCHEMA)
	_write_atomic(table, path)
	return path


def open_dataset(root, dataset=SPOTS_DIR):
	"""Open a dataset of the output folder with ``iset`` and ``ifov`` as partition columns."""
	return ds.dataset(os.path.join(root, dataset), format="parquet", partitioning="hive")
