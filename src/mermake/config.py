"""Run parameters for mermake.

A settings file (TOML) is loaded into a :class:`Config`. Only the experiment
specific keys in ``[paths]`` are required. Everything else has a default in
this module, so a settings file only lists what differs. Every field carries a
``doc`` string in its metadata. These feed ``mermake --print-config`` and the
configuration reference in the documentation.

This module must stay free of cupy: ``cli`` imports it before the GPU is chosen.
"""
import difflib
import glob
import os
import re
import sys
import textwrap
import typing
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from types import SimpleNamespace
from typing import List, Optional, Union

from .io.naming import FolderFilter

if sys.version_info >= (3, 11):
	import tomllib  # Python 3.11+ standard library
else:
	import tomli as tomllib  # Backport for older Python versions


class ConfigError(ValueError):
	"""The settings file is invalid. The message lists every problem found."""


def _f(doc, unit=None, **kwargs):
	return field(metadata={"doc": doc, "unit": unit}, **kwargs)


@dataclass(kw_only=True)
class Paths:
	"""The ``[paths]`` section: where the data is and where results go.

	Relative paths are resolved against the directory containing the settings file.
	"""
	psf_file: str = _f(
		"Point spread function, a `.npy` array with shape (z, y, x), or a pickled dict of such "
		"arrays sampled on a grid across the field of view. It is cropped or zero-padded to "
		"(image depth, patch_size, patch_size) and normalized to sum to 1. With a dict, set "
		"`patch_size` to the spacing of the sampling grid so each patch gets its own PSF.")
	flat_field_tag: str = _f(
		"Path prefix of the flat field files. Every file matching `<tag>*` is loaded (a `.npy` "
		"2D array or a `.npz` with key `im`, one per color channel), box blurred and divided by its median.")
	hyb_folders: List[str] = _f(
		"Directories to search. Each holds one sub-directory per hybridization round, which in "
		"turn holds one `.zarr` per field of view.")
	hyb_range: str = _f(
		"Rounds to process as `first:last`, written like the round folder names, e.g. "
		"`H1_BICAN_set1:H16_BICAN_set1`. The round number and the set number are compared "
		"numerically. The prefix and the middle part must match the folder name; a middle part "
		"of `*` matches anything.")
	output_folder: str = _f(
		"Where result files go. Created if missing. A field of view whose output files already "
		"exist is not recomputed.")
	fov_range: Optional[str] = _f(
		"Only process fields of view with index in `min:max` (inclusive), e.g. `0:10`. "
		"Default: all.", default=None)
	sum_channels: Optional[List[int]] = _f(
		"Advanced. Two channel indices (0-based, in the order the channels are stored, before any "
		"are merged) whose images are summed into one channel when an image is read. For "
		"acquisitions where one color was imaged twice to double its exposure. The sum is "
		"stored at the lower index and the image has one channel fewer, so the flat fields must "
		"match the merged count. Default: none.", default=None)
	dapi_channel: int = _f(
		"Advanced. Index of the dapi channel (0-based, counted after any `sum_channels` merge), or "
		"-1 for the last channel. The other channels are the hyb colors, numbered in stored order "
		"without the dapi channel. The flat fields are matched to the channels by this same index.",
		default=-1)
	hyb_save: str = _f(
		"Advanced. Filename template for the maxima of each color channel. Placeholders: "
		"{fov} (zarr name), {tag} (round folder name), {icol} (color channel index).",
		default='{fov}--{tag}--col{icol}__Xhfits.npz')
	dapi_save: str = _f(
		"Advanced. Filename template for the dapi maxima. Placeholders: {fov}, {tag}.",
		default='{fov}--{tag}--dapiFeatures.npz')
	drift_save: str = _f(
		"Advanced. Filename template for the drift file of a field of view. Placeholders: "
		"{fov}, {ifov} (field of view number), {iset} (set number).",
		default='drift_{fov}--_set{iset}.pkl')
	regex: str = _f(
		"Advanced. Splits a round folder name into five groups: prefix, round number, middle "
		"part, set number and suffix. Used to apply `hyb_range`.",
		default=r'([A-z]+)(\d+)_(.+)_set(\d+)(.*)')


@dataclass(kw_only=True)
class Params:
	"""Deconvolution and peak finding parameters, used for ``[hybs]`` and ``[dapi]``.

	The two sections have different defaults, see ``DEFAULTS``.
	"""
	patch_size: int = _f(
		"Not the raw image size. Edge of the square xy patches that each z-stack is cut into for "
		"FFT deconvolution. It sets GPU memory use: the buffers hold "
		"(patch_size + 2*patch_overlap)^2 pixels per z-plane. The PSF is cropped to this size.",
		unit="pixels")
	patch_overlap: int = _f(
		"Extra context added on each side of a patch (also zero-padded onto the PSF). Maxima "
		"within this margin of an interior patch edge are dropped, because the neighboring patch "
		"covers them. Should be larger than the PSF radius.", unit="pixels")
	beta: float = _f(
		"Regularization weight of the Laplacian term in the deconvolution filter "
		"conj(H) / (|H|^2 + beta*|L|^2). Larger values give a smoother result with less "
		"ringing but weaker deconvolution.")
	threshold: Union[float, List[float]] = _f(
		"A voxel is only a candidate maximum if it is brighter than this after deconvolution. "
		"[hybs]: image intensity, either one number for all color channels or a list with one "
		"number per color channel (all channels but the last, which is dapi). [dapi]: units of "
		"the standard deviation of the deconvolved image, because the dapi channel is divided "
		"by it first, and always one number.")
	blur_radius: int = _f(
		"Width of the xy box blur that is subtracted from each deconvolved patch (a high-pass "
		"that removes diffuse background).", unit="pixels")
	delta: int = _f(
		"Radius of the spherical neighborhood in which a candidate has to be the maximum. "
		"At most 5, a limit of the GPU kernel.", unit="voxels")
	delta_fit: int = _f(
		"Radius of the spherical neighborhood around each maximum, in the deconvolved image, "
		"used to refine its position (the intensity-weighted centroid after subtracting the "
		"neighborhood minimum) and to compute the scores `a` and `hn`. Between 1 and 5: 0 "
		"gives NaN scores and no refinement, and above 5 the GPU kernel's buffer truncates the "
		"neighborhood and biases the position.", unit="voxels")
	sigmaZ: float = _f(
		"Width in z of the Gaussian template that the neighborhood is correlated with to "
		"compute the scores `a` and `hn`. Does not change the position.", unit="voxels")
	sigmaXY: float = _f(
		"Width in x and y of the Gaussian template that the neighborhood is correlated with to "
		"compute the scores `a` and `hn`. Does not change the position.",
		unit="voxels")


OUTPUT_FORMATS = ("parquet", "npz", "both")


@dataclass(kw_only=True)
class Output:
	"""The ``[output]`` section: how results are written."""
	format: str = _f(
		"Format of the spot results. `parquet` writes a dataset under `<output_folder>/spots.parquet`, "
		"partitioned by set and field of view so that later steps can process one field of view at a "
		"time, plus per-image statistics under `<output_folder>/diagnostics.parquet`. `npz` writes the "
		"legacy one-file-per-channel `.npz` files. `both` writes both.",
		default="parquet")


DEFAULTS = {
	"hybs": dict(patch_size=500, patch_overlap=89, beta=0.0001, threshold=3600, blur_radius=30,
	             delta=1, delta_fit=3, sigmaZ=1, sigmaXY=1.5),
	"dapi": dict(patch_size=500, patch_overlap=89, beta=0.01, threshold=3.0, blur_radius=50,
	             delta=5, delta_fit=5, sigmaZ=1, sigmaXY=1.5),
}

SECTIONS = {"paths": Paths, "hybs": Params, "dapi": Params, "output": Output}

# keys that used to exist, with what to tell the user
RETIRED = {
	"tile_size": "renamed to 'patch_size'",
	"overlap": "renamed to 'patch_overlap'",
	"codebook": "is no longer used, remove it",
	"background_range": "background subtraction was removed, remove it",
}


@dataclass(kw_only=True)
class Config:
	"""All run parameters. Build one with :meth:`from_file` or :meth:`from_dict`."""
	paths: Paths
	hybs: Params
	dapi: Params
	output: Output

	@classmethod
	def from_file(cls, path, **kwargs):
		"""Load a TOML settings file. Relative paths are resolved against its directory."""
		try:
			with open(path, "rb") as fp:
				data = tomllib.load(fp)
		except OSError as e:
			raise ConfigError(f"cannot read {path}: {e}")
		except tomllib.TOMLDecodeError as e:
			raise ConfigError(f"{path} is not valid TOML: {e}")
		base_dir = os.path.dirname(os.path.abspath(path))
		try:
			return cls.from_dict(data, base_dir=base_dir, **kwargs)
		except ConfigError as e:
			raise ConfigError(f"{path}:\n{e}")

	@classmethod
	def from_dict(cls, data, base_dir=None, check_files=True):
		"""Validate a settings dict and fill in defaults.

		Parameters
		----------
		data : dict
			Parsed TOML, with the sections ``paths``, ``hybs`` and ``dapi``.
		base_dir : str, optional
			Relative paths are resolved against this directory (default: left as is).
		check_files : bool
			Check that the psf, flat field and input folders exist.

		Raises
		------
		ConfigError
			With one line per problem found.
		"""
		errors = []
		for name in data:
			if name not in SECTIONS:
				errors.append(_unknown(name, SECTIONS, "section"))
		sections = {}
		for name, section_cls in SECTIONS.items():
			values = data.get(name, {})
			if not isinstance(values, dict):
				errors.append(f"[{name}] must be a table")
				continue
			sections[name] = _build(section_cls, name, values, DEFAULTS.get(name, {}), errors)
		if "paths" in sections and sections["paths"] is not None:
			_check_paths(sections["paths"], base_dir, check_files, errors)
		for name in ("hybs", "dapi"):
			if sections.get(name) is not None:
				_check_params(sections[name], name, errors)
		if sections.get("output") is not None and sections["output"].format not in OUTPUT_FORMATS:
			errors.append(f"[output] format must be one of {', '.join(OUTPUT_FORMATS)}, "
			              f"got {sections['output'].format!r}")
		if errors:
			raise ConfigError("\n".join(f"  - {e}" for e in errors))
		return cls(**sections)

	def to_toml(self):
		"""Render the resolved settings as commented TOML (the inverse of :meth:`from_dict`)."""
		out = []
		for name in SECTIONS:
			section = getattr(self, name)
			out.append(f"[{name}]")
			for f in fields(section):
				doc = f.metadata["doc"]
				if f.metadata["unit"]:
					doc += f" ({f.metadata['unit']})"
				out.extend(textwrap.wrap(doc, 92, initial_indent="# ", subsequent_indent="# "))
				value = getattr(section, f.name)
				if value is None:
					out.append(f"# {f.name} = (not set)")
				else:
					out.append(f"{f.name} = {_toml_value(value)}")
				out.append("")
		return "\n".join(out)


def reference_rows():
	"""Describe every setting, for the documentation.

	Returns
	-------
	list of dict
		Keys ``section``, ``key``, ``type``, ``default``, ``required``, ``unit``, ``doc``.
		For ``[hybs]`` and ``[dapi]`` the defaults differ and are listed separately.
	"""
	rows = []
	for name, section_cls in SECTIONS.items():
		for f in fields(section_cls):
			default = DEFAULTS.get(name, {}).get(f.name, f.default)
			rows.append(dict(
				section=name, key=f.name, type=_type_name(f.type),
				default=None if default is MISSING else default,
				required=default is MISSING, unit=f.metadata["unit"], doc=f.metadata["doc"]))
	return rows


def _type_name(tp):
	if typing.get_origin(tp) is typing.Union:
		return " or ".join(_type_name(a) for a in typing.get_args(tp) if a is not type(None))
	if typing.get_origin(tp) in (list, List):
		return f"list of {_type_name(typing.get_args(tp)[0])}s"
	return {int: "integer", float: "number", str: "string"}[tp]


def _toml_value(value):
	if isinstance(value, str):
		if "'" not in value and "\n" not in value:
			return f"'{value}'"
		return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
	if isinstance(value, list):
		return "[" + ", ".join(_toml_value(v) for v in value) + "]"
	return repr(value)


def _unknown(key, valid, what="key"):
	close = difflib.get_close_matches(key, list(valid), n=1)
	hint = f", did you mean '{close[0]}'?" if close else f", valid: {', '.join(valid)}"
	return f"unknown {what} '{key}'{hint}"


def _coerce(value, tp):
	"""Return ``value`` as type ``tp`` or raise TypeError saying what was expected."""
	if typing.get_origin(tp) is typing.Union:
		if value is None:
			return None
		options = [a for a in typing.get_args(tp) if a is not type(None)]
		messages = []
		for option in options:
			try:
				return _coerce(value, option)
			except TypeError as e:
				messages.append(str(e))
		raise TypeError(" or ".join(messages))
	if typing.get_origin(tp) in (list, List):
		item = typing.get_args(tp)[0]
		if not isinstance(value, list):
			raise TypeError(f"a list of {_type_name(item)}s")
		try:
			return [_coerce(v, item) for v in value]
		except TypeError:
			raise TypeError(f"a list of {_type_name(item)}s") from None
	if tp is int:
		if isinstance(value, bool) or not isinstance(value, int):
			raise TypeError("an integer")
		return value
	if tp is float:
		if isinstance(value, bool) or not isinstance(value, (int, float)):
			raise TypeError("a number")
		return float(value)
	if tp is str:
		if not isinstance(value, str):
			raise TypeError("a string")
		return value
	raise AssertionError(tp)


def _build(cls, section, values, defaults, errors):
	"""Construct a section dataclass, appending every problem to ``errors``."""
	known = {f.name for f in fields(cls)}
	ok = True
	for key in values:
		if key in RETIRED:
			errors.append(f"[{section}] '{key}' {RETIRED[key]}")
			ok = False
		elif key not in known:
			errors.append(f"[{section}] " + _unknown(key, known))
			ok = False
	kwargs = {}
	for f in fields(cls):
		if f.name in values:
			try:
				kwargs[f.name] = _coerce(values[f.name], f.type)
			except TypeError as e:
				errors.append(f"[{section}] {f.name} must be {e}, got {values[f.name]!r}")
				ok = False
		elif f.name in defaults:
			kwargs[f.name] = _coerce(defaults[f.name], f.type)
		elif f.default is not MISSING:
			kwargs[f.name] = f.default
		else:
			errors.append(f"[{section}] missing required key '{f.name}': {f.metadata['doc']}")
			ok = False
	return cls(**kwargs) if ok else None


def _check_params(p, section, errors):
	def need(ok, msg):
		if not ok:
			errors.append(f"[{section}] {msg}")
	need(p.patch_size >= 1, f"patch_size must be at least 1, got {p.patch_size}")
	need(p.patch_overlap >= 0, f"patch_overlap must not be negative, got {p.patch_overlap}")
	if isinstance(p.threshold, list):
		need(section == "hybs", "threshold must be one number, only [hybs] takes a list")
		need(len(p.threshold) > 0, "threshold must not be an empty list")
	need(p.beta > 0, f"beta must be positive, got {p.beta}")
	need(p.blur_radius >= 1, f"blur_radius must be at least 1, got {p.blur_radius}")
	need(1 <= p.delta <= 5, f"delta must be between 1 and 5, got {p.delta}")
	need(1 <= p.delta_fit <= 5, f"delta_fit must be between 1 and 5, got {p.delta_fit}")
	need(p.sigmaZ > 0 and p.sigmaXY > 0, "sigmaZ and sigmaXY must be positive")


def _check_paths(p, base_dir, check_files, errors):
	def resolve(path):
		path = os.path.expanduser(path)
		if base_dir and not os.path.isabs(path):
			path = os.path.join(base_dir, path)
		return path

	p.psf_file = resolve(p.psf_file)
	p.flat_field_tag = resolve(p.flat_field_tag)
	p.hyb_folders = [resolve(d) for d in p.hyb_folders]
	p.output_folder = resolve(p.output_folder)
	if check_files:
		if not os.path.isfile(p.psf_file):
			errors.append(f"[paths] psf_file '{p.psf_file}' does not exist")
		if not glob.glob(p.flat_field_tag + "*"):
			errors.append(f"[paths] flat_field_tag '{p.flat_field_tag}' matches no files")
		for d in p.hyb_folders:
			if not os.path.isdir(d):
				errors.append(f"[paths] hyb_folders entry '{d}' is not a directory")
	try:
		pattern = re.compile(p.regex)
	except re.error as e:
		errors.append(f"[paths] regex is not valid: {e}")
		return
	if pattern.groups != 5:
		errors.append(f"[paths] regex must have 5 capture groups, it has {pattern.groups}")
		return
	if p.hyb_range.count(":") != 1:
		errors.append(f"[paths] hyb_range must look like 'first:last', got {p.hyb_range!r}")
	else:
		folder_filter = FolderFilter(p.hyb_range, p.regex, 0, 0)
		for end in (folder_filter.start_parts, folder_filter.end_parts):
			if end is None:
				errors.append(f"[paths] hyb_range {p.hyb_range!r} does not match regex {p.regex!r}")
				break
	if p.sum_channels is not None:
		if len(p.sum_channels) != 2 or p.sum_channels[0] == p.sum_channels[1] or min(p.sum_channels) < 0:
			errors.append(f"[paths] sum_channels must be two different channel indices, "
			              f"0 or more, got {p.sum_channels!r}")
	if isinstance(p.dapi_channel, int) and p.dapi_channel < -1:
		errors.append(f"[paths] dapi_channel must be -1 (the last channel) or a channel index, "
		              f"0 or more, got {p.dapi_channel!r}")
	if p.fov_range is not None:
		try:
			low, high = map(float, p.fov_range.split(":"))
			if low > high:
				errors.append(f"[paths] fov_range {p.fov_range!r} has min larger than max")
		except ValueError:
			errors.append(f"[paths] fov_range must look like 'min:max', got {p.fov_range!r}")


def namespace_to_array(obj, prefix=''):
	"""
	Recursively convert a dataclass, Namespace or dict to a list of (block, key, value) tuples.
	prefix is the accumulated parent keys joined by dots.
	"""
	rows = []
	if is_dataclass(obj) or isinstance(obj, SimpleNamespace):
		obj = vars(obj)
	if isinstance(obj, dict):
		for k, v in obj.items():
			full_key = f"{prefix}.{k}" if prefix else k
			if is_dataclass(v) or isinstance(v, (SimpleNamespace, dict)):
				rows.extend(namespace_to_array(v, prefix=full_key))
			else:
				rows.append((prefix, k, str(v)))
	else:
		# For other types just append
		rows.append((prefix, '', str(obj)))
	return rows
