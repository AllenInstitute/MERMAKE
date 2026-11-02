"""Parsing round, set and field-of-view information from folder and file names."""
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional, Union


def get_ih(file_path):
	"""Round number: the first run of digits in the base name of ``file_path``.

	Names without digits sort last, with ``10**100``.
	"""
	basename = os.path.basename(file_path)
	m = re.search(r'\d+', basename)  # first contiguous digits
	if m:
		return int(m.group())
	return 10**100
	#raise ValueError(f"No number found in {basename}")

def get_ifov(file_path):
	"""Extract ifov from filename - finds last digits before .zarr"""
	filename = Path(file_path).name  # Keep full filename with extension
	match = re.search(r'([0-9]+)[^0-9]*\.zarr', filename)
	if match:
		return int(match.group(1))
	raise ValueError(f"No digits found before .zarr in filename: {filename}")

def get_iset(file_path):
	"""
	Recursively extract 'iset' from filename or parent directories.
	Finds last digits after the word '_set'.
	"""
	path = Path(file_path)  # ensure Path object
	filename = path.name
	match = re.search(r'_set([0-9]+)', filename)
	if match:
		return int(match.group(1))

	# Base case: reached root
	if path.parent == path:
		raise ValueError(f"No digits found after the word _set in {file_path}")

	# Recurse up
	return get_iset(path.parent)

def shard_keys(keys, index, count):
	"""The ``index``-th of ``count`` disjoint shares of the ``(iset, ifov)`` keys.

	The keys are sorted first, so every process gets the same answer whatever order its own
	directory scan returned them in, and then every ``count``-th key is taken. The shares are
	disjoint and together hold every key. A share is empty when there are fewer keys than
	``count``. Shard on fields of view, never on rounds: the drift of a field of view needs all
	of its rounds in one process.

	Parameters
	----------
	keys : iterable of (int, int)
	    ``(iset, ifov)`` pairs, the keys of :meth:`FolderFilter.get_matches`.
	index, count : int
	    Take share ``index`` (from 0) of ``count``.

	Returns
	-------
	list of (int, int)
	    The selected keys in sorted order.
	"""
	if count < 1 or not 0 <= index < count:
		raise ValueError(f"shard {index}/{count}: the index must be in 0..{count - 1}")
	return sorted(keys)[index::count]


def shard_matches(matches, shard=None):
	"""``matches`` of :meth:`FolderFilter.get_matches` restricted to a shard and sorted by key.

	Parameters
	----------
	matches : dict
	    ``{(iset, ifov): [path, ...]}``.
	shard : (int, int), optional
	    ``(index, count)`` as in :func:`shard_keys`. Default: keep everything, just sorted.
	"""
	index, count = shard or (0, 1)
	return {key: matches[key] for key in shard_keys(matches, index, count)}


class FolderFilter:
	"""Select round folders and fields of view from a ``first:last`` range.

	Folder names are split with ``regex_pattern`` into five groups: prefix, round number, middle
	part, set number and suffix. A folder is selected if its prefix and middle part match those
	of the range (a middle part of ``*`` matches anything) and its round and set numbers lie
	between those of ``first`` and ``last``.

	Parameters
	----------
	hyb_range : str
	    ``first:last``, each written like a round folder name, e.g. ``H1_A_set1:H16_A_set1``.
	regex_pattern : str
	    Regular expression with five capture groups.
	fov_min, fov_max : float
	    Only fields of view whose number is in this range (inclusive) are returned.
	"""
	def __init__(self, hyb_range: str, regex_pattern: str, fov_min: float, fov_max: float):
		self.hyb_range = hyb_range
		self.regex = re.compile(regex_pattern)
		self.start_pattern, self.end_pattern = self.hyb_range.split(':')	
		self.fov_min = fov_min
		self.fov_max = fov_max
		
		# Parse start and end patterns
		self.start_parts = self._parse_pattern(self.start_pattern)
		self.end_parts = self._parse_pattern(self.end_pattern)
		
	def _parse_pattern(self, haystack: str) -> Optional[Tuple]:
		"""Parse a pattern using the regex to extract components"""
		match = self.regex.match(haystack)
		if match:
			return match.groups()
		return None

	def _undo_regex(self, groups: Union[Tuple, re.Match, None]) -> Optional[str]:
		"""
		Reconstruct a string from captured groups based on the stored regex pattern.
		If `groups` is a re.Match it will use match.groups().
		If you pass fewer groups than the regex has, reconstruction stops right
		after the last provided group (no extra literal text appended).
		"""
		if groups is None:
			return None

		# Accept either a tuple-of-groups or a Match object.
		if hasattr(groups, "groups"):
			groups = groups.groups()
		groups = tuple(groups)

		pattern = self.regex.pattern
		parts = re.split(r'\([^)]*\)', pattern)  # literal chunks between capture groups

		rebuilt = []
		for i, g in enumerate(groups):
			literal = parts[i] if i < len(parts) else ''
			rebuilt.append(literal)
			rebuilt.append('' if g is None else str(g))

		# Append the trailing literal only if all groups were provided.
		if len(groups) == len(parts) - 1:
			rebuilt.append(parts[len(groups)])

		return ''.join(rebuilt)

	def _compare_patterns(self, file_parts: Tuple, start_parts: Tuple, end_parts: Tuple) -> bool:
		"""
		Compare if file_parts falls within the range defined by start_parts and end_parts
		Groups: (prefix, number, middle, set_number, suffix)
		"""
		if not all([file_parts, start_parts, end_parts]):
			return False
			
		# Extract components
		file_prefix, file_num, file_middle, file_set, file_suffix = file_parts
		start_prefix, start_num, start_middle, start_set, start_suffix = start_parts
		end_prefix, end_num, end_middle, end_set, end_suffix = end_parts
		
		# Convert to integers for comparison
		file_num = int(file_num)
		file_set = int(file_set)
		start_num = int(start_num)
		start_set = int(start_set)
		end_num = int(end_num)
		end_set = int(end_set)
	
		# Check if middle part matches (e.g., 'MER')
		if start_middle == '*':
			pass
		elif file_middle != start_middle or file_middle != end_middle:
			return False
			
		# Check if prefix matches
		if file_prefix != start_prefix or file_prefix != end_prefix:
			return False
			
		num_in_range = start_num <= file_num <= end_num
		set_in_range = start_set <= file_set <= end_set
		
		return num_in_range and set_in_range
	
	def isin(self, text: str) -> bool:
		"""Check if a single text/filename falls within the specified range"""
		file_parts = self._parse_pattern(text)
		if not file_parts:
			return False
		return self._compare_patterns(file_parts, self.start_parts, self.end_parts)
	
	def summary(self, matches):
		"""Table of how many fields of view were found per set and round."""
		counts = defaultdict(lambda: defaultdict(int))
		rounds = set()
		for iset, ifov in matches:
			for path in matches[(iset, ifov)]:
				base = os.path.basename(os.path.dirname(path))
				parts = list(self._parse_pattern(base))[:3]
				undone = self._undo_regex(parts)
				counts[iset][undone] += 1
				rounds.add(undone)
		if not rounds:
			return 'No matching images found.'
		rounds = sorted(rounds)
		width = max(len(r) for r in rounds) + 1
		lines = ['SUMMARY'.center(len(rounds) * width + width, '-')]
		lines.append('set'.ljust(width) + ''.join(r.rjust(width) for r in rounds))
		for iset in counts:
			lines.append(str(iset).ljust(width) + ''.join(str(counts[iset][r]).rjust(width) for r in rounds))
		lines.append(''.center(len(rounds) * width + width, '-'))
		return '\n'.join(lines)

	def get_matches(self, folders):
		"""Find the selected fields of view below ``folders``.

		Returns
		-------
		dict
		    ``{(iset, ifov): [path, ...]}``, where each path is a ``.zarr`` entry of a selected round
		    folder. The list holds one path per round.
		"""
		matches = dict()
		for root in folders:
			if not os.path.exists(root):
				continue
			try:
				with os.scandir(root) as entries:
					for sub in entries:
						if sub.is_dir(follow_symlinks=False) and self.isin(sub.name):
							try:
								with os.scandir(sub.path) as items:
									# we might need other ways to determine set
									iset = get_iset(str(sub.name))
									for item in items:
										if item.is_dir(follow_symlinks=False) and '.zarr' in item.name:
											ifov = get_ifov(str(item.name))
											if self.fov_min <= ifov <= self.fov_max:
												matches.setdefault((iset,ifov), []).append(item.path)

							except PermissionError:
								continue
			except PermissionError:
				continue
		return matches
