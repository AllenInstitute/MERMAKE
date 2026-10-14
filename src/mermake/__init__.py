import os
from importlib.metadata import version, PackageNotFoundError


# ---------------- Version ----------------
try:
	# Try to get version from installed package metadata
	__version__ = version("mermake")
except PackageNotFoundError:
	# Fallback: read version from a _VERSION file or hardcode for in-place runs
	version_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "VERSION")
	if os.path.exists(version_file):
		with open(version_file) as f:
			__version__ = f.read().strip()
	else:
		raise ValueError

