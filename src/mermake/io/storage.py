"""Input paths on a local disk or behind a URL such as ``s3://bucket/prefix``.

A local path goes through ``os`` exactly as it always has. A path with ``://`` goes through fsspec:
``s3://`` needs s3fs, an optional dependency (``pip install 'mermake[s3]'``). Credentials come from
the usual AWS chain (environment variables, ``~/.aws``, the instance or container role), or from
``[paths] storage_options``, which is passed to fsspec as is.

Only the images are read this way. The PSF, the flat fields and the output folder stay local.
"""
import os

_options = {}
_read_threads = None
_filesystems = {}


def is_remote(path):
	"""True if ``path`` is a URL (``s3://...``) rather than a local path."""
	return isinstance(path, str) and "://" in path


def configure(options=None, read_threads=None):
	"""Set the fsspec options of remote paths and the read threads they must serve.

	``read_threads`` raises the connection pool of s3 (10 by default) so that every read thread
	gets a connection.
	"""
	global _options, _read_threads
	_options = dict(options or {})
	_read_threads = read_threads
	_filesystems.clear()


def _protocol(path):
	return path.split("://", 1)[0]


def storage_options(path):
	"""The fsspec options for ``path``, as ``dask.array.from_zarr`` takes them."""
	options = dict(_options)
	if _protocol(path) in ("s3", "s3a"):
		config = dict(options.get("config_kwargs") or {})
		config.setdefault("max_pool_connections", max(_read_threads or 0, 10))
		options["config_kwargs"] = config
	return options


def filesystem(path):
	"""The fsspec file system of a remote ``path``, made once per protocol."""
	protocol = _protocol(path)
	if protocol not in _filesystems:
		import fsspec
		try:
			_filesystems[protocol] = fsspec.filesystem(protocol, **storage_options(path))
		except ImportError as e:
			raise ImportError(f"reading {protocol}:// paths needs an extra package: {e} "
			                  f"For S3: pip install 'mermake[s3]'") from e
	return _filesystems[protocol]


def exists(path):
	if not is_remote(path):
		return os.path.exists(path)
	return filesystem(path).exists(path)


def isdir(path):
	if not is_remote(path):
		return os.path.isdir(path)
	return filesystem(path).isdir(path)


def listdir(path):
	"""``[(name, path, is_dir)]`` of the entries of the folder ``path``."""
	if not is_remote(path):
		with os.scandir(path) as entries:
			return [(e.name, e.path, e.is_dir(follow_symlinks=False)) for e in entries]
	root = path.rstrip("/")
	out = []
	for info in filesystem(path).ls(root, detail=True):
		name = info["name"].rstrip("/").rsplit("/", 1)[-1]
		out.append((name, f"{root}/{name}", info.get("type") == "directory"))
	return out


def read_text(path, encoding=None):
	if not is_remote(path):
		with open(path, "r", encoding=encoding) as f:
			return f.read()
	return filesystem(path).cat_file(path).decode(encoding or "utf-8")
