"""How fast can this machine read the images?

    uv run --no-sync python scripts/bench_read.py <fov.zarr> [<fov.zarr> ...] [--threads 4]

Reads every chunk file of each zarr (the compressed bytes, nothing is decoded) from all the given
paths at the same time and reports the throughput. Run it from one process, then from several at
once on different fields of view (``... & ... & wait``): if the total does not grow, the storage is
the limit of the whole machine, however many GPUs it has. mermake needs about 0.8 GB per image for
4 x 40 x 2800 x 2800 lz4 data, so a throughput of 140 MB/s allows one image every 6 seconds.
Files that were read recently may come from a cache, so use fields of view nothing has read.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("zarr", nargs="+", help="the .zarr entries of fields of view")
	parser.add_argument("--threads", type=int, default=4, help="reader threads [4]")
	args = parser.parse_args(argv)

	files = []
	for path in args.zarr:
		# the chunks of the array itself, or of <round folder>/<fov>/data for the older layout
		found = sorted(glob.glob(os.path.join(path, "*.0.0")) + glob.glob(os.path.join(path, "data", "*.0.0")))
		if not found:
			parser.error(f"no chunk files in {path}")
		files += found

	def read(name):
		with open(name, "rb") as f:
			return len(f.read())

	start = time.perf_counter()
	with ThreadPoolExecutor(args.threads) as pool:
		size = sum(pool.map(read, files))
	seconds = time.perf_counter() - start
	print(f"{len(files)} files, {size / 1e6:.0f} MB in {seconds:.2f} s: {size / 1e6 / seconds:.0f} MB/s "
	      f"({len(args.zarr)} zarr, {args.threads} threads)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
