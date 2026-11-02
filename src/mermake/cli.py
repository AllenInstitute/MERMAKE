"""Command line interface for mermake."""
import os
import re
import sys
import argparse
from importlib import resources

from . import launch
from .config import Config, ConfigError


def example_text():
	"""The packaged example settings file."""
	return resources.files("mermake").joinpath("example.toml").read_text()


def is_valid_file(path):
	"""Argparse type: load and validate a TOML settings file and return a ``Config``."""
	if not os.path.exists(path):
		raise argparse.ArgumentTypeError(f"{path} does not exist.")
	try:
		return Config.from_file(path)
	except ConfigError as e:
		raise argparse.ArgumentTypeError(f"invalid settings in {e}")


class CustomArgumentParser(argparse.ArgumentParser):
	"""An argument parser that prints a minimal example settings file when no settings file is given.
	"""
	def error(self, message):
		# Customizing the error message
		if "the following arguments are required: settings" in message:
			message = message.replace("settings", "settings.toml")
			message += '\nA minimal settings file is shown below (`mermake --print-example` prints it)\n\n'
			message += example_text()
		super().error(message)


def parse_shard(text):
	"""Argparse type: ``I/N`` becomes ``(I, N)``, the share ``I`` (from 0) of ``N`` shares."""
	m = re.fullmatch(r"(\d+)/(\d+)", text.strip())
	if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
		raise argparse.ArgumentTypeError(f"{text!r} is not I/N with 0 <= I < N, for example 0/4")
	return int(m.group(1)), int(m.group(2))


def positive_int(text):
	"""Argparse type: an integer of at least 1."""
	try:
		value = int(text)
	except ValueError:
		value = 0
	if value < 1:
		raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of at least 1")
	return value


def build_parser(prog=None):
	"""Build the argument parser."""
	prog = prog or (sys.argv[0] if sys.argv[0] else "mermake")
	usage = f'{prog} [-opt1, [-opt2, ...]] settings.toml'
	parser = CustomArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser.add_argument('settings', type=is_valid_file, nargs='?', help='settings file')
	parser.add_argument('-g', '--gpu', type=int, default=None,
		help="The gpu to use, counted among the visible ones. By default the environment is left "
		     "alone, so a scheduler's choice (CUDA_VISIBLE_DEVICES) is kept and the first visible gpu is used")
	parser.add_argument('--gpus', default=None, metavar='LIST',
		help="Run one worker per gpu on this machine: 'all' or comma separated gpu numbers, for example 0,1,2,3. "
		     "Each worker takes its own share of the fields of view")
	parser.add_argument('--shard', type=parse_shard, default=None, metavar='I/N',
		help="Process only share I (from 0) of N of the fields of view. Run one process per share, for "
		     "example one per gpu or one per SLURM array task. Every round of a field of view stays together")
	parser.add_argument('--prefetch', type=positive_int, default=5, metavar='N',
		help="Images read ahead of the gpu [5]. Each one takes the memory of a whole image (about 2.3 GiB "
		     "for 4 channels of 40 x 2800 x 2800), and a run holds a few more than this")
	parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	parser.add_argument('--print-example', action="store_true",
		help="Print a minimal example settings file and exit")
	parser.add_argument('--print-config', action="store_true",
		help="Print every setting with its description and the value that would be used, then exit")
	parser.add_argument('--dry-run', action="store_true",
		help="Validate the settings and list the images that would be processed, then exit")
	parser.add_argument('--suggest-threshold', action="store_true",
		help="Sample a few images, count the spots found at every threshold and suggest values "
		     "for the threshold settings, then exit. Needs a GPU")
	parser.add_argument('--sample-images', type=int, default=3,
		help="Number of random images --suggest-threshold uses [3]")
	parser.add_argument('--max-false-positives', type=float, default=0.3,
		help="Largest estimated fraction of false positives --suggest-threshold accepts, "
		     "larger values give lower thresholds [0.3]")
	return parser


def dry_run(config, shard=None):
	"""Print what a run would process. Needs no GPU."""
	from .io.naming import FolderFilter, shard_matches
	paths = config.paths
	fov_min, fov_max = (-float('inf'), float('inf'))
	if paths.fov_range:
		fov_min, fov_max = map(float, paths.fov_range.split(':'))
	folder_filter = FolderFilter(paths.hyb_range, paths.regex, fov_min, fov_max)
	matches = folder_filter.get_matches(paths.hyb_folders)
	total = len(matches)
	matches = shard_matches(matches, shard)
	if shard:
		print(f"shard {shard[0]}/{shard[1]}: {len(matches)} of {total} fov groups")
	print(folder_filter.summary(matches))
	print(f"{sum(map(len, matches.values()))} images in {len(matches)} (set, fov) groups")
	print(f"output folder: {paths.output_folder}")


def main(argv=None):
	"""Entry point for the ``mermake`` command."""
	argv = sys.argv[1:] if argv is None else list(argv)
	parser = build_parser()
	args = parser.parse_args(argv)

	if args.print_example:
		print(example_text(), end="")
		return
	if args.settings is None:
		parser.error("the following arguments are required: settings")
	if args.print_config:
		print(args.settings.to_toml())
		return
	if args.dry_run:
		dry_run(args.settings, args.shard)
		return

	if args.gpus is not None:
		if args.gpu is not None:
			parser.error("--gpus and --gpu cannot be combined")
		try:
			devices = launch.parse_gpus(args.gpus)
		except ValueError as e:
			parser.error(str(e))
		return launch.run_workers(launch.worker_jobs(argv, devices, shard=args.shard))

	# set this before cupy is imported to make sure to capture the correct gpu
	if args.gpu is not None:
		try:
			os.environ["CUDA_VISIBLE_DEVICES"] = launch.resolve_device(args.gpu)
		except ValueError as e:
			parser.error(str(e))
	from mermake.gpu.runtime import check_cuda
	check_cuda()
	if args.suggest_threshold:
		from mermake.threshold import suggest
		suggest(args, n_images=args.sample_images, max_fp=args.max_false_positives)
		return
	from mermake.pipeline import run
	run(args)
