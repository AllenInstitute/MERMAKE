"""Command line interface for mermake."""
import os
import sys
import argparse
from importlib import resources

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
	def error(self, message):
		# Customizing the error message
		if "the following arguments are required: settings" in message:
			message = message.replace("settings", "settings.toml")
			message += '\nA minimal settings file is shown below (`mermake --print-example` prints it)\n\n'
			message += example_text()
		super().error(message)


def build_parser(prog=None):
	"""Build the argument parser."""
	prog = prog or (sys.argv[0] if sys.argv[0] else "mermake")
	usage = f'{prog} [-opt1, [-opt2, ...]] settings.toml'
	parser = CustomArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser.add_argument('settings', type=is_valid_file, nargs='?', help='settings file')
	parser.add_argument('-g', '--gpu', type=int, default=0, help="The gpu to use [0]")
	parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	parser.add_argument('--print-example', action="store_true",
		help="Print a minimal example settings file and exit")
	parser.add_argument('--print-config', action="store_true",
		help="Print every setting with its description and the value that would be used, then exit")
	parser.add_argument('--dry-run', action="store_true",
		help="Validate the settings and list the images that would be processed, then exit")
	return parser


def dry_run(config):
	"""Print what a run would process. Needs no GPU."""
	from .io.naming import FolderFilter
	paths = config.paths
	fov_min, fov_max = (-float('inf'), float('inf'))
	if paths.fov_range:
		fov_min, fov_max = map(float, paths.fov_range.split(':'))
	folder_filter = FolderFilter(paths.hyb_range, paths.regex, fov_min, fov_max)
	matches = folder_filter.get_matches(paths.hyb_folders)
	print(folder_filter.summary(matches))
	print(f"{sum(map(len, matches.values()))} images in {len(matches)} (set, fov) groups")
	print(f"output folder: {paths.output_folder}")


def main(argv=None):
	"""Entry point for the ``mermake`` command."""
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
		dry_run(args.settings)
		return

	# set this before cupy is imported to make sure to capture the correct gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	from mermake.gpu.runtime import check_cuda
	check_cuda()
	from mermake.pipeline import run
	run(args)
