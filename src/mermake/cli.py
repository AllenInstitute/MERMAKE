"""Command line interface for mermake."""
import os
import sys
import argparse
# Try to import the appropriate TOML library
if sys.version_info >= (3, 11):
	import tomllib  # Python 3.11+ standard library
else:
	import tomli as tomllib  # Backport for older Python versions


def is_valid_file(path):
	"""Argparse type: load a TOML settings file and return the raw dict."""
	if not os.path.exists(path):
		raise argparse.ArgumentTypeError(f"{path} does not exist.")
	try:
		with open(path, "rb") as f:
			return tomllib.load(f)  # Return raw dict
	except Exception as e:
		raise argparse.ArgumentTypeError(f"Error loading TOML file {path}: {e}")

toml_text = r"""
        [paths]
        codebook = 'codebook.csv'
        psf_file = 'psf_file.npy'
        flat_field_tag = 'Scope_'
        hyb_range = 'H1_*_set1:H16_*_set2'
        hyb_folders = ['experiment_folder']
        output_folder = 'output'
        #---------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------#
        #           you probably dont have to change any of the settings below                  #
        #---------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------#
        hyb_save = '{fov}--{tag}--col{icol}.npz'
        dapi_save = '{fov}--{tag}--dapiFeatures.npz'
		drift_save = 'drift_Conv_zscan__{ifov:0>3}--_set{iset}.pkl'
        regex = '''([A-z]+)(\d+)_(.+)_set(\d+)(.*)''' #use triple quotes to avoid double escape
        [hybs]
        tile_size = 500
        overlap = 89
        beta = 0.0001
        threshold = 3600
        blur_radius = 30
        delta = 1
        delta_fit = 3
        sigmaZ = 1
        sigmaXY = 1.5
        
        [dapi]
        tile_size = 300
        overlap = 89
        beta = 0.01
        threshold = 3.0
        blur_radius = 50
        delta = 5
        delta_fit = 5
        sigmaZ = 1
        sigmaXY = 1.5"""

class CustomArgumentParser(argparse.ArgumentParser):
	def error(self, message):
		# Customizing the error message
		if "the following arguments are required: settings" in message:
			message = message.replace("settings", "settings.toml")
		message += '\n'
		message += 'The format for the toml file is shown below'
		message += '\n'
		message += toml_text
		super().error(message)


def build_parser(prog=None):
	"""Build the argument parser."""
	prog = prog or (sys.argv[0] if sys.argv[0] else "mermake")
	usage = f'{prog} [-opt1, [-opt2, ...]] settings.toml'
	parser = CustomArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser.add_argument('settings', type=is_valid_file, help='settings file')
	parser.add_argument('-g', '--gpu', type=int, default=0, help="The gpu to use [0]")
	parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	return parser


def main(argv=None):
	"""Entry point for the ``mermake`` command."""
	args = build_parser().parse_args(argv)

	# set this before cupy is imported to make sure to capture the correct gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	from mermake.gpu.runtime import check_cuda
	check_cuda()
	from mermake.pipeline import run
	run(args)
