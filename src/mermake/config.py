import glob
import os
import argparse
from argparse import Namespace
from types import SimpleNamespace

from .io.metadata import get_xml_field


class Config:
	def __init__(self, args):
		self.args = args
		self.some_data = 'foo'

def dict_to_namespace(d):
	"""Recursively convert dictionary into SimpleNamespace."""
	for key, value in d.items():
		if isinstance(value, dict):
			value = dict_to_namespace(value)
		elif isinstance(value, list):
			value = [dict_to_namespace(i) if isinstance(i, dict) else i for i in value]
		d[key] = value
	return SimpleNamespace(**d)
def namespace_to_dict(obj):
	"""Recursively convert namespace objects to dictionaries"""
	if isinstance(obj, argparse.Namespace):
		return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
	elif isinstance(obj, list):
		return [namespace_to_dict(item) for item in obj]
	elif isinstance(obj, dict):
		return {k: namespace_to_dict(v) for k, v in obj.items()}
	else:
		return obj

def namespace_to_array(obj, prefix=''):
	"""
	Recursively convert Namespace or dict to list of (block, key, value) tuples.
	prefix is the accumulated parent keys joined by dots.
	"""
	rows = []
	if isinstance(obj, (Namespace, SimpleNamespace)):
		obj = vars(obj)
	if isinstance(obj, dict):
		for k, v in obj.items():
			full_key = f"{prefix}.{k}" if prefix else k
			if isinstance(v, (Namespace, SimpleNamespace, dict)):
				rows.extend(namespace_to_array(v, prefix=full_key))
			else:
				rows.append((prefix, k, str(v)))
	else:
		# For other types just append
		rows.append((prefix, '', str(obj)))
	return rows


def count_bits(args):
	group = args.config['codebooks'][0]
	with open(group['codebook_path'], 'r') as fp:
		return next(fp).rstrip().count('bit')

def count_colors(args):
	batch = args.batch
	sset = next(iter(batch))
	fov = next(iter(batch[sset]))
	hyb = next(iter(batch[sset][fov]))
	dic = batch[sset][fov][hyb]

	path = dic['zarr']
	dirname = os.path.dirname(path)
	basename = os.path.basename(path)
	file = glob.glob(os.path.join(dirname,'*' + basename + '.xml'))[0]
	colors = get_xml_field(file, 'z_offsets').split(':')[-1]
	return int(colors)

def count_hybs(args):
	bits = count_bits(args)
	colors = count_colors(args) - 1
	num = bits / colors
	print(num)
	print(args)
