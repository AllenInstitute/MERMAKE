"""The mermake processing pipeline: read, deconvolve, find maxima, align, save."""
import sys
import concurrent.futures
from types import SimpleNamespace

import numpy as np
import cupy as cp
import dask.array as da

from .align import drift, drift_save
from .config import dict_to_namespace
from .deconvolver import Deconvolver
from .io.queue import ImageQueue, Block
from .io.readers import load_flats
from .maxima import find_local_maxima

_last_len = 0


def print_clean(msg):
	"""Overwrite the current terminal line with ``msg``."""
	global _last_len
	# Clear previous line
	sys.stdout.write('\r' + ' ' * _last_len + '\r')
	sys.stdout.write(msg)
	sys.stdout.flush()
	_last_len = len(msg)  # Save new message length


def build_config(args):
	"""Attach each top-level section of the settings dict to ``args`` as a namespace."""
	for key, value in vars(dict_to_namespace(args.settings)).items():
		setattr(args, key, value)
	return args


def build_deconvolvers(psfs, chan_shape, args):
	"""Load the PSFs onto the GPU for the hyb and dapi channels.

	These can be very large objects in gpu ram, adjust accordingly to suit gpu specs.
	"""
	# THIS SHOULD PROBABLY BE SET TO THE ZDEPTH OF THE PSF AND NOT THE DEPTH OF THE IMAGE
	zpad = chan_shape[0] - 1 # this needs to be about the same size as the input z depth
	hybs = Deconvolver(psfs, chan_shape, zpad=zpad, **vars(args.hybs))
	# shrink the zpad to limit the loaded psfs in ram since dapi isnt deconvolved as strongly
	# or you could just use a single psf, ie (0,1500,1500)
	dapi = Deconvolver(psfs, chan_shape, zpad=zpad//2, **vars(args.dapi))
	return SimpleNamespace(hybs=hybs, dapi=dapi)


def process_dapi(image, queue, ctx):
	"""Find the dapi maxima (positive and negative) and save them."""
	icol = -1
	data = image[icol].data
	flat = ctx.flats[icol]
	if isinstance(data, da.Array):
		return
	args, chan, buffer = ctx.args, ctx.chan, ctx.buffer
	chan.set(data)
	# Deconvolve in-place into the buffer
	ctx.deconvolver.dapi.apply(chan, flat_field=flat, output=buffer, **vars(args.dapi))
	# the dapi channel is further normalized by the stdev
	std_val = float(cp.asnumpy(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size)))
	cp.divide(buffer, std_val, out=buffer)
	Xh_plus = find_local_maxima(buffer, raw=chan, **vars(args.dapi))
	cp.multiply(buffer, -1, out=buffer)
	Xh_minus = find_local_maxima(buffer, raw=chan, **vars(args.dapi))
	# save the data
	setattr(image, 'Xh_plus', Xh_plus)
	setattr(image, 'Xh_minus', Xh_minus)
	ctx.executor.submit(queue.save_xfits, image, icol)


def find_hyb_maxima(chan, flat, ctx):
	"""Deconvolve ``chan`` patch by patch and return the maxima kept from each patch.

	Returns a ``(n, 8)`` float32 cupy array in full-image coordinates.
	"""
	args = ctx.args
	patch_overlap = args.hybs.patch_overlap
	# there is probably a better way to do the Xh stacking
	Xhf = []
	for x, y, patch, raw in ctx.deconvolver.hybs.patch_wise(chan, flat, **vars(args.hybs)):
		Xh = find_local_maxima(patch, raw=raw, **vars(args.hybs))
		keep = cp.all((Xh[:,1:3] >= patch_overlap) & (Xh[:,1:3] < cp.array([patch.shape[1] - patch_overlap, patch.shape[2] - patch_overlap])), axis=-1)
		if cp.any(keep):
			Xh = Xh[keep]
			Xh[:,1] += x - patch_overlap
			Xh[:,2] += y - patch_overlap
			Xhf.append(Xh)
	if Xhf:
		return cp.vstack(Xhf)
	return cp.zeros([0,8], dtype=cp.float32)


def process_hyb_channel(image, icol, queue, ctx):
	"""Process one color channel of ``image`` and save its maxima."""
	data = image[icol].data
	if isinstance(data, da.Array):
		return
	ctx.chan.set(data)
	Xhf = find_hyb_maxima(ctx.chan, ctx.flats[icol], ctx)
	setattr(image, f'col{icol}', Xhf)
	ctx.executor.submit(queue.save_xfits, image, icol)


def finish_block(image, ctx):
	"""Close out a block of images for a fov: do the drift, then start a new block."""
	block = ctx.block
	if not block.add(image) or hasattr(image, 'last'):
		# do the drift
		if result := drift(block):
			ctx.executor.submit(drift_save, *result)
			del result
		# do the decoding
		# here eventually

		# now add the image to the newly cleared block
		block.clear()
		block.add(image)


def process_image(image, queue, ctx):
	"""Run every stage on one image."""
	print('running on:', image.path, flush=True)
	process_dapi(image, queue, ctx)

	for icol in range(ctx.ncol - 1):
		process_hyb_channel(image, icol, queue, ctx)
	# this block of images for a fov is over
	finish_block(image, ctx)


def run(args):
	"""Run the pipeline described by the parsed command line ``args``."""
	build_config(args)
	psfs = np.load(args.paths.psf_file, allow_pickle=True)

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=9)

	print_clean('Finding input image files.')

	with ImageQueue(args) as queue:
		print_clean(queue.summary)
		# set some things based on input images
		ncol, *chan_shape = queue.shape

		# buffers to copy into
		buffer = cp.empty(chan_shape, dtype=cp.float32)
		chan = cp.empty(chan_shape, dtype=cp.uint16)
		flats = load_flats(shape=queue.shape, **vars(args.paths))

		print_clean('Loading PSFs into GPU ram.')
		deconvolver = build_deconvolvers(psfs, chan_shape, args)

		if args.check:
			from .viz import view_napari
			deconvolver.buffer = buffer
			deconvolver.flats = flats
			view_napari(queue, deconvolver, args)

		ctx = SimpleNamespace(
			args=args, ncol=ncol, buffer=buffer, chan=chan, flats=flats,
			deconvolver=deconvolver, executor=executor, block=Block(),
		)

		print_clean('Starting image processing.\n')
		for image in queue:
			process_image(image, queue, ctx)
			del image
		executor.shutdown(wait=True)
