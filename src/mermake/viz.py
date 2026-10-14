import sys

import numpy as np


def view_napari(queue, deconvolver, args):
	"""Show the first image, its deconvolution and maxima in napari, then exit."""
	import cupy as cp
	from .maxima import find_local_maxima

	image = next(queue)
	buffer = deconvolver.buffer
	flats = deconvolver.flats
	import napari
	viewer = napari.Viewer()
	color = ['red','green','blue', 'white']
	ncol = queue.shape[0]
	for icol in range(ncol-1):
		chan = cp.asarray(image[icol].compute())
		flat = flats[icol]
		deconvolver.hybs.apply(chan, flat_field=flat, output=buffer, blur_radius=None)
		deco = cp.asnumpy(buffer)
		deconvolver.hybs.apply(chan, flat_field=flat, output=buffer, **vars(args.hybs))
		Xh = find_local_maxima(buffer, raw = chan, **vars(args.hybs))
		norm = cp.asnumpy(buffer)
		# Stack 3D images: original, deco, norm
		stacked = np.stack([cp.asnumpy(chan), deco, norm], axis=0)
		viewer.add_image(stacked, name=f"channel {icol}", colormap=color[icol], blending='additive')
		# Add corresponding points for this stack
		points = cp.asnumpy(Xh[:, :3])
		viewer.add_points(points, size=7, border_color=color[icol],face_color='transparent', opacity=0.6, name=f"maxima {icol}")
	icol += 1
	chan = cp.asarray(image[icol].compute())
	flat = flats[icol]
	deconvolver.dapi.apply(chan, flat_field=flat, output=buffer, blur_radius=None)
	deco = cp.asnumpy(buffer)
	deconvolver.dapi.apply(chan, flat_field=flat, output=buffer, **vars(args.dapi))
	std_val = float(cp.asnumpy(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size)))
	cp.divide(buffer, std_val, out=buffer)
	Xh_plus = find_local_maxima(buffer, raw = chan, **vars(args.dapi) )
	norm = cp.asnumpy(buffer)
	# Stack 3D images: original, deco, norm
	stacked = np.stack([cp.asnumpy(chan), deco, norm], axis=0)
	viewer.add_image(stacked, name="dapi",  colormap=color[icol], blending='additive')
	points = cp.asnumpy(Xh_plus[:, :3])
	viewer.add_points(points, size=11, border_color=color[icol], face_color='transparent', opacity=0.6, name=f"maxima dapi")
	napari.run()
	sys.exit()
