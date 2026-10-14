import os
from pathlib import Path
import queue
import threading
from itertools import chain
from collections import defaultdict

import cupy as cp
from .. import __version__
from .readers import read_im, Container
from .naming import get_ih, get_ifov, get_iset, FolderFilter
from ..config import namespace_to_array

class Block(list):
	def add(self, image):
		ifov = image.ifov
		if not self or self.ifov() == ifov:
			if self:
				del self[-1].data
			self.append(image)
			return True
		else:
			return False
	def drift_file(self):
		return self[0].drift_file
	def iset(self):
		return self[0].iset
	def ifov(self):
		return self[0].ifov
	def __repr__(self):
		paths = [image.path for image in self]
		return f"Block({paths})"
	
class ImageQueue:
	__version__ = __version__
	def __init__(self, args, prefetch_count=5):
		self.args = args
		self.args_array = namespace_to_array(self.args.settings)
		self.__dict__.update(vars(args.paths))

		os.makedirs(self.output_folder, exist_ok = True)
		
		fov_min, fov_max = (-float('inf'), float('inf'))
		if hasattr(self, "fov_range"):
			fov_min, fov_max = map(float, self.fov_range.split(':'))
		
		filtered = FolderFilter(self.hyb_range, self.regex, fov_min, fov_max)
		matches = filtered.get_matches(self.hyb_folders)

		# do output summary, eventually replace with terminal gui
		counts = defaultdict(lambda: defaultdict(int))
		rounds = dict()
		for iset,ifov in matches:
			for path in matches[(iset,ifov)]:
				base = os.path.basename(os.path.dirname(path))
				parts = list(filtered._parse_pattern(base))
				parts = parts[:3]
				undone = filtered._undo_regex(parts)
				counts[iset][undone] += 1
				rounds[undone] = True
		rounds = sorted(rounds)
		width = max(len(r) for r in rounds) + 1
		summary = ''
		summary += 'SUMMARY'.center(len(rounds) * width + width, '-')	
		summary += '\n'
		summary += 'set'.ljust(width)
		for undone in rounds:
			summary += undone.rjust(width)
		summary += '\n'
		for iset in counts:
			summary += str(iset).ljust(width)
			for undone in rounds:
				summary += str(counts[iset][undone]).rjust(width)
			summary += '\n'
		summary += ''.center(len(rounds) * width + width, '-')
		self.summary = summary
			
		# Peek at first image to set shape/dtype
		first_image = None
		for path in chain.from_iterable(matches.values()):
			try:
				first_image = read_im(path)
				break
			except Exception:
				continue
		if first_image is None:
			raise RuntimeError("No valid images found.")
		self.shape = first_image.shape
		self.dtype = first_image.dtype
	
		ordered = []
		for key in matches:
			ordered.extend(self.hsorted(matches[key]))

		self.files = iter(ordered)

		# Start worker thread(s)
		self.queue = queue.Queue(maxsize=prefetch_count)
		self.stop_event = threading.Event()
		self.thread = threading.Thread(target=self._worker, daemon=True)
		self.thread.start()

	def hsorted(self, files):
		return sorted(files, key=lambda f: get_ih(os.path.dirname(f)))

	def containerize(self, path):
		# everythign in this method is done async and prefetched

		container = Container(path)

		# check if the fov has been fitted
		# we should test whether the load worked!!!!!!!!!!
		for icol in range(self.shape[0] - 1):
			filename = self.get_name(path, icol)
			filepath = os.path.join(self.output_folder, filename)
			# only compute if the image has not been processed
			if not os.path.exists(filepath):
				container[icol].compute()
			else:
				Xh = cp.load(filepath)['Xh']
				setattr(container, f'col{icol}', Xh)
		icol = -1
		filename = self.get_name(path, icol)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			container[icol].compute()
		else:
			container.Xh_plus = cp.load(filepath)['Xh_plus']
			container.Xh_minus = cp.load(filepath)['Xh_minus']
		# attach drift file to container
		iset = get_iset(path)
		ifov = get_ifov(path)
		#filename = self.drift_save.format(ifov=ifov, iset=iset)
		filename = self.get_name(path)
		filepath = os.path.join(self.output_folder, filename)
		container.drift_file = filepath
		if os.path.exists(filepath):
			drifts, files, ref, ref_path = cp.load(filepath, allow_pickle=True)
			container.drifts = drifts
			container.drift_files = files
			container.drift_ref = ref
			container.drift_ref_path = ref_path
		container.iset = iset
		container.ifov = ifov
		return container

	def _worker(self):
		"""Continuously read images and put them in the queue."""
		for path in self.files:
			if self.stop_event.is_set():
				break
			try:
				container = self.containerize(path)
				self.queue.put(container)
			except Exception as e:
				print(f"Warning: failed to read {path}: {e}")
				#dummy = lambda : None
				#dummy.path = path
				#self.queue.put(dummy)
				#self.queue.put(False)
				continue
		# Signal no more images
		container.last = True
		self.queue.put(None)

	def __iter__(self):
		return self

	def __next__(self):
		img = self.queue.get()
		if img is None:
			raise StopIteration
		return img
	
	'''
		# this code is so we can eventually add a --watch parameter so mermake keeps
		# runnning after processing all fovs and then processes any new fovs that appear
		# If we reach here, there are no more images in the current batch
		if False:
			# In watch mode, look for new files
			import time
			time.sleep(60)
			
			# Find any new files
			new_matches = self._find_matching_files()
			# Filter to only files we haven't processed yet
			new_matches = [f for f in new_matches if f not in self.processed_files]
			
			if new_matches:
				# New files found!
				new_matches.sort()
				self.matches = new_matches
				self.files = iter(self.matches)
				self.processed_files.update(new_matches)  # Mark as seen
				
				# Prefetch the first new image
				self._prefetch_next_image()
				
				# Try again to get the next image
				return self.__next__()
			else:
				# No new files yet, but we'll keep watching
				return self.__next__()

		self.close()
		raise StopIteration
	'''

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

	'''
	def close(self):
		self.stop_event.set()
		self.thread.join()
	'''
	def close(self):
		self.stop_event.set()
    
		# Drain the queue so the worker thread can finish
		try:
			while True:
				self.queue.get_nowait()
		except queue.Empty:
			pass
    
		# Use a timeout to avoid hanging indefinitely
		self.thread.join(timeout=2.0)
    
		if self.thread.is_alive():
			print("Warning: worker thread did not terminate", flush=True)

	def get_name(self, path, icol=None):
		path_obj = Path(path)
		fov = path_obj.stem	        # the filename without extension
		tag = path_obj.parent.name  # the parent directory name
		ifov = get_ifov(path)
		iset = get_iset(path)
		if icol is None:
			return self.drift_save.format(ifov=ifov, fov=fov, iset=iset)
		elif icol == -1:
			return self.dapi_save.format(fov=fov, tag=tag)
		else:
			return self.hyb_save.format(fov=fov, tag=tag, icol=icol)


	def _is_fitted(self, path):
		for icol in range(self.shape[0] - 1):
			filename = self.get_name(path, icol)
			filepath = os.path.join(self.output_folder, filename)
			if not os.path.exists(filepath):
				return False
		filename = self.get_name(path, -1)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			return False
		return True

	def save_xfits(self, image, icol=-1, attempt=1, max_attempts=3):
		
		# determine filename and data based on icol
		if icol == -1:
			data = {'Xh_plus': image.Xh_plus,'Xh_minus': image.Xh_minus,}
		else:
			Xh = getattr(image, f'col{icol}')
			data = {'Xh': Xh}
		
		# save if file doesn't exist
		path = image.path
		filename = self.get_name(path, icol)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			cp.savez_compressed(filepath, version=__version__, args=self.args_array, **data)
			#  Optional integrity check after saving
			# this seems to greatly slow everything down
			#try:
			#	with np.load(filepath) as dat:
			#		_ = dat["Xh"].shape  # Try accessing a key
			#except Exception as e:
			#	os.remove(filepath)
			#	if attempt < max_attempts:
			#		return self.save_hyb(path, icol, Xhf, attempt=attempt+1, max_attempts=max_attempts)
			#	else:
			#		raise RuntimeError(f"Failed saving xfit file after {max_attempts} attempts: {filepath}")
