"""Choose GPUs and run one mermake worker per GPU on this machine.

One mermake process drives one GPU. To use several GPUs of a machine, run one process per GPU,
each on its own share of the fields of view (``--shard``). ``mermake --gpus 0,1,2,3 settings.toml``
does that. Under SLURM or any other scheduler that sets ``CUDA_VISIBLE_DEVICES``, device numbers
count among the devices the job can see.

This module must not import cupy: ``CUDA_VISIBLE_DEVICES`` has to be set before cupy is imported.
"""
import os
import signal
import subprocess
import sys


def _split(value):
	return [d.strip() for d in value.split(",") if d.strip()]


def resolve_device(index, environ=None):
	"""The ``CUDA_VISIBLE_DEVICES`` entry for GPU number ``index``.

	If the environment already restricts the visible devices (as SLURM does), ``index`` counts
	among them, so ``--gpu 1`` is the second device the job was given. Otherwise it is the
	physical device number.

	Raises
	------
	ValueError
	    If fewer than ``index + 1`` devices are visible.
	"""
	environ = os.environ if environ is None else environ
	value = environ.get("CUDA_VISIBLE_DEVICES")
	if value is None:
		return str(index)
	devices = _split(value)
	if not 0 <= index < len(devices):
		raise ValueError(f"GPU {index} requested but only {len(devices)} are visible "
		                 f"(CUDA_VISIBLE_DEVICES={value!r})")
	return devices[index]


def query_gpu_indices():
	"""Physical indices of the GPUs on this machine, from ``nvidia-smi``. Empty if it cannot be run."""
	try:
		out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
		                     capture_output=True, text=True, timeout=30, check=True).stdout
	except (OSError, subprocess.SubprocessError):
		return []
	return _split(out.replace("\n", ","))


def parse_gpus(spec, environ=None, query=query_gpu_indices):
	"""The ``CUDA_VISIBLE_DEVICES`` entry of each worker, from the ``--gpus`` value.

	Parameters
	----------
	spec : str
	    ``all`` for every visible GPU, or comma separated GPU numbers such as ``0,1,3`` (counted
	    like ``--gpu``). A number can be repeated, which runs several workers on one GPU.
	environ : mapping, optional
	    Default ``os.environ``.
	query : callable
	    Lists the GPUs of the machine when ``CUDA_VISIBLE_DEVICES`` is not set.

	Raises
	------
	ValueError
	    If the value is malformed or no GPU can be found.
	"""
	environ = os.environ if environ is None else environ
	if spec == "all":
		value = environ.get("CUDA_VISIBLE_DEVICES")
		devices = _split(value) if value is not None else query()
		if not devices:
			raise ValueError("--gpus all found no GPU; list them instead, for example --gpus 0,1")
		return devices
	try:
		indices = [int(part) for part in spec.split(",")]
	except ValueError:
		raise ValueError(f"--gpus must be 'all' or comma separated GPU numbers, got {spec!r}")
	return [resolve_device(i, environ) for i in indices]


def strip_option(argv, name):
	"""``argv`` without the option ``name`` and its value (``--name value`` or ``--name=value``)."""
	out = []
	skip = False
	for token in argv:
		if skip:
			skip = False
		elif token == name:
			skip = True
		elif not token.startswith(name + "="):
			out.append(token)
	return out


def worker_jobs(argv, devices, shard=None, environ=None, cpus=None):
	"""The command and environment of every worker.

	Worker ``k`` runs with only ``devices[k]`` visible and takes share ``index * len(devices) + k``
	of ``count * len(devices)`` shares of the fields of view, where ``shard = (index, count)`` is the
	share of the whole run that this machine has (for example one SLURM array task per node).

	Parameters
	----------
	argv : list of str
	    The command line of the launcher, without the program name.
	devices : list of str
	    One ``CUDA_VISIBLE_DEVICES`` entry per worker.
	shard : (int, int), optional
	    This machine's share. Default: everything.
	environ : mapping, optional
	    Default ``os.environ``.
	cpus : int, optional
	    Cores of this machine, to split between the workers. Default: the cores this process may use.

	Returns
	-------
	list of (list of str, dict)
	    ``(command, environment)`` per worker.
	"""
	environ = dict(os.environ if environ is None else environ)
	index, count = shard or (0, 1)
	n = len(devices)
	if cpus is None:
		cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
	# without this every worker would start a reader thread pool as large as the whole machine
	environ.setdefault("DASK_NUM_WORKERS", str(max(1, cpus // n)))
	base = strip_option(strip_option(argv, "--gpus"), "--shard")
	jobs = []
	for k, device in enumerate(devices):
		cmd = [sys.executable, "-m", "mermake", *base, "--shard", f"{index * n + k}/{count * n}"]
		jobs.append((cmd, {**environ, "CUDA_VISIBLE_DEVICES": device}))
	return jobs


def run_workers(jobs):
	"""Run the workers at the same time and wait for all of them.

	SIGTERM (``scancel``, the time limit) is passed on to the workers. A worker that fails does
	not stop the others, since their fields of view are independent, but it makes the exit
	status non-zero.

	Returns
	-------
	int
	    0 if every worker succeeded, otherwise the exit status of the first one that did not.
	"""
	procs = [subprocess.Popen(cmd, env=env) for cmd, env in jobs]

	def forward(signum, frame):
		for proc in procs:
			if proc.poll() is None:
				proc.send_signal(signum)

	previous = signal.signal(signal.SIGTERM, forward)
	try:
		codes = [proc.wait() for proc in procs]
	except KeyboardInterrupt:
		# the terminal sent SIGINT to the workers as well, give them time to stop
		for proc in procs:
			try:
				proc.wait(timeout=30)
			except subprocess.TimeoutExpired:
				proc.kill()
		return 130
	finally:
		signal.signal(signal.SIGTERM, previous)
	for code in codes:
		if code:
			return code if code > 0 else 128 - code
	return 0
