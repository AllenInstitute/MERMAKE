# Using several GPUs and nodes

One `mermake` process drives one GPU, and works through its images one at a time. To go faster, run several processes, each on its own GPU and its own share of the fields of view. Nothing is shared between them, so the same pattern works for several GPUs in one machine and for several nodes of a SLURM cluster.

## How the work is split

Each process is given a **shard**: `--shard I/N` means share `I` (counting from 0) of `N`. The fields of view are sorted by set and number and dealt out like cards, so every process computes the same split without talking to the others, and the shares are disjoint and together cover everything.

The split is by field of view, never by round, because the drift of a field of view is computed from all of its rounds. Every round of a field of view is processed by the same process.

Results are written one field of view at a time and each file is written under a temporary name and renamed when complete, so processes never write the same file. A run that is killed (a time limit, a crash) leaves no half-written result, and running the same command again continues where it stopped.

## Several GPUs in one machine

```bash
mermake --gpus all settings.toml        # one worker per visible GPU
mermake --gpus 0,2 settings.toml        # only these two
```

`--gpus` starts one worker per GPU, each with only its GPU visible and its own shard, and waits for all of them. If a worker fails, the others carry on and the exit status is non-zero. SIGTERM is passed on to the workers.

GPU numbers count among the GPUs the environment makes visible. If `CUDA_VISIBLE_DEVICES=2,3` is set, `--gpus all` means those two and `--gpu 1` means GPU 3. Without `--gpu` or `--gpus`, `CUDA_VISIBLE_DEVICES` is left alone, so a GPU assigned by a scheduler is used as given.

You can also run the processes yourself, which is what `--gpus` does for you:

```bash
CUDA_VISIBLE_DEVICES=0 mermake --shard 0/2 settings.toml &
CUDA_VISIBLE_DEVICES=1 mermake --shard 1/2 settings.toml &
wait
```

## SLURM

`scripts/run_slurm_parallel.sbatch` is a job array in which every task runs one share of the fields of view on its own GPU. `scripts/run_slurm_report.sbatch` makes the plots once at the end.

```bash
uv sync --extra cuda12 --extra report                 # once, in the clone the jobs use
jobid=$(sbatch --parsable --array=0-7 scripts/run_slurm_parallel.sbatch)
sbatch --dependency=afterok:$jobid scripts/run_slurm_report.sbatch
```

- **One GPU per task** (the default `--gres=gpu:1`) is the simplest. Each task needs only one free GPU, so SLURM can place the tasks anywhere, and it makes no difference whether they land on one node or on many. `--array=0-15%4` runs 16 shards with at most 4 at a time.
- **Several GPUs per task** (for example `--gres=gpu:4 --array=0-1`) runs the workers of a task with `--gpus all`, and the tasks split the shares between them: 2 tasks of 4 GPUs make 8 shards. Raise `--cpus-per-task` and `--mem` in step with the number of GPUs.
- **The environment is prepared once, outside the jobs.** The scripts use `uv run --no-sync`, so many tasks starting together do not race to change the same checkout. The older `run_slurm.sbatch` switches branch and syncs inside the job, which is fine for a single job but not for an array.
- **The input and the output folder must be visible from every node**, on a shared filesystem. Each field of view is one file per plane, so a network filesystem sees many small reads.
- The scripts could not be tested against a real SLURM cluster while writing them. Check `--partition`, `--gres` and the module lines for your site.

## How much memory each process needs

**Host RAM.** An image is held in memory whole, as uint16 after the 8-bit data is squared (and after `sum_channels`). Four channels of 40 planes of 2800 x 2800 are 2.3 GiB, about three times their size on disk. A process holds `--prefetch` images in the read-ahead queue plus three more: one waiting to be queued, one being processed and one kept until the next image of the same field of view joins the drift block. With the default `--prefetch 5` that is **8 images, about 19 GiB per process** for that size, and a machine with four GPUs needs about four times that. If the GPU is the bottleneck a smaller `--prefetch` loses nothing: `--prefetch 2` holds 5 images, 12 GiB.

**GPU RAM.** One channel at a time is on the GPU, cut into patches of `patch_size + 2 * patch_overlap` pixels, so the memory depends on `patch_size`, `patch_overlap` and the number of z planes, plus the size of one channel for the dapi step. It does not depend on the number of channels or on the number of fields of view. Lower `patch_size` if the GPU runs out of memory. A pickled dict of several PSFs multiplies the PSF part by the number of PSFs.

**CPU.** Reading and decompressing the images uses threads. `--gpus` divides the cores of the machine between its workers. Under SLURM give each task `--cpus-per-task` cores.

## Things to know

- Do not run two processes on the same field of view at the same time. Use `--shard`, or settings files whose `[paths] fov_range` do not overlap.
- Sharding looks at the images that exist when the process starts. Start all the shards after the acquisition has finished, or they may split a growing set differently.
- A shard with no images (more shards than fields of view) prints a message and exits.
- `--dry-run --shard I/N settings.toml` lists what one shard would process, without a GPU.
- `--suggest-threshold` samples images from the whole set and ignores `--shard`.
