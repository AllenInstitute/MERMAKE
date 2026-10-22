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

The numbers here were measured with `scripts/bench_pipeline.py` on real data: four channels of 40 planes of 2800 x 2800 (an image is 2.34 GiB as uint16, about three times its size on disk), with the default `patch_size` 500 and `patch_overlap` 89.

**Host RAM.** An image is held in memory whole, as uint16 after the 8-bit data is squared (and after `sum_channels`). A process holds `--prefetch` images in the read-ahead queue plus three more: one waiting to be queued, one being processed and one kept until the next image of the same field of view joins the drift block. The peak resident size of a process was **20.9 GiB with the default `--prefetch 5`** and **13.9 GiB with `--prefetch 2`**, which is three images less. The time per image did not change (18.9 s both times), because the GPU was the bottleneck and the queue was always full. A machine with four GPUs needs four times the memory of one process.

**GPU RAM.** One channel at a time is on the GPU, cut into patches of `patch_size + 2 * patch_overlap` pixels, so the memory depends on `patch_size`, `patch_overlap`, the number of z planes and the size of one channel (for the dapi step). It does not depend on the number of channels or fields of view. The peak was **7.0 GiB** for the data above (it was 9.2 GiB before the maxima step stopped allocating arrays the size of the whole image). Lower `patch_size` if the GPU runs out of memory. A pickled dict of several PSFs multiplies the PSF part by the number of PSFs.

**More than one process per GPU does not help.** Two workers fit on a 15 GiB T4 (14.4 GiB used) and gave the same results, but took 379 s for two fields of view that take 2 x 182 s one after the other: the GPU was already busy 95-99 % of the time, and the work is bandwidth-bound FFTs. Use one process per GPU.

**CPU.** Reading and decompressing the images uses threads. `--gpus` divides the cores of the machine between its workers. Under SLURM give each task `--cpus-per-task` cores.

## Where the time goes

Per image on one Tesla T4, same data and settings, from `scripts/bench_pipeline.py`:

| | seconds per image | |
|---|---|---|
| Before this work | 26.4 | |
| Three stalls and wasteful kernels removed | 18.8 | in this version |
| plus real-to-complex FFT | 11.2 | not applied, see below |
| plus FFT sizes that factor into small primes | 6.5 | not applied, see below |

What changed in this version, with identical results (the spots, the diagnostics and the drift matched a run of the old code on one field of view, 3.2 million spots, apart from 1e-6 voxel in the drift, which also differs between two runs of the old code):

- `box_1d` asked `cupy.shares_memory` whether its two buffers overlap on every call. The exact answer takes about 20 ms of host time on a patch, with the GPU idle, twice per patch. The cheap bounds test comes first now (21 % of the time).
- `find_local_maxima` allocated three arrays the size of the whole image to hold the maxima, flushed the memory pool after every call, and converted a uint16 image to float32 as a whole. This took 1.8 GiB off the peak GPU memory and a few percent off the time.
- `reflect` visited every element of the padded stack to change a few planes. It copies slices now.

**Now about 73 % of the time is the FFTs of the deconvolution.** Two changes would make them much faster, but they change the results a little, so they are not applied:

- **Real-to-complex FFT** (`rfftn`/`irfftn` instead of `fftn`/`ifftn`). The image and the PSF are real, so the result is the same up to rounding. It takes the FFT time and the memory of the PSF spectrum and of the FFT buffer in half: 18.8 s to 11.2 s per image. On one field of view, 18 of 3.2 million spots changed (0.0006 %), all of them at the threshold, and `h` of the others changed by at most 5e-6 (relative) for 99 % of the spots.
- **FFT sizes.** cuFFT is fast for sizes whose prime factors are 2, 3, 5 and 7, and slow otherwise. The padded patch has `(3 * depth - 2) x (patch_size + 2 * patch_overlap)` squared voxels, which is 118 x 678 x 678 for 40 planes and the defaults: 118 = 2 x 59 and 678 = 2 x 3 x 113. A forward and inverse FFT of that takes 108 ms, and one of 120 x 672 x 672 takes 33 ms. In the xy plane this is a setting: `patch_overlap = 110` in `[hybs]` and `[dapi]` gives 720 pixels (2^4 x 3^2 x 5), 2.1 times faster FFTs and a run that is 35 % shorter (118 s instead of 182 s for the nine rounds of a field of view). The z padding is set in code (`zpad = depth - 1`), and 120 planes instead of 118 would make them about a third faster again. **Changing the patch geometry moves about 0.4 % of the spots**, those within a percent of the threshold: `h` of the matched spots changed by 1 % or less for 99 % of them, and their positions by 0.012 voxel or less. This did not depend on which `patch_overlap` was tried (86, 93 and 110 all gave 0.36-0.42 %). It is a change of a parameter of the analysis, not of the arithmetic, so decide it yourself.

Both changes together (real-to-complex FFTs, `patch_overlap = 86` and 120 and 80 planes of z padding) took 69 s instead of 182 s for the nine rounds.

**The storage may be the limit before the GPU is.** On the machine these numbers come from, the input (a FUSE mount) delivered about 135-150 MB/s in total, however many processes read at the same time (one process: 153 MB/s, two: 66 and 70, four: 33-35 each). An image is 0.8 GB, so the most that machine can read is one image every 6 seconds, about what the fastest settings above need. Decompressing is cheap (0.3 s for an image) and squaring to uint16 takes 1 s. If that is true of your cluster, more GPUs will not help beyond the point where they read as fast as the storage delivers. Check with `scripts/bench_read.py`: run it on one field of view, then on several at once (`... & ... & wait`) and see whether the total grows.

To measure your own run:

```bash
uv run --no-sync python scripts/bench_pipeline.py settings.toml --images 9 --skip 1   # stages, GPU functions, memory
uv run --no-sync python scripts/bench_patch.py --depth 40                              # one patch, FFT shapes
uv run --no-sync python scripts/bench_read.py round/Conv_zscan1__006.zarr              # read throughput
```

`bench_pipeline.py` shows the wait for the reader: if it is not near 0 the GPU is waiting for the input. Give it a settings file whose output folder has no results.

## Things to know

- Do not run two processes on the same field of view at the same time. Use `--shard`, or settings files whose `[paths] fov_range` do not overlap.
- Sharding looks at the images that exist when the process starts. Start all the shards after the acquisition has finished, or they may split a growing set differently.
- A shard with no images (more shards than fields of view) prints a message and exits.
- `--dry-run --shard I/N settings.toml` lists what one shard would process, without a GPU.
- `--suggest-threshold` samples images from the whole set and ignores `--shard`.
