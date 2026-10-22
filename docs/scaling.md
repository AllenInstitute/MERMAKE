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

The numbers here were measured with `scripts/bench_pipeline.py` on real data: four channels of 40 planes of 2800 x 2800 (an image is 2.34 GiB as uint16, about three times its size on disk) and the default settings.

**Host RAM.** An image is held in memory whole, as uint16 after the 8-bit data is squared (and after `sum_channels`). A process holds up to `--prefetch` images in the read-ahead queue plus three more: one waiting to be queued, one being processed and one kept until the next image of the same field of view joins the drift block. Size the memory for a full queue: `(prefetch + 3)` images plus about 2 GiB, which is 20.9 GiB for the default `--prefetch 5` (measured while the GPU was the bottleneck) and 13.4 GiB for `--prefetch 2`. The queue only fills when the GPU is slower than the reader. With the default settings the GPU needs 8.1 s per image and the reader 6.3 s on that machine, so a run peaked at 12.8 GiB, and `--prefetch 2` gave the same 8.1 s per image. If memory is tight, lower `--prefetch`. A machine with four GPUs needs four times the memory of one process.

**GPU RAM.** One channel at a time is on the GPU, cut into patches of `patch_size + 2 * patch_overlap` pixels, so the memory depends on `patch_size`, `patch_overlap`, the number of z planes and the size of one channel (for the dapi step). It does not depend on the number of channels or fields of view. The peak was **7.0 GiB** with the defaults (`patch_overlap = 110`, `real_fft = true`). Real FFTs halve the FFT buffers, which pays for the 13 % larger patches of the new `patch_overlap`. With complex FFTs and the old code the peak was 9.2 GiB. Lower `patch_size` if the GPU runs out of memory. A pickled dict of several PSFs multiplies the PSF part by the number of PSFs.

**More than one process per GPU does not help.** Two workers fit on a 15 GiB T4 (14.4 GiB used) and gave the same results, but took 166 s for two fields of view that take 155 s one after the other: the GPU was busy 95 % of the time, and the work is bandwidth-bound FFTs. Use one process per GPU.

**CPU.** Reading and decompressing the images uses threads. `--gpus` divides the cores of the machine between its workers. Under SLURM give each task `--cpus-per-task` cores.

## Where the time goes

Per image on one Tesla T4, same data, from `scripts/bench_pipeline.py`:

| | seconds per image |
|---|---|
| Version 0.1.0 | 26.4 |
| Three stalls and wasteful kernels removed (same results) | 18.8 |
| plus `real_fft = true` and `patch_overlap = 110`, **the defaults now** | **8.1** |

Running the nine rounds of a field of view took 87 s instead of 185 s, startup included.

What changed, with identical results (the spots, the diagnostics and the drift matched a run of the old code on one field of view, 3.2 million spots, apart from 1e-6 voxel in the drift, which also differs between two runs of the old code):

- `box_1d` asked `cupy.shares_memory` whether its two buffers overlap on every call. The exact answer takes about 20 ms of host time on a patch, with the GPU idle, twice per patch. The cheap bounds test comes first now (21 % of the time).
- `find_local_maxima` allocated three arrays the size of the whole image to hold the maxima, flushed the memory pool after every call, and converted a uint16 image to float32 as a whole. This took 1.8 GiB off the peak GPU memory and a few percent off the time.
- `reflect` visited every element of the padded stack to change a few planes. It copies slices now.

What changed in the defaults, which does change the results a little:

- **`real_fft = true`** uses real-to-complex FFTs (`rfftn`/`irfftn`). The image and the PSF are real, so the result is the same up to rounding. It halves the FFT time and the memory of the PSF spectrum and of the FFT buffer. On one field of view 18 of 3.2 million spots changed (0.0006 %), all of them at the threshold, and `h` of the others changed by at most 5e-6 (relative) for 99 % of the spots. `real_fft = false` gives the complex FFTs of version 0.1.0.
- **`patch_overlap = 110`** (it was 89). The padded patch has `(3 * depth - 2) x (patch_size + 2 * patch_overlap)` squared voxels. cuFFT is fast for sizes whose prime factors are 2, 3, 5 and 7, and slow otherwise: 678 = 2 x 3 x 113 was slow, 720 = 2^4 x 3^2 x 5 is not, and the FFT is 2.1 times faster. **Any change of the patch geometry moves about 0.4 % of the spots**, those within a percent of the threshold: `h` of the matched spots changed by 1 % or less for 99 % of them, and their positions by 0.012 voxel or less. This did not depend on the value (86, 93 and 110 all gave 0.36-0.42 %). To get the results of version 0.1.0 back, set `patch_overlap = 89` and `real_fft = false` in `[hybs]` and `[dapi]`: that reproduced the old run exactly. If you change `patch_size`, choose `patch_overlap` so that `patch_size + 2 * patch_overlap` is again a product of 2, 3, 5 and 7.

**Not done: the z padding.** It is set in code to `depth - 1` planes on each side, so 118 planes for 40, and 118 = 2 x 59 is also a slow size. With 120 planes for the color channels and 80 for dapi (and `patch_overlap = 86`) a field of view took 69 s. It changes the deconvolution a little at the top and bottom planes, so it was left out.

**The storage may be the limit before the GPU is.** On the machine these numbers come from, the input (a FUSE mount) delivered about 135-150 MB/s in total, however many processes read at the same time (one process: 153 MB/s, two: 66 and 70, four: 33-35 each). An image is 0.8 GB, so the most that machine can read is one image every 6 seconds, which is already close to the 8.1 s the GPU needs. Decompressing is cheap (0.3 s for an image) and squaring to uint16 takes 1 s. If that is true of your cluster, more GPUs will not help beyond the point where they read as fast as the storage delivers. Check with `scripts/bench_read.py`: run it on one field of view, then on several at once (`... & ... & wait`) and see whether the total grows.

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
