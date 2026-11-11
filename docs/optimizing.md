# Optimizing a stage

How to make a stage faster without changing what it computes: measure,
change one thing, prove the results are unchanged, measure again, and
keep a record. Every tool below is in this repository.

## The loop

1. **Measure a baseline** on the device you optimize for, with the size
   and settings your application uses:

   ```shell
   mix run bench/pipeline.exs --kernels --size 544x728 --search coarse_to_fine --paths 8 --coarse-paths 3 --packed --runs 30
   ```

   Each stage is timed compiled alone, then each of its Vulkan kernels
   alone, with the real values the kernels before it produced. Note the
   largest kernels and the "Call" row, the time of a call that does
   nothing: no kernel change removes it.

2. **Pick the largest kernel** and read its shader in `shaders/`;
   [How the shaders work](../README.md#how-the-shaders-work) says what
   each stage's shaders do and the layouts they share.

3. **Change one thing.** After editing a shader, regenerate its SPIR-V
   with `mix nx_stereo.shaders` (needs `glslangValidator`); the binaries
   are committed with the sources.

4. **Prove the results are unchanged.** `mix test --include vulkan`
   compares every shader with its Nx code on every Vulkan device, and
   the benchmark's "Differing" column must stay 0. For the native
   backend, `mix test` compares every native stage with the Nx code and
   `cargo test` in `native/` runs the program's own tests. A change that
   alters results is an algorithm change: check it with
   `mix run bench/datasets.exs` against ground truth, not only against
   the old output.

5. **Measure again**, with the same command. Times are medians; a gain
   smaller than the run-to-run spread is not a gain. Alternate baseline
   and change a few times when it is close.

6. **Measure on the target board**, if it is not the machine you work
   on: a GPU's cost model differs from another's (see below). An
   application that runs nx_stereo on a board should be able to load a
   new nx_stereo build there without a full firmware and time it; OVCS's
   `mix vision_core.board_bench --hot` is one example.

7. **Record it.** Put the device, size, settings and the before and
   after times in the commit message. Keep the ideas that did not help,
   with their numbers, where the next person will look, so they are not
   tried again.

## What "unchanged" means

| Output | Compared with the Nx code on EXLA |
| --- | --- |
| Integer outputs, matched pixels, depth in millimetres | identical |
| Disparity | identical; a subpixel value may differ in its last bit on a GPU, whose division need not be correctly rounded |
| Sums (the diagnostics) | within their rounding, 1e-5 relative: a reduction in another order rounds differently |
| Rectified images | within one grey level, where a sample falls on a rounding half |
| Temporal filter | identical to Nx's evaluator, op by op: EXLA fuses its blend into a multiply-add, which rounds once |

## Testing a kernel

A kernel's test compiles its `Nx.block` for Vulkan with a default
function that returns zeros, so only the kernel can give the right
answer, and compares the result with the Nx code on EXLA
(`test/nx_stereo/kernels_test.exs`). Also test the sizes and settings
the kernel does not take, such as a width that is not a multiple of 4:
they must fall back to the Nx code.

## Reading the numbers

- A kernel's time includes a call's. Compare kernels with each other and
  with "Call"; their sum is not the stage's time.
- "Chain" runs every stage in one call, as an application does: it is
  less than the sum of the stages, by the calls it saves.
- Estimate the bytes a kernel reads and writes over its time. Near the
  device's copy bandwidth, it is memory-bound: only moving fewer bytes
  helps (packing, fusing two kernels, recomputing instead of storing).
- On a board, keep the GPU at a fixed clock while measuring: a governor
  that lowers it between bursts makes every number noisy.

## What we learned on the Adreno 623

Measured on an Arduino VENTUNO Q at 728x544 over 128 disparities; each
point held with results equal to EXLA's.

- **Shared memory is slow for moving data.** Staging tiles through it
  cost more than reading global memory directly: the band costs'
  transpose went from 1.88 to 0.74 ms without it, the coarse selection
  from 2.11 to 0.98 ms.
- **Loads consecutive across a wave's threads are fast**, strided ones per
  thread are not; vector loads (`vec2`, `vec4`) helped the depth kernel
  (0.47 to 0.32 ms) and the temporal filter (0.72 to 0.44 ms).
- **Arrays indexed by a value computed at run time spill to scratch
  memory.** Keep indices constant: the band scans shift their window
  with a barrel shifter over packed bytes (9.4 to 8.0 ms for the eight
  scans).
- **Fewer bytes win when a kernel is memory-bound:** band costs stored
  6 bits a candidate instead of 8 (16.3 to 15.6 ms for the fine pass).
- **Fewer kernels win when one stores what the next reads:** the 3-path
  selection inside the vertical scan (2.21 to 1.81 ms).
- **Workgroup size matters:** 64 threads instead of 128 for the coarse
  scans (5.59 to 5.31 ms).
- **A thread should not run outside the image:** the folded diagonal
  scans keep every thread busy (1.78 to 1.61 ms per pair of scans).
- **Reading more than a few volumes at once slows a kernel sharply,** so
  paired scans write the sum of two paths and the selection reads four
  volumes instead of eight.

Tried and dropped, as they did not help there: a register-relative
window (spills to scratch), a branch-free mirrored barrel shifter,
`uvec4` scan loads and stores, merged horizontal and vertical
dispatches, row-major horizontal scans, a quarter-size base, deeper
prefetching in the coarse scans, a row-major coarse volume, split coarse
stores, four tiles per workgroup in the coarse selection, a tiled band
selection, direct reads, `uvec4` pairs, split planes or two pixels a
thread in the band costs, direct reads or 8 pixels a thread in the
speckle filter, a quad median, larger transpose tiles, pairing the
forward and reverse scans with an 8-volume selection (the selection
grew by 1.18 ms) and `subgroupAdd` reductions (4 times slower than
shuffles).

## Local Vulkan measurements

The November 2026 measurements use Intel Iris Xe (RPL-P), Mesa 26.0.8 and
IREE compiler 3.13.0rc20261016 at 728x544 working resolution, from
1456x1088 packed frames,
over 128 disparities. They establish local improvements; the Adreno 623
needs its own measurements before choosing a deployment profile. The
[metadata](benchmarks/vulkan-2026-11/metadata.json) records dependency revisions,
runtime settings, workloads and quality metrics.

The complete-chain stream uploads changing CPU frames, carries temporal
state and downloads every published output. It excludes camera capture,
compilation, fixture generation and display. Compare both sustained FPS
and individual frame latency: overlapping downloads can improve throughput
while making a frame wait behind the next computation.

### Complete original-versus-current comparison

Six alternating rounds of 200 changing frames compare the original revision
`eb19f7cf8a23ad8a0e9c5313f43bb78adbfe18c2` with these changes. Every published
output and the final carried state are identical across 16 changing frames
for each configuration. The original matcher, shaders, diagnostics and segment
code are loaded under separate module names. A stage adapter forwards the
requested band and margin into the original matcher, since the original stage
did not expose them. Both versions therefore run the specified search settings.
All three profiles below use the original sparse census, so the speed comparison
does not trade away matcher quality.

| Fine band / coarse paths | Serial FPS, original → current | Overlap FPS, original → current | Current overlap intervals above 33.33 ms |
| --- | --- | --- | --- |
| 16 / 3 | 27.142 → 46.237 | 30.465 → 54.433 | 0 / 1200 |
| 16 / 8 | 26.620 → 45.320 | 28.147 → 52.687 | 0 / 1200 |
| 32 / 8 | 16.128 → 33.366 | 16.764 → 34.026 | 65 / 1200 |

Margins are 4 for band 16 and 8 for band 32; fine paths are 8 throughout.
These are medians across rounds. Band-32 overlap has median frame latency
34.548 ms despite throughput above 30 FPS; serial latency is 29.636 ms.
Neither throughput nor these short runs guarantee every frame's deadline.
The [original and current repetitions](benchmarks/vulkan-2026-11/original-comparison.csv)
include individual round timings, latency percentiles and missed intervals.
The [comparison script](benchmarks/vulkan-2026-11/experiments/original_comparison.exs)
accepts `BASELINE_ROOT`, a directory containing the original revision, and runs
from the current repository root. For example, prepare that directory with
`git archive eb19f7cf8a23ad8a0e9c5313f43bb78adbfe18c2` and extract the archive
under `_build/bench/original`; then set `BASELINE_ROOT` to its absolute path.

CPU byte packing now selects BinaryBackend explicitly. With an EXLA default,
packing a 1456x1088 frame previously allocated an EXLA buffer and took a median
1.719 ms; the CPU view takes 0.00058 ms. Downloaded-image unpacking falls from
1.638 to 0.00056 ms. With BinaryBackend already selected, both versions remain
below 0.001 ms. The [packing repetitions](benchmarks/vulkan-2026-11/packing-bench.csv)
contain six alternating rounds of 400 calls, including deallocation. These
numbers measure byte views, excluding camera acquisition and Vulkan upload.

### CPU collection with an EXLA global default

Collected tensors now use BinaryBackend views explicitly. Previously, each
`Nx.from_binary` in the collection task followed the global backend, allocating
EXLA buffers for data that had just been downloaded. Packed-image unpacking
already selects BinaryBackend; the other published outputs now do the same.

For the real 6,634,752-byte published bundle, six alternating rounds of 200
collection calls measure reconstruction and task overhead after downloading
once. The median per-call time falls from 6.951 to 0.039 ms with EXLA configured
globally. With BinaryBackend already global, it remains about 0.03 ms (0.028
before, 0.026 after). Output bytes, shapes and types agree across both versions.
The [isolated collection repetitions](benchmarks/vulkan-2026-11/collection-backend.csv)
exclude Vulkan execution and download; returned buffers are deallocated.

A separate complete-stream comparison uses the wide blended profile, an EXLA
global default, and six alternating rounds of 120 frames per version/mode.
Every published output and final carry agrees across 16 changing frames first.

| Collection | Serial FPS | Overlap FPS | Serial median latency | Overlap median latency |
| --- | ---: | ---: | ---: | ---: |
| Global-backend reconstruction | 23.198 | 34.846 | 43.293 ms | 49.162 ms |
| CPU binary views | 32.449 | 33.921 | 30.686 ms | 34.787 ms |

These are medians across repetitions. Overlap throughput is slightly lower
with CPU views in this comparison, while its late completion count falls from
177/720 to 48/720 and latency improves. The serial path now exceeds 30 FPS
locally. The [complete-stream repetitions](benchmarks/vulkan-2026-11/collection-backend-stream.csv)
and [telemetry](benchmarks/vulkan-2026-11/collection-backend-stream-telemetry.csv)
retain that tradeoff. The longer runs below use the BinaryBackend default,
under which reconstruction behavior is unchanged.

### Sustained changing-frame run

A separate run processes 144,000 timed frames: 20 rounds of 1200 frames for
each profile and collection mode, with 24 warm-up frames per round. Serial
and overlap outputs and final carry agree across 16 changing frames before
each profile is timed. All runs use packed frames and cloud decimation four.

| Profile | Aggregate serial FPS | Aggregate overlap FPS | Serial late completions | Overlap late completions | Overlap late individual latencies |
| --- | ---: | ---: | ---: | ---: | ---: |
| Band 16, margin 4, coarse 3, sparse | 44.781 | 55.644 | 2 / 24,000 | 0 / 24,000 | 44 / 24,000 |
| Band 16, margin 2, coarse 8, blend | 45.107 | 54.221 | 3 / 24,000 | 0 / 24,000 | 36 / 24,000 |
| Band 32, margin 8, coarse 8, blend | 32.635 | 34.183 | 2933 / 24,000 | 1229 / 24,000 | 17,042 / 24,000 |

Aggregate FPS divides all timed frames by their total elapsed time; it is
not the fastest round. A late completion or latency exceeds 33.33 ms.
Median round p50 latency for the wider profile is 30.402 ms serial and
34.667 ms with overlap. Its worst round p99 is 44.380 ms serial and
42.105 ms with overlap. Thus all profiles exceed 30 FPS in aggregate locally,
but the wider profile has substantially less deadline headroom.

Resident process memory stays around 515–521 MiB across timed rounds; this
observation does not include separate compiler processes. GPU clocks vary,
and the temperature samples are the integrated CPU package sensor. The
[time-series figure](benchmarks/vulkan-2026-11/sustained-stream.png),
[per-round results](benchmarks/vulkan-2026-11/sustained-stream.csv),
[summary](benchmarks/vulkan-2026-11/sustained-summary.json),
[round environment](benchmarks/vulkan-2026-11/sustained-environment.csv), and
[raw telemetry](benchmarks/vulkan-2026-11/sustained-telemetry.csv) preserve
those observations. No target-board performance is inferred from them.

### Memory and scans

The pinned IREE Vulkan allocator pools allocations up to 64 MiB by default
([allocator source](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/runtime/src/iree/hal/drivers/vulkan/allocator.c)).
A controlled width sweep crosses that boundary between widths 720 and
724 at height 544: compiled-call medians rise from 10.542 to 26.931 ms.
Fusion and packing therefore matter beyond their individual kernel times.

- Horizontal fine paths share one output volume. At band 16, coarse paths
  3 and fine paths 8, compiled-chain scratch falls from 67,721,728 to
  61,385,216 bytes. Each invocation writes its forward path, then adds its
  reverse path to those same locations.
- Bands 32 and 64 distribute four candidates to each subgroup lane and
  accumulate eight paths in ten bits per candidate. Band 32 with eight
  coarse paths falls from 121,978,112 to 61,781,248 scratch bytes. Ten bits
  preserve the maximum aggregate, 8 × 127 = 1016.
- Contiguous low words plus shared high bytes reduce aggregate shuffles.
  Only band 32 takes the unchanged-base shortcut; that branch regresses
  band 64 on changing bases.
- Band 64 calculates four census distances at a time. Narrower bands retain
  their existing load order because their measurements show no consistent
  benefit.

Measured changes are isolated comparisons, not additive gains:

| Change | Complete changing-frame workload | Median FPS before → after |
| --- | --- | --- |
| Cheaper wide aggregation and unchanged-base shortcut | Band 32, margin 8, coarse/fine paths 8, sparse census; four alternating rounds of 200 frames | 31.101 → 33.993 |
| Four signatures at a time in cost construction | Band 64, margin 16, coarse/fine paths 8, sparse census; six alternating rounds of 150 frames | 15.270 → 15.986 |
| Full-range horizontal fusion | Full 128 disparities, three paths; ten alternating rounds of 300 frames | 23.799 → 26.981 |

The band-32 comparison reduces completion intervals above 33.33 ms from
220/800 to 44/800. Band 64 remains below 30 FPS locally. The corresponding
[wide-scan repetitions](benchmarks/vulkan-2026-11/wide-scan-stream.csv) and
[band-64 repetitions](benchmarks/vulkan-2026-11/band64-cost-stream.csv)
retain each round, including latency percentiles.

Full-range fusion wins all ten paired rounds but still falls below 30 FPS.
Its [repetitions](benchmarks/vulkan-2026-11/full-range-fusion.csv) also include
coarse-to-fine controls: band 16 with three coarse paths changes from 57.189
to 56.305 FPS, and band 32 from 37.190 to 37.128. Those longer runs do not
reproduce an earlier short-run coarse-to-fine gain. The fused full-range
shader is therefore selected only for 128 disparities and three paths.

### Point-cloud density

`--cloud-decimation` exposes the stage's existing sampling step in both complete
pipeline benchmarks. The default of four publishes a 136x182 point grid at the
working size above; two publishes 272x364 and one publishes 544x728. Each grid
point contains three f32 coordinates. Invalid points remain NaN.

Six balanced rotations of 150 changing frames per mode compare band 32, margin
8, eight coarse/fine paths and blended census. Both modes agree on all outputs
and carried state at each density before timing. All published outputs are
downloaded, including the larger cloud.

| Sampling step | Points in grid | Serial FPS | Overlap FPS | Overlap median latency | Overlap intervals above 33.33 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 24,752 | 32.484 | 33.734 | 34.928 ms | 48 / 900 |
| 2 | 99,008 | 29.291 | 33.204 | 59.347 ms | 97 / 900 |
| 1 | 396,032 | 31.689 | 33.901 | 36.677 ms | 77 / 900 |

These are medians across repetitions, not guarantees of a 30 FPS deadline.
The density/throughput relationship is not monotonic in this run, and step two
has substantially worse overlapped latency. Benchmark the actual density and
collection mode; the default remains four. The
[density repetitions](benchmarks/vulkan-2026-11/cloud-density.csv) and
[clock/temperature samples](benchmarks/vulkan-2026-11/cloud-density-telemetry.csv)
retain the observations.

Inspection finds the same 61,781,248-byte scratch allocation at all three
steps. The separate published bundle grows from 6,634,752 to 7,525,632 and
11,089,920 bytes, so crossing the scratch-pool threshold does not explain the
step-two anomaly. A second step-two experiment compares six alternating rounds
of 200 frames: ordinary overlap reaches a median 33.445 FPS and 59.662 ms
latency; starting a collection task first reaches 33.134 FPS and 59.630 ms.
Task start order does not guarantee GPU queue order, and this experiment does
not demonstrate a scheduling improvement. The
[collection-order repetitions](benchmarks/vulkan-2026-11/density-collection-order.csv)
preserve the result; the scheduling API stays unchanged.

### Compiling separate resident stage groups

A prototype compiles the fixed benchmark chain in two or three groups while
keeping intermediate tensors on Vulkan. Four balanced rotations of 150 changing
frames compare all variants at each size. Every published output and final carry
agrees across 16 changing frames per mode before timing.

Splitting immediately after rectification gives these median overlap results:

| Working size (width x height) | One call → two calls, FPS | Median latency, one call → two calls |
| --- | ---: | ---: |
| 728 x 544 | 34.458 → 32.761 | 34.076 → 36.112 ms |
| 768 x 576 | 23.693 → 30.473 | 47.117 → 38.580 ms |
| 800 x 600 | 22.016 → 22.073 | 89.074 → 54.132 ms |
| 960 x 720 | 15.845 → 16.033 | 124.278 → 74.113 ms |

At 768x576, the largest scratch allocation falls from 69,009,664 to
64,585,792 bytes, crossing below the pinned runtime's 64 MiB pool limit.
The throughput improvement repeats in every paired round. Even here, 227/600
overlap completion intervals exceed 33.33 ms, and serial throughput reaches
28.984 FPS. Splitting is slower at the original 728x544 working size; at the
larger sizes it mainly changes latency. Three groups add more overhead.

The [repetitions](benchmarks/vulkan-2026-11/stage-partitioning.csv),
[allocation sizes](benchmarks/vulkan-2026-11/stage-partitioning-allocations.json)
and [telemetry](benchmarks/vulkan-2026-11/stage-partitioning-telemetry.csv)
retain all variants. This stays an explicit prototype for the fixed benchmark
chain: it discards camera frames after rectification and does not implement
general custom-stage input lifetimes. The production NxSegment API and its
default single compiled call are unchanged by this experiment.

Run the saved prototype from the repository root, with a separate existing
`TMPDIR` if another BEAM process is active:

```shell
PARTITION_SIZE=576x768 MIX_ENV=test mix run docs/benchmarks/vulkan-2026-11/experiments/partition.exs
```

### Descriptor and margin tradeoffs

The [quality observations](benchmarks/vulkan-2026-11/quality-profiles.csv)
cover 15 Middlebury quarter-resolution scenes, 27 ETH3D scenes and every
20th KITTI 2015 training pair at half resolution (10 pairs). All profiles
use eight coarse paths, eight fine paths, P1 = 8, P2 = 64, uniqueness
1.0526315789473684 and supported ground truth in [0, 128). Images are
cropped to even dimensions. Means weight scenes equally.

Each cell below is bad-or-missing percentage / matched-pixel mean absolute
error in pixels, on clean images. A missing estimate counts as bad, so
rejecting difficult pixels cannot improve the first metric by itself.

| Fine profile | Middlebury | ETH3D | KITTI |
| --- | --- | --- | --- |
| Band 16, margin 4, sparse | 15.076 / 1.704 | 7.363 / 0.693 | 6.497 / 0.675 |
| Band 16, margin 4, blend | 14.853 / 1.680 | 7.163 / 0.680 | 6.455 / 0.664 |
| Band 16, margin 2, blend | 14.898 / 1.696 | 6.949 / 0.676 | 6.365 / 0.657 |
| Band 32, margin 8, sparse | 14.480 / 1.541 | 8.323 / 0.705 | 6.381 / 0.630 |
| Band 32, margin 8, blend | 14.252 / 1.531 | 7.992 / 0.689 | 6.301 / 0.610 |
| Band 32, margin 6, blend | 14.232 / 1.533 | 7.896 / 0.687 | 6.285 / 0.610 |

Here "blend" means `dense_9x5_blend`. Band 32 with margin 8 improves both
metrics relative to the same sparse profile in all 21 dataset/condition
means: clean, independent noise at standard deviations 2/4/8, two right-eye
exposure changes and right-eye blur. Coverage can still decrease, and
individual scenes can regress. Across the 364 paired scene/condition results,
both metrics improve in 252 pairs and worsen in 51; 61 are mixed or tied.
The [paired counts](benchmarks/vulkan-2026-11/quality-paired-comparison.json)
retain the breakdown. The CSV includes coverage for every scene.

Disparity and absolute-error maps show the spatial tradeoffs for
[Middlebury Adirondack](benchmarks/vulkan-2026-11/middlebury-Adirondack.png),
[ETH3D delivery_area_1l](benchmarks/vulkan-2026-11/eth3d-delivery_area_1l.png)
and [KITTI 000000_10](benchmarks/vulkan-2026-11/kitti-000000_10.png).
These are matcher outputs before speckle and temporal filtering; grey marks
unscored ground truth and blue marks a missing prediction in the error maps.
The KITTI example has slightly higher bad-or-missing rate with the optional
profiles despite lower matched-pixel error.

Smaller margins improve some clean-image means but slightly worsen noisy
Middlebury error. A wider band is not uniformly more accurate: ETH3D
favors the narrower profiles in these measurements. Keep the sparse
production default and select the optional descriptor and band using
representative images and the target board's complete-chain timing.

For example, reproduce a noisy wide-profile comparison with:

```shell
mix run bench/datasets.exs --datasets middlebury,eth3d,kitti --search coarse_to_fine --paths 8 --coarse-paths 8 --band 32 --margin 8 --fine-census dense_9x5_blend --perturbation noise4 --warmup 3 --runs 20 --csv quality-noise4.csv
```

### Depth-boundary checks

The [region observations](benchmarks/vulkan-2026-11/quality-boundaries.csv)
separate known pixels within two pixels of an adjacent ground-truth disparity
jump greater than two pixels from the remaining known pixels. Both endpoints
must have valid, in-range ground truth; a missing ground-truth value never
creates an edge. A 5x5 dilation defines the neighbourhood. Step, flat-surface
and missing-truth controls pass, and recombining all 624 region pairs recovers
the previous whole-image metrics (maximum error difference below 0.000001 px).

For band 32, margin 8 and eight coarse/fine paths, Middlebury's 15 scenes give
these equal-scene means. Cells are bad-or-missing percentage / matched-pixel
mean absolute error:

| Input / region | Sparse descriptor | Blended descriptor |
| --- | ---: | ---: |
| Clean / boundary | 34.530 / 4.253 | 33.941 / 4.229 |
| Clean / remaining pixels | 12.197 / 1.187 | 11.990 / 1.178 |
| Noise 4 / boundary | 41.817 / 4.856 | 41.656 / 4.829 |
| Noise 4 / remaining pixels | 23.094 / 1.811 | 22.795 / 1.810 |

Boundary coverage changes from 96.155% to 96.239% on clean images, but from
95.370% to 94.989% with noise. The improvement does not eliminate boundary
errors or preserve coverage in every condition.

This particular boundary mask covers 349,405 Middlebury pixels, only 328
pixels across seven ETH3D scenes, and no KITTI pixels in these sparse-truth
samples. It therefore supports the Middlebury comparison above, not a
cross-dataset boundary ranking. Undefined scores remain empty in the CSV.
The [summary](benchmarks/vulkan-2026-11/quality-boundaries-summary.json)
records how many scenes contribute. Reproduce it from the repository root:

```shell
MIX_ENV=test mix run docs/benchmarks/vulkan-2026-11/experiments/boundary_quality.exs
```

### Existing penalty controls

The stages already expose live P1 and P2 settings. A
[six-setting sweep](benchmarks/vulkan-2026-11/quality-penalties.csv) covers both
blended profiles on 52 scenes, clean and with noise 4. Twenty-four independent
Vulkan/EXLA comparisons pass before scoring; all 208 default-penalty control
rows reproduce the previous public-API metrics.

For the wide blended profile on clean images, changing these settings gives:

| P1 / P2 | Middlebury bad-or-missing / error | ETH3D bad-or-missing / error | KITTI bad-or-missing / error |
| --- | ---: | ---: | ---: |
| 8 / 48 | 14.133 / 1.501 | 8.494 / 0.738 | 6.402 / 0.613 |
| 8 / 64, default | 14.252 / 1.531 | 7.992 / 0.689 | 6.301 / 0.610 |
| 10 / 79 | 14.447 / 1.577 | 6.920 / 0.630 | 6.321 / 0.606 |

Percentages and matched-pixel error show different tradeoffs across datasets;
no setting in this sweep improves both metrics across all six dataset/input
conditions. Defaults remain 8 / 64. The
[complete summary](benchmarks/vulkan-2026-11/quality-penalties-summary.json)
includes coverage and the narrow profile. Reproduce the sweep with:

```shell
MIX_ENV=test mix run docs/benchmarks/vulkan-2026-11/experiments/penalty_quality.exs
```

A [follow-up sweep](benchmarks/vulkan-2026-11/quality-penalty-refinement.csv)
checks P1 = 9/10 with P2 = 56/60/64 near the default. It repeats another
24 reference comparisons and 208 baseline rows. The best alternatives improve
both metrics in four of six dataset/input means, but none does so in all six.
For example, wide 10 / 60 improves Middlebury to 14.104% / 1.525 px and ETH3D
to 7.209% / 0.662 px, while KITTI worsens to 6.395% / 0.612 px. The
[refinement summary](benchmarks/vulkan-2026-11/quality-penalty-refinement-summary.json)
and [script](benchmarks/vulkan-2026-11/experiments/penalty_refinement.exs)
retain the comparison without changing defaults.

A final [uniqueness sweep](benchmarks/vulkan-2026-11/quality-uniqueness.csv)
tests 1.03/1.04 with the default and two refined penalty settings. Its 24
reference comparisons and 208 baseline controls also pass. Lowering the wide
profile's default uniqueness factor to 1.03 raises clean-image coverage by
1.192 / 0.827 / 0.201 percentage points on Middlebury / ETH3D / KITTI and lowers
bad-or-missing rate, but raises matched-pixel error by 0.116 / 0.017 / 0.006 px.
None of the combined settings improves both metrics across all six
conditions. The [summary](benchmarks/vulkan-2026-11/quality-uniqueness-summary.json)
and [reproduction script](benchmarks/vulkan-2026-11/experiments/uniqueness_quality.exs)
preserve the coverage/precision tradeoff; the default factor is unchanged.

### Filter settings

The [filter observations](benchmarks/vulkan-2026-11/filter-profiles.csv)
compare the existing `min_support: 12, max_diff: 1.0` with
`min_support: 14, max_diff: 1.5`, using clean images and independent noise
with standard deviation 4. Both band-16/margin-2 and band-32/margin-8 profiles
use eight coarse/fine paths and blended census. These are single-frame
speckle-filter results, before temporal filtering.

For band 32, the latter setting retains 0.23–0.66 percentage points more
pixels across the six dataset/condition means and lowers bad-or-missing
rate by 0.12–0.44 points. Matched-pixel error rises by 0.003–0.026 pixels.
This is a coverage/precision choice; the filter defaults remain unchanged.

### Experiments without a retained change

Smaller/larger wide-scan workgroups (32/128/256 versus 64 threads) provide
no consistent improvement. Starting a collection task before the next
compute task improves the wide stream slightly but slows the narrow one;
[collection-order repetitions](benchmarks/vulkan-2026-11/collection-order.csv)
show the tradeoff. Serial collection keeps frame latency lower than
throughput-oriented overlap on the wide profile.

Image-gradient-dependent P2 improves Middlebury but worsens ETH3D and
KITTI. Alternative window shapes, heavier blur, base placement rules and
penalty grids also fail to provide a uniform quality improvement. These
experiments do not change the defaults.

Applying the blended descriptor to the coarse pass also has mixed results.
Across 5824 comparisons, eight-path coarse blending at band 32 raises
Middlebury clean-image mean error by 0.054 pixels while improving ETH3D by
0.010 pixels. The coarse descriptor therefore remains sparse.

Speckle-filter early exits preserve 224 reference comparisons but depend on
scene content. A row-level exit reduces a constant-plane kernel from 0.147
to 0.087 ms, while random disparities rise from 0.147 to 0.185 ms and mostly
missing maps from 0.146 to 0.172 ms. The original fixed traversal stays in
place.

A slope-aware speckle prototype passes 300 independent reference checks,
but a 9776-comparison sweep finds no consistent advantage over jointly
adjusting the existing support and distance thresholds. It is not retained.

A capped brightness-difference term alongside census also remains an
experiment. Across 5096 comparisons, its best weak setting improves some
Middlebury and ETH3D means but worsens KITTI; stronger weights increase
exposure-related regressions. This is a bounded integer experiment inspired
by combined matching costs, not an implementation of the full
[AD-Census system](https://nlpr.ia.ac.cn/2011papers/gjhy/gh75.pdf).

### Selecting a profile on the board

Start with the complete workload at the intended camera and working sizes.
The local evidence makes these useful candidates, with eight fine paths:

- Band 16, margin 4, three coarse paths and sparse census: the default
  coarse-to-fine baseline with the most local frame-budget headroom.
- Band 16, margin 2, eight coarse paths and blended census: a narrow candidate
  that improves the clean ETH3D and KITTI means in this evaluation.
- Band 32, margin 8, eight coarse paths and blended census: a wider candidate
  with lower Middlebury and KITTI matched-pixel error, requiring more GPU work.

The wider stage configuration is:

```elixir
{NxStereo.Stage.Disparity,
 search: "coarse_to_fine",
 num_disparities: 128,
 paths: 8,
 coarse_paths: 8,
 band: 32,
 margin: 8,
 fine_census: :dense_9x5_blend}
```

Run this on the target when available, with its intended runtime and cooling:

```shell
mkdir -p _build/bench/tmp/stream
TMPDIR="$PWD/_build/bench/tmp/stream" MIX_ENV=test mix run bench/stream.exs --size 544x728 --frame 1088x1456 --search coarse_to_fine --disparities 128 --paths 8 --coarse-paths 8 --band 32 --margin 8 --fine-census dense_9x5_blend --cloud-decimation 4 --packed --frames 1200 --rounds 20 --warmup 24 --mode both
```

Change `--band 16 --margin 2` for the narrow blended candidate; use
`--band 16 --margin 4 --coarse-paths 3 --fine-census sparse_9x7` for the baseline.
Repeat with the intended cloud density. A profile must meet both its throughput
and acceptable latency distribution after including real camera acquisition,
application scheduling and display. Those costs are outside this benchmark.
Use representative captured scenes to select quality settings: these datasets
do not establish a universal ranking for the eventual cameras and scene.

### Final local validation

All 198 tests pass with Vulkan and CPU-IREE enabled. The unchanged native
backend's 59 library and two binary tests also pass. Regenerating all 76 SPIR-V
modules produces identical bytes, and every module passes `spirv-val` for
Vulkan 1.1. A final packed 1456x1088-to-728x544 wide-blend pipeline comparison
reports zero differing elements for all ten stage outputs. That final run is
for correctness only; its single-run, contended timings are excluded from the
performance evidence above.

Formatting and compilation with warnings treated as errors pass. Credo has
25 existing core/test readability findings (26 at the starting revision) and
four existing benchmark findings; there are no new findings. The root
repository's documentation checker passes 42 published guides but does not
cover this standalone library, whose relative file and heading links are
checked separately. The [validation record](benchmarks/vulkan-2026-11/validation.json)
records the checks and their scope.

### Reproducibility checks

Benchmark comparisons count unexpected NaNs and infinity mismatches explicitly.
Expected paired NaNs in stage outputs still agree, as do same-sign infinities;
disparity comparisons require finite values on both sides.

All 2184 saved matcher-quality rows were repeated through the public
`sgm_coarse_to_fine/3` API and matched their recorded metrics. The filter
sweep also repeats exactly. Fine-pass subgroup checks cover 96 combinations
of geometry, path count, band and supported subgroup width, with identical
final outputs: widths 8/16/32 for bands 16/32, and 16/32 for band 64. These
remain Intel checks; the target board still needs its own validation.

Khronos validation layers 1.4.341 run 54 additional checks: 27 with GPU-assisted
storage-buffer instrumentation and 27 with core/synchronization validation.
They cover fine bands 16/32/64 with three/eight paths, the two optional dense
descriptors, and fused full-range three-path matching, including partial tiles.
All report zero errors. A four-byte control buffer deliberately read at index
seven reports out-of-bounds errors; its valid control reports none. These are
correctness checks; their instrumented timings are excluded from performance
results. The [GPU-assisted validation guide](https://vulkan.lunarg.com/doc/view/latest/linux/gpu_validation.html)
describes the checks and their limits.

The current NxIREE dependency names temporary compiler files using a
VM-local counter. Concurrent BEAM processes can collide when sharing a
temporary directory. Run benchmarks sequentially, or assign each process
a different existing `TMPDIR`. The rechecks above use isolated directories;
all reported performance comparisons ran alone.

## The native backend

`cargo run --release --example bench -p stereo_match -- 480 360 64 3 4 30`
in `native/` times the native matching at a width, height, disparities,
paths, threads and runs of your choice. The same rules hold: one change
at a time, `cargo test` and `mix test` to prove the results unchanged,
medians to compare.
