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

## The native backend

`cargo run --release --example bench -p stereo_match -- 480 360 64 3 4 30`
in `native/` times the native matching at a width, height, disparities,
paths, threads and runs of your choice. The same rules hold: one change
at a time, `cargo test` and `mix test` to prove the results unchanged,
medians to compare.
