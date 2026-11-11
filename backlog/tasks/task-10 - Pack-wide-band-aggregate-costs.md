---
id: TASK-10
title: Pack wide band aggregate costs
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - memory
  - vulkan
dependencies:
  - TASK-6
  - TASK-8
---

## Description

Eight-path matching with a band of 32 retains seven byte volumes until selection. The 728x544 complete segment allocates 121.6 MB of scratch storage and takes about 44.5 ms locally. Wider bands can improve disparity estimates but currently exceed the desired frame budget.

## Acceptance criteria

- [x] Preserve every aggregate cost and final disparity for bands 32 and 64, all image orientations, partial workgroups and supported penalty limits.
- [x] Preserve three-path and narrow-band implementations.
- [x] Reduce the band-32 complete segment's scratch allocation below 64 MiB and measure the complete-chain benefit.
- [x] Validate SPIR-V, reference parity and repeated frames with changing data and carry state.
- [x] Record local throughput and latency without treating them as target-board guarantees.

## Design

Process opposite non-horizontal paths together, with each lane owning four candidates of a line. The first traversal writes its path bytes into its output; the reverse traversal adds those bytes and the preceding pair's aggregate. Store the cumulative cost in 10 bits per candidate: eight paths of at most 127 fit within 1023. The selector reads the final aggregate directly.

A lane holds four costs as 40 bits. Three subgroup shuffles assemble each output word from the relevant lane's low and high parts and the next lane's low part. All lanes participate before the extra words are conditionally written. Each lane reads its first traversal's temporary word before overwriting that word; no other lane overwrites it first. Folded diagonals map the opposite line by `(2 * width - height - line) % width`.

Keep one-lane-per-line scans for narrow bands and the existing independent vertical scan for three paths. Combining pairs without compression still exceeds the runtime pool limit; 16-bit aggregates also exceed it.

## Prototype evidence

Twenty thousand randomized packing cases cover sums up to 1016. Thirty-two GPU comparisons cover four geometries, two bands and four penalty pairs, including (0, 1) and (78, 79); all aggregate costs and disparities agree exactly. Complete segments with three or eight coarse paths preserve published outputs and carry tensors.

At band 32, margin 8, eight fine paths and three coarse paths, scratch falls from 121,582,080 to 61,385,216 bytes. Six alternating rounds on Intel Iris Xe give median compiled calls of 36.357 to 20.906 ms and serial complete-chain runs of 44.490 to 28.672 ms. With eight coarse paths, serial-chain medians are 44.974 to 33.898 ms. Further measurement must distinguish streaming throughput from serial latency.

## Integration checks

A regression using unrelated images and alternating disjoint bands fails for both band widths when the aggregate selector discards its tenth bit (penalties 0/79). Both tests pass with the integrated ten-bit decoder. Thirty-two comparisons against the original independent-path implementation preserve every aggregate cost and disparity across four geometries and four penalty settings. All 73 current SPIR-V modules validate. The shared wide recurrence prevents the ordinary and accumulating scans from drifting apart.
