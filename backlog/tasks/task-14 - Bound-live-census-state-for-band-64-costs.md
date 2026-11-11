---
id: TASK-14
title: Bound live census state for band 64 costs
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
dependencies:
  - TASK-12
---

## Description

The band-64 cost kernel keeps 64 right census signatures live before calculating Hamming distances. Its cost is disproportionate to narrower bands on the local Intel GPU.

## Acceptance criteria

- [x] Preserve every matching cost and disparity for partial tiles, image orientations, both path counts and all census modes.
- [x] Measure repeated isolated GPU and changing-frame complete-chain performance.
- [x] Limit the change to band 64 when narrower variants do not improve.
- [x] Regenerate and validate the affected SPIR-V and run the relevant regression tests.

## Design

Load four consecutive right signatures at a time, immediately calculate their Hamming distances, and pack them into the existing six-bit cost words. Keep the shared-memory tile and all layouts unchanged. Compare chunks of four and eight before choosing. Preserve the original processing order for narrower bands.

## Prototype evidence

Ninety-six comparisons preserve all costs and disparities across eight geometries, three census modes, both path counts and chunk sizes four/eight. Ten alternating isolated GPU rounds at 728x544 reduce the band-64 kernel from 3.244 to 0.399 ms on constant bases and 4.457 to 0.635 ms on changing bases with chunks of four. Chunk eight is similar. Narrower bands show no consistent gain. Complete changing-frame results determine retention.

## Complete-chain result

Six alternating rounds of 150 changing frames compare the complete 728x544 pipeline, eight coarse/fine paths, band 64 and margin 16. Median throughput is 15.270 FPS with the original kernel, 15.986 FPS with chunks of four, and 16.008 FPS with chunks of eight. Chunk four improves every paired round (1.8–6.4%); chunk eight regresses in one round. Keep four. The median frame period falls from 65.487 to 62.553 ms. Band 64 still falls below 30 FPS on this local Intel workload. All outputs and carried tensors agree across sixteen changing validation frames.

## Integration checks

The integrated shader preserves all 96 cost/disparity comparisons, validates for Vulkan 1.1, and passes the full 186-test suite. Only band_cost_b64.spv changes; narrower cost binaries remain unchanged.
