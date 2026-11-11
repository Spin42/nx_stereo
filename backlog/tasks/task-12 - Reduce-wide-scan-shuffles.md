---
id: TASK-12
title: Reduce wide scan shuffles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
dependencies:
  - TASK-10
---

## Description

The packed wide pipeline meets 30 FPS only narrowly on the local changing-frame workload. Each scan step spends several subgroup operations arranging its ten-bit aggregate, even when the disparity band does not move.

## Acceptance criteria

- [x] Preserve all path costs and disparities at both band widths, path counts, image orientations and penalty limits.
- [x] Measure isolated GPU and complete changing-frame performance with alternating variant order.
- [x] Choose optimizations separately for bands 32 and 64 when their results differ.
- [x] Validate regenerated shaders and the integrated regression suite.

## Design

Keep ten bits per candidate but store each lane's low 32 bits together, followed by one shared high-byte word for each four lanes. Two XOR shuffles combine those four bytes, replacing six arbitrary shuffles and the bit-position arithmetic. The selector and following aggregation stages decode the same layout. Storage remains unchanged.

For band 32, test a shortcut for unchanged bases: normalize the lane's word directly, then use relative subgroup shuffles for its preceding and following candidates. Keep the general recurrence for base changes and for band 64 if its additional branch does not help reliably.

## Prototype evidence

Five variants preserve every compared cost and disparity in 320 cases: four geometries, bands 32/64, paths 3/8, and four penalty settings. All five also preserve every published output and final carry over sixteen changing complete frames.

Eight alternating GPU rounds compare complete fine pipelines on constant and block-varying bases. Band 32's median drops from 20.324 to 15.752 ms on constant bases and 21.590 to 16.709 ms on changing bases with both changes and relative shuffles. Band 64 favors packing alone on changing bases (42.680 to 36.623 ms); adding the relative shortcut there regresses to 41.145 ms. Complete-chain measurements are recorded separately before integration.

## Complete-chain result

Four alternating rounds of 200 changing frames per variant at 728x544, eight coarse paths, eight fine paths, band 32 and margin 8 give median throughput 31.101 to 33.993 FPS with packing and the relative shortcut. Completion intervals above 33.33 ms fall from 220/800 to 44/800; this remains an unpaced local throughput test, not a target-board or camera-latency guarantee. Every output and final carried tensor agrees across sixteen validation frames. Three-path tests show no material regression for band 32; band 64 keeps its general recurrence.
