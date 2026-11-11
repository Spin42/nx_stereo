---
id: TASK-8
title: Fuse horizontal fine search scans
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - memory
  - vulkan
dependencies:
  - TASK-4
  - TASK-6
---

## Description

The forward and reverse horizontal scans allocate separate path volumes. At 728x544 with a band of 16, the complete segment's scratch allocation is 67,721,728 bytes, just above the [IREE Vulkan allocator's 64 MiB pooled-allocation limit](https://github.com/iree-org/iree/blob/89ece87b89852100c0a4ac3d515e14842d090967/runtime/src/iree/hal/drivers/vulkan/allocator.c#L25-L26). A controlled width comparison shows a local compiled-call increase from about 10.5 ms at width 720 to 29.0 ms at width 728.

## Acceptance criteria

- [x] Preserve path sums and final disparities exactly for every band, changing bases, partial workgroups and penalty limits.
- [x] Preserve both 3- and 8-path matching and the complete segment's published and carried outputs.
- [x] Reduce the complete segment's scratch allocation and measured runtime at the original resolution.
- [x] Validate SPIR-V and pass existing reference tests.
- [x] Record GPU-only and complete-chain results separately, with target timing still pending.

## Design

One invocation, or one lane group for a wide band, owns a complete image row. It scans forward into the output volume, then scans backward and adds each pixel's first path to the reverse path in that same output. The two path costs fit in one byte because each is at most 127. A memory barrier orders the first traversal's writes before the second reads them. No other invocation accesses those output words.

Keep the column-major layout and the existing selector. This removes a dispatch and one full path allocation without changing the recurrence, candidate bands or matching decisions. Combining the other directions saves more memory but has a larger local kernel cost; retain the simpler horizontal change first.

## Validation

All 155 tests pass with Vulkan enabled. Before integration, 192 comparisons cover four image sizes, every band, four penalty pairs including (0, 1) and (78, 79), and horizontal, vertical or combined fusion. Every path sum and final disparity matches byte for byte. The complete segment's published outputs and carries also match. All generated shaders validate for Vulkan 1.1.

At 728x544, band 16, fine 8 paths and coarse 3 paths, scratch storage falls from 67,721,728 to 61,385,216 bytes. Six alternating local rounds on Intel Iris Xe give median compiled calls of 25.386 to 10.590 ms and complete serial segment runs of 32.663 to 18.255 ms. The large gain comes from staying within IREE's allocation pool; GPU-only horizontal scans remain close (1.132 to 1.197 ms with smooth bases and 1.208 to 1.218 ms with changing bases). These local results do not establish Adreno performance.
