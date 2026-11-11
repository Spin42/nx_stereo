---
id: TASK-7
title: Reduce row alignment workgroup state
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
dependencies:
  - TASK-1
---

## Description

Row alignment accumulates four image bands in each workgroup. Each invocation keeps 24 integer sums and dynamically indexes the current band, although each sampled pixel belongs to just one band.

## Acceptance criteria

- [x] Preserve the diagnostic sums exactly for byte and word images, partial rows and widths not divisible by the band count.
- [x] Measure the kernel against the previous shader with identical inputs.
- [x] Validate SPIR-V and the existing diagnostic reference tests.

## Design

Dispatch one workgroup per sampled row and band. Each invocation keeps six sums, and the workgroup reduces only its own band. Write the same row-major partials for the unchanged column reduction. Integer accumulation preserves the result regardless of reduction order. Keep the existing policy of excluding trailing columns outside the equal-width bands.

An alternative is a subgroup-only reduction. It would introduce subgroup assumptions into this diagnostic kernel; reducing the state preserves the existing portable shared-memory reduction.

## Validation

All 12 Vulkan kernel reference tests pass. Row-alignment sums match exactly for byte and word images at 544x728, 272x364 and 31x68, including missing disparities. Further comparisons at 2x5, 3x5, 5x6, 9x11, 31x70 and 31x71 preserve the sums with partial bands and rows. Both regenerated SPIR-V modules validate for Vulkan 1.1.

On Intel Iris Xe (Mesa 26.0.8), GPU timestamps at 544x728 measure 0.069 to 0.025 ms for bytes and 0.083 to 0.040 ms for words. The benchmark alternates shaders on the same input buffers and compares the partial sums byte for byte. These are kernel measurements, not target-board frame times.
