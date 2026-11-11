---
id: TASK-19
title: Fuse full-range three-path horizontal scans
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
dependencies: []
---

## Description

The full 128-disparity three-path pipeline writes a forward volume then dispatches a reverse scan. A fused prototype preserves all four matcher outputs and improves the complete changing-frame chain in all ten alternating comparisons of 300 frames per variant. The same fusion does not reliably improve coarse-to-fine chains, so restrict integration to the demonstrated case.

## Acceptance criteria

- [x] Add a fused 128-disparity horizontal shader only for the full three-path pipeline.
- [x] Preserve original shaders and dispatches for 64-disparity and eight-path matching.
- [x] Verify cost and scan volumes, final disparity, consistency, median and complete published outputs against the previous implementation.
- [x] Regenerate and validate SPIR-V and run the full regression suite.
- [x] Save repeated local measurements, explicitly reporting that full-range matching still falls below 30 FPS.

## Design

Extend cost_first.comp behind FUSED=1. Each invocation writes its own forward and raw-cost words, then reads exactly those words in reverse after memoryBarrierBuffer, adding the reverse recurrence. No inter-invocation communication is added beyond the existing subgroup recurrence. Eight-bit paired paths remain bounded by 254. Register one extra cost_horizontal_d128 SPIR-V variant and choose it only when disparities=128 and paths=3. Retain two-dispatch horizontal scans elsewhere.

## Validation

The final local suite passed all 198 tests with Vulkan and CPU-IREE enabled. Retained measurements and their limitations are recorded in docs/optimizing.md and docs/benchmarks/vulkan-2026-11/. Status remains In Progress until the pull request merges.
