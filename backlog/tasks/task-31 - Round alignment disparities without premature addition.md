---
id: TASK-31
title: Round alignment disparities without premature addition
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
  - vulkan
dependencies: []
---

## Description

The row-alignment shader rounds disparity with floor(d + 0.5). For the f32 value immediately below 0.5, the addition rounds to 1.0 before floor, choosing the wrong source column. Identical horizontal ramps consequently produce nonzero alignment costs instead of zero on both Intel Vulkan and llvmpipe. This affects the public diagnostic's accepted positive disparity inputs.

## Acceptance criteria

- [x] Reproduce incorrect alignment sums on both local Vulkan implementations.
- [x] Add a known-answer regression for byte and word images below, at and above a half-pixel boundary.
- [x] Round positive disparity without first adding one half.
- [x] Keep ordinary alignment and complete pipeline checks passing.

## Design

Use the integer part plus a fractional comparison, as in the depth rounding correction. The shader already rejects nonpositive disparities. Keep the output layout, sampling, bands and settings unchanged.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
