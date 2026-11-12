---
id: TASK-26
title: Define sharpness without horizontal pairs
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
dependencies: []
---

## Description

A single-column image contains no horizontal neighbor pairs, so all six sharpness sums must be zero. The Nx reference attempts a zero-width slice and fails before compilation. Native Rust and the shader naturally sum zero pairs.

## Acceptance criteria

- [x] Reproduce the zero-width slice failure on a single-column image.
- [x] Return six f32 zeros and zero sharpness/brightness for such images.
- [x] Verify byte and s32 images on CPU and Vulkan without changing other image sizes.

## Design

Return the empty-pair sums directly when the statically known width is one. Keep the existing calculation for larger widths. The result is independent of pixel values because no pair exists.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
