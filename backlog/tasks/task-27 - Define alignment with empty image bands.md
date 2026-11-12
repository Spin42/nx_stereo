---
id: TASK-27
title: Define alignment with empty image bands
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
dependencies: []
---

## Description

Row alignment divides the image into equally sized bands, excluding remainder columns. With fewer columns than bands, all bands have width zero. Nx attempts an empty slice and crashes; native and Vulkan summation naturally produce zero samples.

## Acceptance criteria

- [x] Reproduce the failure for byte and s32 images on CPU and Vulkan.
- [x] Return correctly shaped zero sums and no measured offset for empty bands.
- [x] Cover default and custom band counts and preserve ordinary-size results.

## Design

When the static width is below the requested band count, return an f32 tensor of zero sums with shape {bands, 2 * reach + 2}. Otherwise retain the existing default shader and custom-options Nx paths.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
