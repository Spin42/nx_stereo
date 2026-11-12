---
id: TASK-35
title: Preserve trailing byte pixels in Vulkan diagnostics
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-12'
labels:
  - correctness
  - vulkan
dependencies: []
---

## Description

On llvmpipe, sharpness loses the last pixels of byte images whose total byte count is not divisible by four. Five independent shape cases fail, while the same images stored as s32 pixels pass. The byte shader reads uint words, including an incomplete word beyond the declared byte-buffer extent. Inspect rectification and row alignment, which use the same storage access pattern.

## Acceptance criteria

- [x] Reproduce trailing-byte loss through the public stages with known answers.
- [x] Verify the word-sized read against padded input and identify every affected dispatch.
- [x] Keep supported byte-image shapes correct without relying on out-of-range word reads.
- [x] Validate Intel and llvmpipe, byte and s32 storage, and complete repeated stage checks.

## Design

For diagnostic byte inputs with an incomplete final word, widen the image to s32 before the existing word kernel. This preserves the dedicated reductions and avoids a new byte-storage capability requirement. For rectification, use its existing generated Nx implementation when the source byte extent cannot support whole-word loads. Keep ordinary camera geometries on their existing kernels. Capture failing public regressions before applying guards.

## Evidence

All three focused public-stage tests fail on llvmpipe: Sharpness, RowAlignment and Rectification. A direct unchanged sharpness shader reads a 3x5 image from either 15 bytes or the same bytes plus one zero padding byte. The 15-byte input loses three neighbour pairs on llvmpipe (9 instead of 12); the 16-byte padded input gives the exact expected sums. Intel returns the expected sums in both cases, demonstrating why a single driver missed the invalid whole-word read.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
