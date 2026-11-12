---
id: TASK-24
title: Replicate borders for subwindow image sizes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
dependencies: []
---

## Description

Image.shift constructs a zero-length or negative-length interior slice when its offset reaches an image dimension. Median filters on a single row, a single column, or a 2x2 image therefore fail despite their replicated-border contract. Dense census uses the same helper.

## Acceptance criteria

- [x] Reproduce the failure for both median windows on small images.
- [x] Replicate the entire nearest edge when the shift reaches or exceeds the dimension.
- [x] Compare CPU and Vulkan results with independently sorted clamped neighborhoods.
- [x] Check dense census callers and normal image sizes for regressions.

## Design

An offset whose magnitude reaches the axis size contains only the replicated edge. Broadcast the appropriate one-element edge slice to the original shape. Keep the existing interior-slice path for smaller offsets, preserving the usual compilation and execution behavior.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
