---
id: TASK-23
title: Decode packed camera frames correctly
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

NxSegment forwards camera frames as packed u32 words when byte storage is disabled. Decode currently treats those words as individual pixels, so a valid working width can be refused as enlargement or the pixel values can be averaged incorrectly.

## Acceptance criteria

- [x] Reproduce the failure through Segment.run with byte storage disabled.
- [x] Decode packed and byte frames to the same independently expected pixels at equal size, integer downscaling and noninteger downscaling.
- [x] Keep decoded pixels in 32-bit storage until the packed output boundary on devices without byte storage.
- [x] Verify CPU and Vulkan segment results and downstream image stages.

## Design

Decode understands the existing packed-frame format just as Rectification does: unpack u32 words to s32 pixels before resizing. Area resizing preserves s32 pixel storage when given s32 pixels, while retaining u8 outputs for byte input. NxSegment already packs s32 image outputs for collection, so no new transport format or segment special case is required.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
