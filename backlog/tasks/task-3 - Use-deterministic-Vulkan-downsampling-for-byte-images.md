---
id: TASK-3
title: Use deterministic Vulkan downsampling for byte images
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - vulkan
  - correctness
  - quality
dependencies: []
---

## Description

The byte-image coarse-to-fine path lowers its 2x2 average through generic IREE reductions. On the Intel Vulkan driver, partial reduction tiles at sizes including 128x258 and 480x722 produce incorrect values that change between calls. The same algorithm on s32 images uses the dedicated shrink shader and matches EXLA. Incorrect coarse estimates exclude the true disparity from the fine search.

## Acceptance criteria

- [ ] Repeated u8 coarse-to-fine matching at 128x258 agrees with EXLA.
- [ ] Both input types use the existing integer 2x2 average and preserve their final disparity results.
- [ ] Existing Vulkan, native and reference tests pass.
- [ ] Quality experiments use the validated production downsampling path.

## Design

Convert both images to s32 before the internal shrink block and use the existing pair shader for either input type. Its Nx fallback computes the same rounded integer average on other backends. This avoids the problematic generic reduction and reuses the conversion already needed by matching; no new shader or approximation is needed.
