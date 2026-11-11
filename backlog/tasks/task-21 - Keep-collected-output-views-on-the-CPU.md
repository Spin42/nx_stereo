---
id: TASK-21
title: Keep collected output views on the CPU
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - correctness
dependencies:
  - TASK-16
---

## Description

NxSegment downloads a packed output bundle, then reconstructs each tensor with Nx.from_binary using the global default backend of its collection task. A globally configured EXLA or device backend can therefore allocate backend buffers for data that has just been downloaded. CPU collection should return BinaryBackend views, as image unpacking already does.

## Acceptance criteria

- [x] Reproduce non-BinaryBackend collected outputs with an EXLA global default using the public collect API.
- [x] Return BinaryBackend views for all published tensor types while preserving bytes, shapes and packed-image unpacking.
- [x] Measure collection or complete-stream cost under EXLA and BinaryBackend defaults, without conflating it with device execution time.
- [x] Run relevant collector regressions and the final test suite.

## Design

Keep device download and carry behavior unchanged. Select Nx.BinaryBackend explicitly in unbundle's Nx.from_binary call. Use a synchronous regression test that restores the global default after testing the collection task; cover f32 values including NaNs, u16 depth, scalar counts and packed images. Compare the previous and fixed collectors outside compilation.

## Validation

The final local suite passed all 198 tests with Vulkan and CPU-IREE enabled. Retained measurements and their limitations are recorded in docs/optimizing.md and docs/benchmarks/vulkan-2026-11/. Status remains In Progress until the pull request merges.
