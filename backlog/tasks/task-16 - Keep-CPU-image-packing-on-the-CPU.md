---
id: TASK-16
title: Keep CPU image packing on the CPU
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - pipeline
dependencies:
  - TASK-13
---

## Description

Packing.pack/3 and unpack_download/1 describe CPU byte views, but Nx.from_binary/2 follows the caller's default backend. With an EXLA default, packing CPU camera bytes creates an EXLA buffer before the Vulkan upload. Unpacking a downloaded image can likewise upload its bytes again.

## Acceptance criteria

- [x] Reproduce unwanted device-backed results with an EXLA default through the public packing functions.
- [x] Keep CPU packing and downloaded-image views on BinaryBackend while preserving all bytes and dimensions.
- [x] Measure packing with CPU and EXLA defaults and preserve compiled pack_image/unpack behavior.
- [x] Run the packing, segment and Vulkan regression tests.

## Design

Select Nx.BinaryBackend explicitly when constructing CPU views from binaries in Packing.pack/3 and unpack_download/1. Keep the compiled tensor packing functions unchanged. The public contract already identifies these operations as CPU views; their placement should not depend on unrelated default-backend configuration.

## Validation

- Both new EXLA-default regressions failed with EXLA-backed results before the fix and pass with exact byte and shape preservation afterward.
- All 32 packing, segment and Vulkan-kernel tests pass. Compiled pack_image/unpack remain unchanged.
- Six alternating rounds of 400 calls at 1456x1088: EXLA-default packing 1.719 ms → 0.00058 ms; unpacking 1.638 ms → 0.00056 ms. BinaryBackend-default timings stay below 0.001 ms for both versions. Raw rounds are in docs/benchmarks/vulkan-2026-11/packing-bench.csv.
