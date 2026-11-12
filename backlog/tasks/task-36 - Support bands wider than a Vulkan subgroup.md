---
id: TASK-36
title: Support bands wider than a Vulkan subgroup
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

The existing band-64 penalty-boundary regression fails on llvmpipe. Its subgroup size is fixed at eight, while the wide-band scans and selection exchange sixteen packed words across sixteen subgroup lanes. Intel's subgroup size is 32, which hides the unsupported shuffle sources. A valid Vulkan implementation must not return incorrect disparities because its subgroup is smaller than the band.

## Acceptance criteria

- [x] Reproduce the mismatch and inspect device subgroup capabilities.
- [x] Cover three-path and eight-path band matching at penalty limits on both drivers.
- [x] Preserve the fast subgroup implementation where the whole band fits.
- [x] Use a correct self-contained path where the subgroup is too small.
- [x] Validate shader binaries and the complete regression suite.

## Design

Branch uniformly on gl_SubgroupSize. Where WORDS exceeds it, one invocation owns the complete scan line or selection pixel and performs the same integer recurrence without subgroup communication. Other invocations return. Preserve buffer layouts and dispatch counts, including ten-bit aggregated costs and folded diagonals. The ordinary subgroup path remains available without host device-name assumptions or dependency changes.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
