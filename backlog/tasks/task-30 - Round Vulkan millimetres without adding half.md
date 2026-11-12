---
id: TASK-30
title: Round Vulkan millimetres without adding half
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

Computing floor(mm + 0.5) can round the addition itself before floor is evaluated. A depth just below 0.5 mm becomes 1 mm on llvmpipe, although the separately returned metre value and the Nx/Rust rounding require zero. An independent audit of 196608 half-millimetre boundary cases reproduced the inconsistency. Intel's division and contraction choices masked this specific case in the initial sweep.

## Acceptance criteria

- [x] Reproduce the metre/millimetre inconsistency on a local Vulkan implementation.
- [x] Capture a focused public depth API regression on llvmpipe.
- [x] Round the scaled f32 depth by its integer and fractional parts, without premature half-addition rounding.
- [x] Preserve negative and overflow handling and verify both available Vulkan implementations.
- [x] Distinguish division rounding differences across devices from inconsistent conversion of the returned metres.

## Design

Materialize the scaled millimetres with precise f32 multiplication. Add one to the integer part only when the fractional part is at least one half. Clamp negative rounded values before unsigned conversion, as required by the Nx/Rust contract and TASK-25. Keep overflow beyond 65535 as zero.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
