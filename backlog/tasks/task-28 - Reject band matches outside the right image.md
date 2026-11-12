---
id: TASK-28
title: Reject band matches outside the right image
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
  - matching
  - vulkan
dependencies: []
---

## Description

A band can lie wholly above a pixel's column, so every candidate points outside the right image. Selection adds a large cost to those candidates but still applies uniqueness to the resulting finite costs. With uniqueness 1.0, aggregation can create a unique winner among impossible candidates. The consistency check cannot reject it because its projected right coordinate is outside the image. An independent audit found impossible output disparities in all four band sizes with either path count.

## Acceptance criteria

- [x] Reproduce impossible disparities on Vulkan with supported band/path/uniqueness settings.
- [x] Reproduce the same defect in the Nx reference with a focused test.
- [x] Reject impossible winners before consistency and median filtering in both Nx and Vulkan.
- [x] Check all band widths and path counts against geometric bounds and CPU/Vulkan parity.
- [x] Preserve ordinary matching results and measure any performance effect.

## Design

Require the winning integer disparity to be no greater than its pixel column before accepting it. Cost masking remains necessary to prefer valid candidates when a band crosses the image edge. The geometric acceptance condition handles bands with no valid candidates and does not depend on the uniqueness threshold. Apply the condition to the Nx band selector and its Vulkan shader; full search always includes disparity zero and does not need a separate change.

## Verification

The complete 246-test suite and all 76 SPIR-V validations pass. Six alternating matcher timing rounds are recorded in [the validation report](../../docs/validation.md); timing variation does not support a reliable speed claim.
