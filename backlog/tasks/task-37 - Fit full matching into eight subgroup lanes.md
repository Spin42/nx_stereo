---
id: TASK-37
title: Fit full matching into eight subgroup lanes
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

Full 128-disparity matching also assumes sixteen subgroup lanes: each invocation owns eight candidates. Both existing full-range reference tests fail on llvmpipe's eight-lane subgroups. Reads beyond the subgroup cannot provide the recurrence's complete minimum or neighbours, and the upper disparity range is not correctly processed.

## Acceptance criteria

- [x] Reproduce incorrect full-range output on the eight-lane device.
- [x] Cover both path counts, penalty limits and a known shift above 64 disparities.
- [x] Keep the full candidate range and packed cost layouts intact on both drivers.
- [x] Measure the layout change on the hardware GPU before reporting performance.
- [x] Run final regression and shader validation checks.

## Design

Use eight lanes for both supported disparity ranges: eight candidates per lane at 64 disparities, sixteen at 128. The existing packed load/store loops already support multiple words per invocation. Generalize the selection's constant-index gather to the block width. Each workgroup continues to cover eight scan lines, and the external volume layout and dispatch counts stay unchanged. Measure the register/coalescing tradeoff rather than assuming a speed gain.

## Verification

The complete 246-test suite and all 76 SPIR-V validations pass. Six alternating matcher timing rounds are recorded in [the validation report](../../docs/validation.md); timing variation does not support a reliable speed claim.
