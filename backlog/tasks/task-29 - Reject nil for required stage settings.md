---
id: TASK-29
title: Reject nil for required stage settings
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
  - settings
dependencies: []
---

## Description

Stage.init_settings skips validation for every nil value, including required numeric and enumeration settings. Several stages therefore return an apparently valid state containing nil parameters that cannot compile or run. Nil is only meaningful for explicitly optional settings such as the inferred coarse path count and matcher selection.

## Acceptance criteria

- [x] Reproduce acceptance of nil for required settings across the declared stages.
- [x] Reject invalid nil values during initialization with the setting's normal validation error.
- [x] Preserve optional nil defaults and explicitly allowed nil values.
- [x] Run stage settings and initialization regressions.

## Design

Skip validation for nil only when the declaration has a nil default. Otherwise use the declared type/range/value checks, which also preserve any explicit nil in an allowed-values list. Check values before cross-setting ordering so an absent penalty is reported at its own key.

## Verification

The complete 246-test suite passes with Vulkan and CPU-IREE enabled, using Intel and llvmpipe Vulkan devices. All 76 shader modules pass Vulkan 1.1 SPIR-V validation. See [the stage validation report](../../docs/validation.md) for scope, commands and remaining limitations.
