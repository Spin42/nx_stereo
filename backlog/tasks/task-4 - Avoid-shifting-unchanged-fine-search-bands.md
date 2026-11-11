---
id: TASK-4
title: Avoid shifting unchanged fine search bands
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
dependencies:
  - TASK-1
---

## Description

Fine SGM scans use a packed barrel shifter to align the previous pixel's candidates with the current search band. Coarse-to-fine matching repeats each coarse band over a 2x2 block, so many consecutive pixels need no alignment.

## Acceptance criteria

- [ ] Preserve every path cost, including stationary bands, changing bands, missing estimates and supported penalty limits.
- [ ] Retain the shortcut only for band widths and directions with a measured local gain.
- [ ] Compare the complete pipeline with the same inputs and settings.
- [ ] Regenerate and validate SPIR-V, and pass the existing Vulkan parity tests.
- [ ] Record local timings and the need for target-board validation.

## Design

When the current and previous band bases are equal, fill the recurrence window directly from the normalized previous costs, with P2 at both ends. Keep the packed barrel shifter for other steps. Both branches feed the same cost update. Specialize the shortcut by band width and direction where measurements justify it; small bands can cost more in branch overhead than they save.

Alternatives are shifting every step, which spends work on an identity mapping, and a dynamically indexed previous-cost array, which spills on the target GPU. Compare output buffers byte for byte before measuring speed.
