---
id: TASK-6
title: Distribute wide-band scans across subgroup lanes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
dependencies:
  - TASK-4
---

## Description

Holding all candidates and a barrel-shift window in one invocation becomes expensive for 32- and 64-disparity bands. Wide bands can improve matching quality, but their scan cost currently prevents useful runtime comparisons with the default band.

## Acceptance criteria

- [ ] Preserve path buffers exactly for stationary, varying and missing bases, all scan directions and supported penalty limits.
- [ ] Preserve full matcher output for 3 and 8 paths, wide and tall images, and partial workgroups.
- [ ] Demonstrate a complete matcher improvement for 32- and 64-disparity bands.
- [ ] Keep existing subgroup-size requirements and the faster narrow-band implementation.
- [ ] Validate all SPIR-V and pass Vulkan reference comparisons.

## Design

Use one subgroup lane per packed word of four candidates. Normalize the previous costs with a reduction within the pixel's lanes, and gather the shifted window with subgroup shuffles. Each invocation holds four previous costs and loads the two words containing its four matching costs. Consecutive lanes write consecutive path words. The required lane group is `band / 4`, already required by band selection.

Keep one invocation per line for bands 8 and 16: local measurements favour that implementation. Share scan geometry between implementations so folded diagonals and border restarts follow the same rules. Target-board timing remains necessary; local correctness does not depend on the target GPU's performance.
