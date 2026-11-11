---
id: TASK-1
title: Improve Vulkan stage performance and quality
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - vulkan
  - quality
dependencies: []
---

## Description

Improve the Vulkan stereo stages for a 728x544, 128-disparity pipeline targeting at least 30 pairs per second on the Arduino VENTUNO Q. Local investigation uses the Intel Iris Xe Vulkan device and EXLA references; target performance remains unverified until the board is available.

## Acceptance criteria

- [x] Establish reproducible local test, stage timing and ground-truth quality baselines.
- [x] Reduce measured work or memory traffic without changing outputs, verified against the Nx reference.
- [x] Evaluate quality changes on ground-truth scenes, including depth boundaries and missing matches.
- [x] Preserve public defaults unless evidence supports a change; expose useful quality controls through stages.
- [x] Run applicable formatting, compilation, native and Vulkan tests and validate regenerated SPIR-V.
- [x] Document retained and rejected experiments, local limitations and exact target benchmark commands.

## Design

Start with the existing coarse-to-fine configuration: 3 coarse paths, 8 fine paths and a band of 16. Separate exact shader optimizations from algorithm changes. Reduce redundant intermediate traffic and dispatches before changing image size or path count. Assess algorithm changes against ground truth with both error and coverage, rather than agreement with another matcher alone. Keep target-dependent changes provisional until measured on the Adreno 623.

Alternative approaches are full-resolution full-range SGM (high bandwidth), fewer paths or lower resolution (quality loss), and wider fine bands (more bandwidth). Compare their quality and local cost where useful; retain only changes backed by tests and measurements.

## Further quality experiments

The edge-adaptive P2 prototype is not retained. Its independent CPU recurrence agrees in 42 geometry/parameter checks, but 2912 ground-truth comparisons show a tradeoff: thresholds of 15/30/60 grey levels and penalty divisors 2/4 improve Middlebury while generally worsening ETH3D and KITTI. For eight coarse paths, band 32 and the blended descriptor, threshold 30/divisor 4 lowers Middlebury bad-or-missing rate by 0.474 percentage points and EPE by 0.0721 pixels, but raises ETH3D by 0.327 points/0.0244 pixels and KITTI by 0.167 points/0.0011 pixels. The added shader input and branch do not justify a default change. Controls reproduce the earlier quality sweeps.

## Further exact-kernel experiments

Wide-scan workgroups of 32/128/256 threads preserve all compared outputs but provide no consistent benefit over 64. Speckle-filter early-exit variants preserve 224 CPU-reference cases; a row-level exit improves a constant plane from 0.147 to 0.087 ms but regresses random maps from 0.147 to 0.185 ms and mostly missing maps from 0.146 to 0.172 ms. Neither change is retained.

A coarse-descriptor sweep covers 5824 comparisons. With eight coarse paths, band 32 and blended fine census, using blended coarse census raises Middlebury clean-image mean error by 0.0542 pixels despite a small bad-or-missing reduction. ETH3D improves by 0.0103 pixels, while KITTI changes by +0.0005 pixels. The sparse coarse descriptor stays unchanged.

## Additional quality and portability checks

- All 2184 saved matcher-quality rows reproduce through the public API with per-process compiler temporary directories. NxIREE's VM-local temporary-name counter can collide across concurrent BEAM processes, so local experiments now use distinct TMPDIR paths. Timed comparisons always run alone.
- The slope-aware speckle prototype passes 300 independent reference cases. A 9776-row sweep of ordinary support/distance settings versus limited slopes does not justify adding the gradient calculation; it is not retained. Existing 14/1.5 settings improve coverage and bad-or-missing rate but increase matched-pixel error versus 12/1.0; both are documented as a tradeoff.
- A bounded intensity-plus-census cost passes 84 independent packed-cost checks and 728 zero-weight public-matcher controls. Its 5096-row, seven-condition sweep shows dataset-dependent regressions; no new cost mode is retained.
- Current fine shaders preserve four final outputs across 96 subgroup-width comparisons: three geometries including partial and tall images, smooth/changing bases, bands 16/32/64 and three/eight paths. Supported widths are 8/16/32 (band 64 uses 16/32). These are local Intel checks.

## Validation and handoff

The local validation record and retained performance/quality evidence are in docs/benchmarks/vulkan-2026-11, with interpretation in docs/optimizing.md. All acceptance criteria have local evidence. Arduino performance remains deliberately unmeasured in this phase. Status remains In Progress until the pull request merges.
