---
id: TASK-5
title: Benchmark coarse-to-fine quality on ground truth
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - quality
  - benchmark
dependencies:
  - TASK-1
---

## Description

The dataset benchmark only runs full-range matching and skips entire scenes whose ground truth exceeds the search range. It cannot reproduce the coarse-to-fine configuration used by the Vulkan stages or expose the tradeoff between coverage, error and runtime.

## Acceptance criteria

- [ ] Benchmark full or coarse-to-fine matching with explicit paths, band, margin and penalties.
- [ ] Score only finite ground truth within an explicitly selected disparity range and report the fraction scored.
- [ ] Report coverage, error over matched pixels, bad-or-missing pixels and mean absolute disparity error separately.
- [ ] Warm GPU runs before recording median and p90 times; report the device and settings.
- [ ] Export per-scene CSV results for reproducible comparisons.
- [ ] Test metric boundaries, missing estimates and out-of-range truth without downloading a dataset.

## Design

Keep the existing dataset loaders and automatic disparity selection for full-range benchmarks. An explicit disparity range, and coarse-to-fine's fixed 128-disparity range, include all scenes but mask unsupported truth pixels. Crop the last row or column only when coarse-to-fine requires even dimensions. Use one matcher function for both EXLA and Vulkan and compare validity plus subpixel error within tolerance.

Extract the score calculation into a small benchmark helper, tested with synthetic arrays. The command reports equal scene-weighted dataset means, with pixel counts available in CSV. Timing includes the result download and excludes warmup and compilation. It is a matcher benchmark, not a complete camera pipeline or a target-device FPS guarantee.
