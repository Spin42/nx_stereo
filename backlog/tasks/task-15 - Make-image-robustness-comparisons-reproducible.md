---
id: TASK-15
title: Make image robustness comparisons reproducible
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - benchmark
  - quality
dependencies:
  - TASK-5
  - TASK-9
---

## Description

Clean-image averages hide tradeoffs under image noise, exposure differences and unequal sharpness. Reproducible perturbations are needed to assess the optional census descriptors without changing ground-truth geometry.

## Acceptance criteria

- [x] Provide explicit clean, noise, right-exposure and right-blur modes with fixed documented random seeds.
- [x] Apply perturbations before compilation/timing and preserve ground truth.
- [x] Record the condition in reports and CSV metadata.
- [x] Verify deterministic independent eye noise, exposure clipping and blur boundaries, then compare a public CLI run with the research sweep.

## Design

Add a benchmark-only image transform module and a fixed --perturbation selection to datasets.exs. Use the seven conditions already measured: clean; independent Gaussian noise with standard deviations 2, 4 or 8 grey levels; right-image gain 0.8 with offset 8, or gain 1.2 with offset -8; and a right-image 3x3 box blur with zero padding. Round and clip to byte intensities. Keep the default clean. This adds no production matching behavior.

## Validation

Five regression tests fail before the helper exists and pass with the implementation. Twenty-one transform comparisons (seven conditions at three shapes) agree exactly with the independent sweep helpers. The public noisy Adirondack run with eight coarse paths, band 32, margin 8 and blended census reproduces coverage, bad-pixel rate, bad-or-missing rate and mean absolute error exactly; its Vulkan output has zero differing pixels against EXLA. The complete suite passes 186 tests, and the separately enabled CPU-IREE chain test passes.
