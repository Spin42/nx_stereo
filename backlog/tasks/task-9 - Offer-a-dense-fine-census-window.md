---
id: TASK-9
title: Offer a dense fine census window
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - quality
  - vulkan
dependencies:
  - TASK-5
---

## Description

A dense 9x5 census descriptor for the fine search improves average bad-or-missing rate and disparity error on Middlebury, ETH3D and KITTI when the coarse search keeps its existing sparse 9x7 descriptor. The comparison uses 52 scenes and the same penalties, path counts, band and margin.

## Acceptance criteria

- [x] Keep the current sparse descriptor as the default.
- [x] Expose the dense descriptor for band matching and the fine pass of coarse-to-fine matching, including the disparity stage and dataset benchmark.
- [x] Reject unsupported settings before compilation and preserve the coarse descriptor.
- [x] Match the Nx reference on Vulkan, including partial feature tiles and all supported bands.
- [x] Record per-dataset quality, coverage and local cost, and test sensitivity to sensor noise, exposure and focus differences.

## Design

Add `:census` to band matching and `:fine_census` to coarse-to-fine matching and the disparity stage. Supported values are `:sparse_9x7`, `:dense_9x5` and `:dense_9x5_blend`. The first two compare each neighbour with the centre using the existing one-level threshold. The blended variant averages the centre with its rounded 3x3 Gaussian mean (weights 1,2,1 in each axis), rounds that average, then applies the same comparison. The dense window has 44 comparisons and fits the existing 6-bit cost format and penalty limits. Use a dedicated census shader and the same scan, selection and postprocessing stages.

Keep full-range matching and the coarse pass unchanged. The option is a static compilation choice, not a live parameter. Do not infer a universally better default from scene averages; publish the tradeoffs and keep the alternative opt-in.

## Additional quality evidence

The dense window alone is sensitive to added sensor noise. Blending its comparison centre with a local Gaussian mean improves average bad-or-missing rate and matched disparity error in all three datasets for all seven tested conditions at band 32 and eight coarse paths: clean, independent Gaussian noise of standard deviation 2/4/8, right exposure gain 0.8/1.2, and right-image blur. This covers 2184 comparisons across 52 scenes, both coarse path counts and three descriptors. Coverage can still decrease, and individual scenes can regress. Keep the default unchanged and expose both alternatives for measured selection.

The blended anchor can be computed exactly as `(16 * centre + weighted_sum + 24) / 32` with integer division, where `weighted_sum` is the unrounded Gaussian sum. This preserves the two specified rounding operations. Both images fit in independent 16-bit arithmetic lanes because the maximum sum is 8184.
