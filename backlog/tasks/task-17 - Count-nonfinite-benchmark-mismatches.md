---
id: TASK-17
title: Count nonfinite benchmark mismatches
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - quality
  - validation
dependencies:
  - TASK-5
---

## Description

The dataset and stage benchmark comparisons rely on a numerical error being greater than a tolerance. A NaN error makes that comparison false, so an unexpected NaN can be reported as zero differing elements. Relative bounds can also hide an infinity mismatch. Reliable quality verification must count these failures explicitly.

## Acceptance criteria

- [x] Reproduce unexpected NaN and infinity mismatches escaping the existing comparisons.
- [x] Preserve finite tolerances and disparity-validity checks.
- [x] Let stage outputs match expected NaNs and same-sign infinities while rejecting other nonfinite mismatches.
- [x] Use the checked helpers in both benchmarks and run their regression tests.

## Design

Extract the existing comparisons into bench/support/compare.exs, preserving their behavior first so regressions demonstrate the failure. For disparity, require finite values on both sides in addition to the existing tolerance and match-validity checks. For general stage output, accept exact equality, paired NaNs, or finite values within the existing relative bound. Keep the stereo implementation unchanged.

## Validation

Three regressions fail with the extracted original comparison logic: unexpected NaNs report 0 instead of 2 differences, an infinity case reports 1 instead of 3, and a disparity case reports 2 instead of 3. After the fix all nine comparison/quality tests pass. A public packed, blended, band-32, eight-path pipeline smoke test at 96x64 reports zero differences for every stage under the stricter comparison. Full-size checks and the complete suite follow the final shader selection.
