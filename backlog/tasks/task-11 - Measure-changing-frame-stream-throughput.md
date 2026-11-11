---
id: TASK-11
title: Measure changing frame stream throughput
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - benchmark
  - performance
dependencies:
  - TASK-8
  - TASK-10
---

## Description

The stage benchmark times a complete chain but repeats the same input and warmed carry. The application carries state between frames and can overlap output download with the next computation. Measure that behavior before recommending a configuration for 30 FPS.

## Acceptance criteria

- [x] Share the existing synthetic scene and complete stage configuration between benchmarks.
- [x] Use changing CPU frames, upload each pair, propagate carry, and download every published output.
- [x] Check serial and overlapped output equality across a repeated sequence before timing.
- [x] Report throughput, completion intervals and frame latency separately, including the 33.33 ms budget.
- [x] Measure narrow and wide bands with three and eight coarse paths locally.

## Design

Precompute a deterministic sequence of textured stereo pairs with changing noise and disparity, outside the timed region. Run either serially or with one computation overlapping the preceding download. Await each computation before passing its carry into the next request; never have two computations using the same old carry. Bound the pipeline to one computation and one download.

Compile and warm up before measuring. For validation, restart carry and compare every published tensor from both modes byte for byte. For timing, still collect each complete output but retain only timestamps. Report sustained FPS from elapsed wall time, completion-period percentiles and upload-to-collected latency percentiles. Camera capture and display are outside this benchmark and no local result establishes target-board throughput.
