---
id: TASK-18
title: Measure point-cloud density in complete pipelines
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - quality
  - performance
  - benchmark
dependencies:
  - TASK-11
---

## Description

The application stage already supports point-cloud decimation, but both complete-pipeline benchmarks hardcode four pixels per point in each direction. Expose that setting in the benchmarks so the quality/performance budget can include a point at every depth pixel without changing the stage's production default.

## Acceptance criteria

- [x] Expose a positive --cloud-decimation option in stage and changing-frame benchmarks, preserving the default of 4.
- [x] Forward it consistently to the individual point-cloud measurement and the complete stage chain.
- [x] Verify published grid dimensions and reference agreement at decimations 1, 2 and 4.
- [x] Measure complete changing-frame throughput and latency, including the larger download, and document local-only results.

## Design

Use the existing Stage.PointCloud setting, with no new point-cloud algorithm. Put the common validation in Bench.Options and pass the selected value through Bench.Scene. Report the compiled cloud-grid dimensions so a benchmark records its actual output density. Compare serial and overlapped collection because denser output increases download volume.
