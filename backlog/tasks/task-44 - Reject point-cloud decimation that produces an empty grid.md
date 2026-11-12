---
id: TASK-44
title: Reject point-cloud decimation that produces an empty grid
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-12'
labels:
  - correctness
  - validation
dependencies:
  - TASK-22
---

## Description

PointCloud initialization accepts decimation four for a 2x3 working image. Execution then raises an Nx slice error because the output has zero rows or columns. Reject configurations that cannot produce the documented nonempty organized tensor at the public stage boundary, and give direct Depth.point_cloud callers an explicit argument error.

## Acceptance criteria

- [x] Capture the accepted-invalid configuration in failing public-boundary regressions.
- [x] Reject a stage decimation larger than either working dimension before compilation.
- [x] Reject nonpositive, noninteger or oversized direct-call decimation with a descriptive ArgumentError.
- [x] Preserve legal equality and remainder cases and rerun affected and full pipeline checks.

## Design

Keep the existing floor-sized sampling grid and default decimation. Stage.init checks the validated setting against the working geometry; Depth.point_cloud validates its static decimation against the tensor shape before building kernels. Do not create empty Nx tensors or silently change sampling density.

## Validation

Three public-boundary regressions fail before the fix; the legal-grid control passes. The fix passes 17 focused depth/settings tests and 18 native-routing/contract tests. The native-routing fixture now explicitly chooses decimation two for its two-row geometry; its native-acceptance assertions remain unchanged. The final full run passes 251 tests. The initial rerun with an absent temporary directory was discarded, and a subsequent run exposed the invalid routing fixture before the clean final run.
