---
id: TASK-20
title: Evaluate resident stage partitioning at larger working sizes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - performance
  - investigation
dependencies:
  - TASK-11
---

## Description

The complete compiled chain approaches the runtime's 64 MiB allocation-cache boundary. A larger working image or a different published cloud layout may cross it. Evaluate whether compiling contiguous stage groups separately reduces peak scratch allocation enough to justify the extra calls, while keeping intermediate tensors and carried state on the same Vulkan device.

## Acceptance criteria

- [x] Compare monolithic, two-part and three-part compilation without changing stage algorithms or published outputs.
- [x] Verify every published output and the final carried tensors across changing frames before timing.
- [x] Inspect allocation sizes and compare complete-chain throughput at current and larger working resolutions.
- [x] Record the measured tradeoff; retain production changes only if complete-chain improvements reproduce.

## Design

Prototype outside production code. Keep the normal NxSegment upload, await, carry and collect behavior, replacing only its compiled function with a composition of compiled contiguous stage groups. Split after rectification and optionally after matching, preserve original global parameter/carry indices, and bundle published outputs in the final group. Intermediate tensors stay on the resident Vulkan backend. Compare against the original monolithic function with alternating runs. A rejected experiment adds evidence only.

## Findings

All four working sizes passed complete-output and carry parity. Splitting after rectification moves the 768x576 scratch allocation below 64 MiB and raises median overlap throughput from 23.693 to 30.473 FPS in four balanced comparisons. It regresses the original 728x544 throughput and still misses many individual frame deadlines. The prototype and full evidence are saved under docs/benchmarks/vulkan-2026-11; the general NxSegment API remains unchanged.
