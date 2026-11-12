---
id: TASK-32
title: Measure paired area resizing for Decode
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - research
  - quality
dependencies:
  - TASK-22
---

## Description

Packed Decode now produces correct pixels through the generic area resizer. Measure whether the existing paired 2x shrink shader can reduce its cost at the common 1456x1088 to 728x544 size without changing any pixel or imposing byte-storage requirements.

## Acceptance criteria

- [x] Compare full output bytes against the current correct implementation on changing images and boundary sizes.
- [x] Measure alternating runs with resident inputs, including output transfer, without competing workloads.
- [x] Check complete Decode-based pipelines before retaining any optimization.
- [x] Retain only reproducible improvements, with CPU fallback and supported storage modes preserved.

## Design

Prototype outside production code using the existing shrink pipeline before adding or changing an API. If unpacking traffic dominates, separately measure a kernel that reads packed words directly. Keep noninteger scaling on the existing area integration path.

## Selected implementation

Use a paired Vulkan shader that averages 2x2 blocks directly from the byte-compatible camera words. Enable it only for exact twofold scaling of matching u8/u32 frame layouts. Its source extent is always a whole number of words, including odd output widths in byte mode. Preserve the existing generic Nx resize for other ratios and CPU execution.

The three viable implementations completed 3,072 known-answer calls across Intel and llvmpipe, including every possible four-byte sum. A simpler strided Nx prototype is excluded because it fails IREE compilation for a 1x1024 output.

Across six alternating 200-frame complete-pipeline runs per mode, packed wide-profile median overlap throughput changes from 30.777 to 33.766 FPS and the packed fast profile from 57.207 to 63.165 FPS. All published bytes and carried state agree across sixteen changing frames before timing, for both storage modes and both profiles. Preserve the generic resize race regression even though exact-2x Decode now bypasses it.
