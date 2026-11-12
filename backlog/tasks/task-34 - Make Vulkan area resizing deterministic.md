---
id: TASK-34
title: Make Vulkan area resizing deterministic
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-12'
labels:
  - correctness
  - vulkan
dependencies: []
---

## Description

Repeated 14x252 to 7x126 area resizes return incorrect pixels on Intel Vulkan. A 256-frame known-answer probe found failures in 11 byte-input calls and four packed-input calls without validation layers. GPU-assisted validation increases this to 253/256 calls in both modes, without reporting shader buffer errors. Several wrong pixels double the expected value or acquire an extra contribution. EXLA and alternative integer or dedicated shrink implementations remain exact. Single-call parity is insufficient to expose the fault.

## Acceptance criteria

- [x] Reproduce incorrect ordinary Vulkan output with independent expected pixels and retain failing examples.
- [x] Capture a repeated public Decode regression before changing production code.
- [x] Identify the faulty generated reduction and remove its dependence on cross-invocation accumulation.
- [x] Verify integer and noninteger resize ratios, byte and packed inputs, repeated changing frames, and both local Vulkan implementations.
- [x] Recheck complete pipelines and measure any retained performance changes separately.

## Design

Inspect the generated reduction and test an explicit per-pixel accumulation of the known area taps, which avoids a parallel reduction. Preserve area weights, rounding, output shape and storage. Treat the existing 2x integer and packed-word prototypes as separate performance candidates; the correctness fix must also cover noninteger resizing. GPU-assisted shutdown currently crashes in the external validation layer after these probes; separate that teardown issue from the demonstrated wrong pixels in ordinary execution.

## Evidence

The generated first resize shader declares a local size of 1x64x1 but never reads LocalInvocationId. All 64 invocations initialize and accumulate the same storage-buffer pixels with ordinary load/add/store instructions. They race; the resulting values depend on scheduling. Explicit tap addition generates elementwise dispatches with per-invocation indices instead. The original public Decode regression failed at pair 17; the unrolled prototype passes 256 changing pairs for each storage mode.

Across eight resize ratios and both byte/packed inputs, each driver completed 1,808 calls. The fixed implementation had no incorrect pixels, unstable frames or rounding-boundary exceptions on Intel Vulkan and llvmpipe. The original reduction failed five Intel calls in the same audit.
