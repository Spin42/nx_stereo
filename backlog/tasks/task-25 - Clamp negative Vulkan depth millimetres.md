---
id: TASK-25
title: Clamp negative Vulkan depth millimetres
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
  - vulkan
dependencies: []
---

## Description

The Nx and Rust depth implementations clamp negative rounded depths to zero before converting to u16. The Vulkan shader converts negative floats directly to uint before packing, which does not establish the same result and may contaminate the neighboring pixel in a packed word.

## Acceptance criteria

- [x] Exercise negative-baseline depth through the public Vulkan API and inspect the conversion portability.
- [x] Match Nx and Rust millimetres without changing metre outputs or valid positive depths.
- [x] Regenerate and validate shader binaries and run focused depth comparisons.

## Design

Clamp rounded millimetres to zero before unsigned conversion in the shader, matching the Nx and Rust contract. Retain zero for overflow beyond 65535 millimetres. Verify both positions in each packed word.

## Portability evidence

Both available local Vulkan implementations currently return zero for the tested negative depths; no host output failure is claimed. The [SPIR-V conversion contract](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertFToU) leaves out-of-range float-to-unsigned conversions undefined. Negative rounded depths are outside the unsigned range. An explicit clamp establishes the same defined behavior as the Nx and Rust implementations before this shader is used on another GPU.
