---
id: TASK-2
title: Keep wide-band selection within Vulkan dispatch limits
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - vulkan
  - correctness
dependencies: []
---

## Description

At 728x544 with a band of 64, the selection requests 99008 workgroups on the X axis, above Vulkan's guaranteed 65535 limit. The documented band cannot run at this resolution. The same problem affects narrower bands at higher resolutions.

## Acceptance criteria

- [ ] A 256x1024 image with band 64 compiles and recovers a known stereo shift on Vulkan.
- [ ] All existing band widths and path counts retain reference parity, including partial workgroups.
- [ ] Regenerated SPIR-V validates for Vulkan 1.1.

## Design

Distribute the selection's linear workgroups across X and Y when the count exceeds 65535, and flatten the shader's two-dimensional invocation index. Keep existing dispatch dimensions for smaller images. This preserves output ordering and avoids increasing the local workgroup size, which would change GPU occupancy.
