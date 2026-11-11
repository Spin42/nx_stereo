---
id: TASK-13
title: Preserve caller owned tensors when placing segment inputs
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-11'
labels:
  - correctness
  - pipeline
dependencies:
  - TASK-11
---

## Description

The changing-frame benchmark exposed a deleted EXLA buffer while building a segment from reusable calibration tensors. Device placement transfers, and therefore deallocates, caller-owned tensors. A camera shared by both eyes or reused during calibration reload can lose its buffers. Uploaded frames have the same ownership problem when they already live on another backend.

## Acceptance criteria

- [x] Reproduce the failure using shared EXLA camera tensors and reusable frames through the public segment API.
- [x] Preserve caller-owned calibration and frame tensors across build, requests and reload.
- [x] Retain the existing fast path for inputs already on the IREE device.
- [x] Verify ordinary CPU frame behavior and repeated changing-frame execution.

## Design

Copy external tensors into the segment's placement instead of transferring ownership. A caller never explicitly requested deallocation through build, reload or send_request. Already resident IREE tensors remain resident. Binary-backend inputs take the same upload route, while EXLA sources stay valid for their owner.
