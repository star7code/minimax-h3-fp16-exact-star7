# Changelog

## 1.0.0 - 2026-08-14

- Initial public release.
- Added a workflow-scoped `MODEL -> MODEL` MiniMax H3 FP16 exact-math patch.
- Preserved FP32 residual math and exact power-of-two overflow protection.
- Added automatic bypass for native-BF16 NVIDIA GPUs and slow-FP16 sm61 GPUs.
- Added Comfy Registry metadata and upstream attribution.
