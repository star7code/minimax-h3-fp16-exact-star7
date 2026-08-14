# Changelog

## 2.0.0 - 2026-08-15

- Added the recommended creation-time `MiniMax H3 Native FP16 Loader - Star7`.
- Preserved MixedPrecisionOps INT8/ConvRot dispatch by keeping forced weight casting disabled for quantized models.
- Made the existing `MODEL -> MODEL` node quantization-aware without changing its class ID.
- Added concise backend, mode, force-cast, weight-patch, and block-count diagnostics.
- Documented the difference between creation-time FP16, post-load FP16, and quantized compute.
- Documented the dynamic/low-VRAM LoRA dequantization caveat.

## 1.0.0 - 2026-08-14

- Initial public release.
- Added a workflow-scoped `MODEL -> MODEL` MiniMax H3 FP16 exact-math patch.
- Preserved FP32 residual math and exact power-of-two overflow protection.
- Added automatic bypass for native-BF16 NVIDIA GPUs and slow-FP16 sm61 GPUs.
- Added Comfy Registry metadata and upstream attribution.
