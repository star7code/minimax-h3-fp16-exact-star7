# Changelog

## 2.0.4 - 2026-08-17

- Added automatic support for native MiniMax H3 checkpoints wrapped with a
  `model.diffusion_model.` prefix and legacy file-level quantization metadata.
- Verified full loading of the third-party `10Eros_Max` INT8/ConvRot model as
  native FP16 MixedPrecisionOps: 50 blocks and 200 ConvRot layers.
- Preserved the existing behavior for official unprefixed checkpoints and
  checkpoints that already contain embedded `comfy_quant` configurations.

## 2.0.3 - 2026-08-15

- Replaced the external Jjk-Nodes prompt box in the packaged workflow with ComfyUI's built-in multiline text node.
- Preserved the prompt node ID, position, size, content, and downstream link.

## 2.0.2 - 2026-08-15

- Removed NVIDIA RTX Video Super Resolution from the packaged RTX 20-series workflow.
- Connected VAE decode directly to video output for a model-free postprocessing path.

## 2.0.1 - 2026-08-15

- Added a packaged RTX 20-series FP16 + activation-chunk example workflow.
- Added Chinese setup, dependency, attention-backend, and connection guidance.
- Added Registry metadata to the bundled workflow and removed the local reference-image filename.

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
