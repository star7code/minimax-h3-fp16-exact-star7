# MiniMax H3 FP16 Exact Fix - Star7

[中文](README.md) · [Example workflows](examples/workflows)

Native FP16 model loading and scoped numerical protection for ComfyUI MiniMax H3 on pre-BF16 architectures. On SM80+, the loader explicitly corrects H3 to native BF16 even when the launcher globally requests FP16. Quantized checkpoints retain eligible INT8/ConvRot kernels instead of being expanded into resident dense FP16 weights.

The overflow-protection method is derived from the MIT-licensed [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix). This package adds a native H3 loader, quantization-aware dispatch, architecture checks, scoped ModelPatcher integration, diagnostics, and workflow support.

## Main features

| Feature | Description |
|---|---|
| Native FP16 loading | Selects FP16 compute while creating MiniMax H3; no `--fp16-unet` launch flag is required |
| Numerical protection | Protects residual, SwiGLU, attention `out_proj`, and MLP `fc2` boundaries with their required FP32 regions and overflow scaling |
| Quantized-path retention | Keeps `force_cast_weights=false` for native MixedPrecisionOps models, preserving eligible INT8/ConvRot kernels |
| Format recognition | Supports native H3 weights, `model.diffusion_model.` prefixes, and file-level `_quantization_metadata` |
| Architecture handling | Always corrects H3 to explicit BF16 on SM80+ and skips the fix on low-FP16-throughput SM61 |
| Lifecycle safety | Installs the fix only on a cloned ModelPatcher and uses weak method binding to avoid retaining obsolete models |
| Diagnostics | Reports load mode, quantization format, force-cast state, weight patches, and DiT block count |

## How it works

MiniMax H3 needs protection at several numerically sensitive points under FP16:

- `condition_proj` receives FP32 input;
- residual streams across the 50 DiT blocks remain FP32;
- normalization output enters attention and MLP branches as FP16;
- SwiGLU pointwise operations use FP32;
- attention `out_proj` input is scaled by `64`, projected, then restored in FP32;
- MLP `fc2` input is scaled by `256`, projected, then restored in FP32;
- the original H3 FP32 output region remains intact.

`Exact` refers to power-of-two overflow scaling. It does not claim bitwise equality among FP16, BF16, INT8, ConvRot, and FP32 execution.

The node changes model compute precision and overflow boundaries only. It does not alter the sampler, sigma, latent, VAE, frame count, resolution, or attention backend. INT8 QK/PV calculations inside CK, SLA, or Sol are not converted to FP16.

## Nodes

| Node | Purpose |
|---|---|
| `MiniMax H3 Native FP16 Loader - Star7` | Recommended entry point; loads a native H3 diffusion model and applies FP16 policy during model creation |
| `MiniMax H3 FP16 Exact Fix (Legacy) - Star7` | Compatibility entry point for workflows that already contain this historical class ID |

On Ampere, Ada, Blackwell, and other SM80+ architectures, this node follows the same policy as the Enhanced Loader: it explicitly creates H3 with BF16, overrides a global `--fp16-unet` setting for H3 only, and installs no FP16 block wrappers.

## Recommended connection order

```text
MiniMax H3 Native FP16 Loader - Star7
  -> LoRA Loader (optional)
  -> Attention Patch (optional)
  -> MiniMax H3 Activation Chunk - Star7 (optional)
  -> Guider / Scheduler / Sampler
```

This project and [MiniMax H3 Activation Chunk - Star7](https://github.com/star7code/minimax-h3-chunk-star7) are independent: this loader handles FP16 loading and numerical protection; the chunk project handles QKV/RoPE/MLP activation chunking and attention selection.

## Model compatibility

| Model or loading path | Support |
|---|---|
| Native dense BF16 / FP32 H3 safetensors | Supported; creates FP16 operations and installs numerical protection |
| Native `int8_tensorwise` + ConvRot H3 | Supported; retains quantized weights and native dispatch |
| Native H3 with `model.diffusion_model.` prefix | Supported |
| Native H3 with file-level `_quantization_metadata` | Supported |
| `convrot_w4a4`, `asym_w4a8_int8` | Native schedule is preserved; verify the full target workflow in the intended environment |
| GGUF, GPTQ, bitsandbytes, or custom quantized loaders | Not supported by the native MixedPrecisionOps contract |
| Structurally modified H3 classes or non-H3 diffusion models | Not supported |

Standard ComfyUI LoRA patches may temporarily dequantize affected layers under dynamic/low-VRAM loading. The plugin reports quantized models with runtime weight patches but does not silently merge or requantize them.

## Supported hardware

| Architecture | Behavior |
|---|---|
| NVIDIA Turing (RTX 20, T4, Quadro RTX, Titan RTX) | Recommended FP16-loader target |
| NVIDIA Volta (V100, Titan V) | FP16 path supported |
| NVIDIA P100 (SM60) | FP16 path available; validate the target workflow |
| NVIDIA P40 / GTX 10 (SM61) | Fix skipped; keeps ComfyUI's default path |
| NVIDIA Ampere and newer (SM80+) | Always corrected to explicit native BF16 with no FP16 block wrappers |
| AMD ROCm | Experimental; depends on the installed PyTorch and ComfyUI environment |

## Installation

Comfy CLI:

```bash
comfy node install minimax-h3-fp16-exact-star7
```

Manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-fp16-exact-star7.git
```

Restart ComfyUI after installing or updating.

## Example workflow

- [General workflow — English](examples/workflows/MiniMax-H3-FP16-Chunk-Star7-English.json): one graph covers SM75 and SM80+ with automatic safe precision selection.
- [通用工作流（中文）](examples/workflows/MiniMax-H3-FP16-Chunk-Star7.json)

## Scope and diagnostics

The target model class is `comfy.ldm.minimax.model.MiniMaxH3Model`. The plugin uses ComfyUI's native model detection, ModelPatcher, MixedPrecisionOps, quantization, dynamic unloading, and cache reload mechanisms. Do not combine it with another process-wide FP16 overflow patch that wraps the same H3 blocks.

Example quantized-loader log:

```text
[Star7 H3 FP16] Enabled v2.0.7 | mode=loader-quantized | backend=int8_tensorwise+convrot:200 | force-cast=False | weight-patches=0 | blocks=50
```

## Attribution and license

FP32 numerical regions and power-of-two scaling are derived from [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix). Native loading, quantization-aware dispatch, hardware detection, ModelPatcher integration, and packaging are maintained by [Star7](https://github.com/star7code).

MIT. Upstream copyright notices are retained in [LICENSE](LICENSE).
