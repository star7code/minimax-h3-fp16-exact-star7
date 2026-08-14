# MiniMax H3 FP16 Exact Fix - Star7

Smart FP16 compatibility nodes for native ComfyUI MiniMax H3 on GPUs without
native BF16 tensor-core support. Version 2 adds a creation-time FP16 loader and
preserves ComfyUI MixedPrecisionOps INT8/ConvRot kernels instead of silently
forcing quantized weights through a dense FP16 fallback.

The overflow method is derived from the MIT-licensed
[Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix).
The Star7 edition adds workflow nodes, quantization-aware dispatch, scoped model
patches, hardware checks, diagnostics, and ComfyUI packaging.

## Recommended node

Use **MiniMax H3 Native FP16 Loader - Star7** instead of the standard UNET
loader on Volta/Turing-class GPUs:

```text
MiniMax H3 Native FP16 Loader - Star7
  -> LoRA loader (optional; read the caveat below)
  -> attention patch (optional)
  -> MiniMax H3 Activation Chunk - Star7 (optional)
  -> Guider / Scheduler / Sampler
```

The loader performs both parts of the fix:

1. It creates dense or MixedPrecisionOps layers with FP16 compute from the
   beginning.
2. It installs FP32 residual math and the exact power-of-two overflow guards.

No `--fp16-unet` startup flag and no separate FP16 patch node are required.

## Existing workflows

The original **MiniMax H3 FP16 Exact Fix - Star7** `MODEL -> MODEL` node remains
registered with the same class ID, so existing workflows do not break.

Version 2 makes it quantization-aware:

| Connected model | Behavior |
|---|---|
| Dense BF16/FP32 weights | Sets FP16 compute and casts dense weights as needed |
| MixedPrecision INT8/ConvRot weights | Sets FP16 activation compute but keeps `force_cast_weights=false`, preserving eligible quantized kernels |
| Ampere or newer NVIDIA GPU | Bypasses the patch and retains native BF16 |
| sm61 NVIDIA GPU | Bypasses the patch because FP16 throughput is very slow |

For new workflows, prefer the dedicated loader because it sets the operation
compute dtype before quantized layers and their output metadata are created.
The post-load node is a compatibility path, not a byte-for-byte equivalent of
the upstream `--fp16-unet` construction path.

## Numerical fix

Unprotected H3 FP16 inference can overflow at several points. These nodes:

- run `condition_proj` with FP32 input;
- keep the residual stream across all 50 DiT blocks in FP32;
- cast normalized attention and MLP branch inputs to FP16;
- run SwiGLU pointwise math in FP32;
- protect attention `out_proj` with a power-of-two scale of 64;
- protect MLP `fc2` with a power-of-two scale of 256;
- leave H3's existing FP32 output islands unchanged.

`Exact` describes the power-of-two overflow transformation. It does not mean
that FP16, INT8, ConvRot, BF16, and FP32 backends produce bit-identical output.

## INT8 / ConvRot behavior

ComfyUI's quantized linear path requires `comfy_force_cast_weights=false`.
Version 1 called `set_model_compute_dtype(FP16)`, which also enabled forced
weight casting and could disable the native INT8/ConvRot route.

Version 2 corrects this:

- the native loader builds MixedPrecisionOps with FP16 compute;
- quantized weights remain quantized;
- the model input and H3 branch activations use FP16;
- `force_cast_weights` remains disabled for quantized models;
- the console reports the detected formats, layer counts, selected mode, force
  cast state, weight patch count, and DiT block count.

The locally inspected `minimax_h3_*_int8_convrot.safetensors` checkpoint contains
200 embedded `comfy_quant` configurations. All 200 decode to
`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`.
Loader construction and dispatch invariants were validated on the RTX 2080 Ti
setup; performance and output still require an end-to-end render after updating.

When combining version 2 with an activation/MLP chunk node, that node must also
preserve weight-only quantized tensors. If an older chunk implementation keeps
`fc1/fc2` as dense resident FP16 weights, disable its `reuse_mlp_weights` option
until the chunk node is updated; otherwise it can undo the loader's ConvRot
preservation inside the MLP path. RoPE-only chunking is unaffected.

## LoRA caveat

Standard ComfyUI LoRA patches may remain as runtime `weight_function` entries
under dynamic/low-VRAM loading. ComfyUI can then dequantize the affected layer
even when this plugin correctly preserves `force_cast_weights=false`.

This plugin logs a warning when LoRA patches are already attached. For maximum
quantized-path retention, use a loader designed to merge or apply the specific
MiniMax H3 LoRA without leaving a dynamic weight function. Requantizing a merged
LoRA is not mathematically identical to applying it to a dense weight, so this
plugin does not do that silently.

## Installation and update

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-fp16-exact-star7.git
```

Existing installation:

```bash
cd ComfyUI/custom_nodes/minimax-h3-fp16-exact-star7
git pull
```

Restart ComfyUI after installation or update.

## Example diagnostic

```text
[Star7 H3 FP16] Enabled v2.0.0 | mode=loader-quantized | backend=int8_tensorwise+convrot:200 | force-cast=False | weight-patches=0 | blocks=50
```

Modes:

- `loader-quantized`: recommended creation-time FP16 with quantized weights;
- `loader-dense`: recommended creation-time FP16 with dense weights;
- `postload-quantized`: compatible post-load path preserving quantized dispatch;
- `postload-dense`: compatible post-load dense FP16 path.

## Supported hardware

| GPU architecture | Recommendation |
|---|---|
| NVIDIA Turing (RTX 20 series, T4, Quadro RTX, Titan RTX) | Recommended |
| NVIDIA Volta (V100, Titan V) | Recommended |
| NVIDIA P100 (sm60) | Expected to help; not locally validated |
| NVIDIA P40 / GTX 10 series (sm61) | Automatically bypassed |
| NVIDIA Ampere or newer | Automatically bypassed; use native BF16 |
| AMD ROCm | Experimental; upstream reports a working configuration |

Local development validation uses an RTX 2080 Ti 22 GB (sm75), Windows,
Python 3.13, PyTorch CUDA 13, and native ComfyUI MiniMax H3.

## Model format support

| Model / loading path | Status | Notes |
|---|---|---|
| Native ComfyUI MiniMax H3 dense BF16 or FP32 safetensors | Supported | Loader creates dense FP16 weights/ops and installs overflow protection |
| Native ComfyUI MiniMax H3 `int8_tensorwise` + ConvRot safetensors | Targeted and structurally validated | Recommended path for the current RTX 2080 Ti workflow; all 200 embedded quantization configs were verified, with an end-to-end render still pending |
| Native ComfyUI MiniMax H3 MixedPrecisionOps `convrot_w4a4` or `asym_w4a8_int8` | Dispatch preserved, not locally rendered | The loader retains quantized weights and FP16 operation metadata, but these formats still require end-to-end validation |
| Standard ComfyUI LoRA on a quantized model | Conditional | Correct FP16 policy remains active, but dynamic/low-VRAM weight functions may dequantize patched layers |
| MiniMax H3 LoRA loader that explicitly preserves or requantizes its quantized layout | Compatible in principle | Verify its own documentation and the runtime diagnostic |
| GGUF, GPTQ, bitsandbytes, or another custom quantized loader | Unsupported | These do not use the native MixedPrecisionOps contract handled here |
| Non-native or forked H3 model class | Unsupported | The patch intentionally requires native `MiniMaxH3Model` |
| Non-H3 diffusion model | Unsupported | Use that model family's own dtype policy |

The plugin does not make every operation INT8. Residual accumulation, overflow
islands, normalization/modulation boundaries, and other numerically sensitive
work intentionally remain FP32 or FP16 according to the fix.

## Compatibility

- Targets `comfy.ldm.minimax.model.MiniMaxH3Model`.
- Preserves the original post-load node class ID.
- Uses ComfyUI's native loader, model patcher, MixedPrecisionOps, quantization,
  offload, and cached-reload mechanisms.
- Internal ComfyUI loader or MiniMax H3 refactors may require an update.
- Do not combine this with another process-wide implementation of the same
  overflow fix unless deliberately testing compatibility.

## Attribution

The overflow analysis, FP32 islands, and power-of-two scaling method originate
from [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix).
The workflow integration, native loader, quantization-aware dispatch, and Star7
packaging are maintained by [Star7](https://github.com/star7code).

## License

MIT. The upstream copyright notice is retained in [LICENSE](LICENSE).
