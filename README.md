# MiniMax H3 FP16 Exact Fix - Star7

A workflow-scoped ComfyUI node that lets native MiniMax H3 use FP16 matrix
multiplication on GPUs without native BF16 tensor-core support while preserving
the numerically sensitive parts of the model in FP32.

The core numerical method is derived from the MIT-licensed
[Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix).
This Star7 edition adds a `MODEL -> MODEL` workflow node, branch-local ComfyUI
object patches, compatibility checks, and integration metadata.

## What it fixes

Forcing MiniMax H3 to FP16 without overflow protection can produce black or
invalid output. This node keeps the H3 residual stream in FP32 and protects the
known overflow points while retaining FP16 attention and MLP matrix
multiplications:

- `condition_proj` receives FP32 input.
- The residual stream across all 50 DiT blocks remains FP32.
- Normalized attention and MLP branch inputs use FP16.
- Attention `out_proj` uses an exact power-of-two scale of 64.
- MLP `fc2` uses an exact power-of-two scale of 256.
- Existing H3 FP32 output islands remain unchanged.

It does not quantize weights, change the sampler, alter the latent, split the
video, or reduce generation settings.

## Supported hardware

| GPU architecture | Recommendation |
|---|---|
| NVIDIA Turing (RTX 20 series, T4, Quadro RTX, Titan RTX) | Recommended |
| NVIDIA Volta (V100, Titan V) | Recommended |
| NVIDIA P100 (sm60) | Expected to help; not locally validated |
| NVIDIA P40 / GTX 10 series (sm61) | Automatically skipped because FP16 is slow |
| NVIDIA Ampere or newer (RTX 30/40/50 series, A-series, etc.) | Automatically skipped; use native BF16 |
| AMD ROCm | Experimental; upstream has a reported working configuration |

Local validation was performed on an RTX 2080 Ti 22 GB (sm75), Windows,
Python 3.13, PyTorch with CUDA 13, and native ComfyUI MiniMax H3.

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-fp16-exact-star7.git
```

Restart ComfyUI. No additional Python package and no `--fp16-unet` startup flag
is required.

## Workflow placement

Place the node directly after the MiniMax H3 UNET loader and before LoRA nodes:

```text
MiniMax H3 UNET Loader
  -> MiniMax H3 FP16 Exact Fix - Star7
  -> LoRA Loader(s)
  -> optional attention / model patches
  -> Guider / Scheduler / Sampler
```

Yes: `UNET -> FP16 Exact Fix - Star7 -> LoRA` is the recommended connection.
Only the connected `MODEL` branch is patched. Other models and branches are
left unchanged.

If you also use **MiniMax H3 Activation Chunk - Star7**, keep the FP16 node
before the activation-chunk node so the latter can detect and preserve the FP16
exact-math path.

## Node input

- `model`: the native MiniMax H3 `MODEL` output.
- `enabled`: enables or bypasses the patch. Default: `true`.

The scale constants are intentionally not exposed as user settings because they
are numerical safety constants, not performance controls.

## Verification

After queueing a prompt on an RTX 20-series or V100 system, the console should
contain a line similar to:

```text
[Star7 H3 FP16 Exact] Enabled v1.0.0 on sm75 | FP16 attention/MLP matmuls, FP32 residual stream, 50 DiT blocks patched.
```

On Ampere or newer GPUs the node reports that native BF16 is available and
leaves the model unchanged.

## Compatibility notes

- Targets ComfyUI's native `comfy.ldm.minimax.model.MiniMaxH3Model`.
- Internal ComfyUI refactors may require an update to this node.
- Do not install another process-wide implementation of the same FP16 fix at
  the same time unless you are deliberately testing compatibility.
- Performance gains depend on model weight format, attention backend, offload
  behavior, resolution, sequence length, and GPU architecture.

## Attribution

The overflow analysis, FP32 islands, and exact scaling method originate from
[Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix).
The workflow-scoped integration and Star7 packaging are maintained by
[Star7](https://github.com/star7code).

## License

MIT. The upstream copyright notice is retained in [LICENSE](LICENSE).
