import logging
from types import MethodType

import torch
import torch.nn.functional as F


NODE_VERSION = "1.0.0"
PATCH_FLAG = "star7_minimax_h3_fp16_exact_fix"
K_OUT_PROJ = 64.0
K_FC2 = 256.0


def _condition_proj_forward(original_forward):
    def forward(self, tensor):
        return original_forward(tensor.to(torch.float32))

    return forward


def _out_proj_forward(original_forward):
    def forward(self, tensor):
        scaled = (tensor / K_OUT_PROJ).to(torch.float16)
        return original_forward(scaled).to(torch.float32).mul_(K_OUT_PROJ)

    return forward


def _mlp_forward(original_forward):
    def forward(self, tensor):
        if tensor.dtype != torch.float16:
            return original_forward(tensor)

        projected = self.fc1(tensor)
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate.to(torch.float32)).mul_(up.to(torch.float32))
        scaled = (activated / K_FC2).to(torch.float16)
        return self.fc2(scaled).to(torch.float32).mul_(K_FC2)

    return forward


def _block_forward(original_forward, minimax_module):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)

        h = minimax_module._mod_scale_shift(
            self.norm1(x), shift_msa, scale_msa, mod_segments
        ).to(torch.float16)
        attention = self.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        x = minimax_module._mod_gate(
            x, gate_msa, attention.to(torch.float32), mod_segments
        )

        h = minimax_module._mod_scale_shift(
            self.norm2(x), shift_mlp, scale_mlp, mod_segments
        ).to(torch.float16)
        mlp = self.mlp(h)
        return minimax_module._mod_gate(
            x, gate_mlp, mlp.to(torch.float32), mod_segments
        )

    return forward


class MiniMaxH3FP16ExactFixStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Runs MiniMax H3 with FP16 matmuls while retaining FP32 residual math and "
        "exact power-of-two overflow protection. Intended for pre-BF16 GPUs such "
        "as RTX 20-series and V100. Apply it directly after the H3 UNET loader; "
        "no --fp16-unet startup flag is required."
    )

    def patch(self, model, enabled=True):
        if not enabled:
            return (model,)

        if not torch.cuda.is_available():
            logging.warning(
                "[Star7 H3 FP16 Exact] CUDA is unavailable; model left unchanged."
            )
            return (model,)

        capability = torch.cuda.get_device_capability()
        is_rocm = torch.version.hip is not None
        if not is_rocm and capability[0] >= 8:
            logging.info(
                "[Star7 H3 FP16 Exact] sm%d%d supports native BF16; patch skipped.",
                capability[0], capability[1],
            )
            return (model,)
        if not is_rocm and capability == (6, 1):
            logging.warning(
                "[Star7 H3 FP16 Exact] sm61 has very slow FP16 throughput; patch skipped."
            )
            return (model,)

        try:
            import comfy.ldm.minimax.model as minimax_module
        except ImportError as exc:
            logging.warning(
                "[Star7 H3 FP16 Exact] Native MiniMax H3 is unavailable (%s); "
                "model left unchanged.",
                exc,
            )
            return (model,)

        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, minimax_module.MiniMaxH3Model):
            logging.warning(
                "[Star7 H3 FP16 Exact] Connected model is not MiniMax H3; "
                "model left unchanged."
            )
            return (model,)

        transformer_options = patched.model_options.setdefault("transformer_options", {})
        if transformer_options.get(PATCH_FLAG):
            logging.info(
                "[Star7 H3 FP16 Exact] Patch is already present; skipping duplicate."
            )
            return (patched,)

        patched.set_model_compute_dtype(torch.float16)

        # The process-wide class patch applies during model construction when
        # ComfyUI is launched with --fp16-unet. Avoid wrapping those methods a
        # second time; this node remains useful as a per-workflow dtype switch.
        if getattr(
            minimax_module.MiniMaxH3Model,
            "_star7_h3_global_fp16_patch",
            False,
        ) and getattr(diffusion_model.blocks[0], "_star7_h3_fp16_fix", False):
            transformer_options[PATCH_FLAG] = NODE_VERSION
            logging.info(
                "[Star7 H3 FP16 Exact] Global fix already active; "
                "MODEL branch set to FP16 without duplicate wrappers."
            )
            return (patched,)

        condition_proj = diffusion_model.condition_proj
        patched.add_object_patch(
            "diffusion_model.condition_proj.forward",
            MethodType(_condition_proj_forward(condition_proj.forward), condition_proj),
        )

        for index, block in enumerate(diffusion_model.blocks):
            out_proj = block.attn.out_proj
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.out_proj.forward",
                MethodType(_out_proj_forward(out_proj.forward), out_proj),
            )
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.mlp.forward",
                MethodType(_mlp_forward(block.mlp.forward), block.mlp),
            )
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.forward",
                MethodType(
                    _block_forward(block.forward, minimax_module),
                    block,
                ),
            )

        transformer_options[PATCH_FLAG] = NODE_VERSION
        logging.info(
            "[Star7 H3 FP16 Exact] Enabled v%s on sm%d%d | "
            "FP16 attention/MLP matmuls, FP32 residual stream, %d DiT blocks patched.",
            NODE_VERSION,
            capability[0],
            capability[1],
            len(diffusion_model.blocks),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3FP16ExactFixStar7": MiniMaxH3FP16ExactFixStar7,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FP16ExactFixStar7": "MiniMax H3 FP16 Exact Fix - Star7",
}
