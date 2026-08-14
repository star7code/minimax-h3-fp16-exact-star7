import logging
from collections import Counter
from types import MethodType

import folder_paths
import torch
import torch.nn.functional as F

import comfy.model_detection
import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.supported_models
import comfy.utils


NODE_VERSION = "2.0.0"
PATCH_FLAG = "star7_minimax_h3_fp16_exact_fix"
PATCH_MODE = "star7_minimax_h3_fp16_mode"
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


def _quantization_summary(diffusion_model):
    formats = Counter()
    for module in diffusion_model.modules():
        quant_format = getattr(module, "quant_format", None)
        layout_type = getattr(module, "layout_type", None)
        if quant_format is None or layout_type is None:
            continue

        weight = getattr(module, "weight", None)
        params = getattr(weight, "_params", None)
        label = quant_format
        if getattr(params, "convrot", False):
            label += "+convrot"
        formats[label] += 1
    return formats


def _format_quantization(formats):
    return ",".join(f"{name}:{count}" for name, count in sorted(formats.items()))


def _supports_fp16_fix():
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"

    if torch.version.hip is not None:
        return True, "ROCm"

    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        return False, f"sm{capability[0]}{capability[1]} supports native BF16"
    if capability == (6, 1):
        return False, "sm61 has very slow FP16 throughput"
    return True, f"sm{capability[0]}{capability[1]}"


def _patch_h3_model(model, loader_native=False):
    import comfy.ldm.minimax.model as minimax_module

    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_module.MiniMaxH3Model):
        raise TypeError("Connected model is not native ComfyUI MiniMax H3")

    transformer_options = patched.model_options.setdefault("transformer_options", {})
    if transformer_options.get(PATCH_FLAG):
        logging.info("[Star7 H3 FP16] Patch is already present; skipping duplicate.")
        return patched

    quant_formats = _quantization_summary(diffusion_model)
    is_quantized = bool(quant_formats)

    patched.set_model_compute_dtype(torch.float16)
    if is_quantized or loader_native:
        # Keep the UUID update from set_model_compute_dtype without forcing
        # MixedPrecisionOps to dequantize its weights.
        patched.force_cast_weights = False

    if getattr(
        minimax_module.MiniMaxH3Model,
        "_star7_h3_global_fp16_patch",
        False,
    ) and getattr(diffusion_model.blocks[0], "_star7_h3_fp16_fix", False):
        mode = "loader-native" if loader_native else "postload"
        transformer_options[PATCH_FLAG] = NODE_VERSION
        transformer_options[PATCH_MODE] = mode
        logging.info("[Star7 H3 FP16] Global overflow fix already active | mode=%s", mode)
        return patched

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
            MethodType(_block_forward(block.forward, minimax_module), block),
        )

    if loader_native:
        mode = "loader-quantized" if is_quantized else "loader-dense"
    else:
        mode = "postload-quantized" if is_quantized else "postload-dense"

    transformer_options[PATCH_FLAG] = NODE_VERSION
    transformer_options[PATCH_MODE] = mode

    weight_patches = len(getattr(patched, "patches", {}))
    backend = _format_quantization(quant_formats) if is_quantized else "dense-fp16"
    logging.info(
        "[Star7 H3 FP16] Enabled v%s | mode=%s | backend=%s | force-cast=%s | weight-patches=%d | blocks=%d",
        NODE_VERSION,
        mode,
        backend,
        bool(patched.force_cast_weights),
        weight_patches,
        len(diffusion_model.blocks),
    )
    if is_quantized and weight_patches:
        logging.warning(
            "[Star7 H3 FP16] Weight patches detected; dynamic low-VRAM LoRA may dequantize affected layers."
        )
    return patched


def _detect_h3_config(state_dict, metadata):
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    detection_state_dict = state_dict
    if prefix:
        stripped = comfy.utils.state_dict_prefix_replace(
            state_dict, {prefix: ""}, filter_keys=True
        )
        if stripped:
            detection_state_dict = stripped
    model_config = comfy.model_detection.model_config_from_unet(
        detection_state_dict, "", metadata=metadata
    )
    if not isinstance(model_config, comfy.supported_models.MiniMaxH3):
        raise ValueError("Selected file is not a native ComfyUI MiniMax H3 diffusion model")
    return model_config


def _load_h3_native_fp16(unet_path, disable_dynamic=False):
    state_dict, metadata = comfy.utils.load_torch_file(
        unet_path, return_metadata=True
    )
    state_dict, metadata = comfy.utils.convert_old_quants(
        state_dict, "", metadata=metadata
    )
    model_config = _detect_h3_config(state_dict, metadata)
    load_device = comfy.model_management.get_torch_device()
    operations = comfy.ops.pick_operations(
        torch.float16,
        torch.float16,
        load_device=load_device,
        model_config=model_config,
    )
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options={
            "dtype": torch.float16,
            "custom_operations": operations,
        },
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError("ComfyUI could not load the selected MiniMax H3 model")

    patched = _patch_h3_model(model, loader_native=True)
    patched.cached_patcher_init = (_load_h3_native_fp16, (unet_path,))
    return patched


class MiniMaxH3FP16LoaderStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = (
        "Recommended MiniMax H3 loader for pre-BF16 GPUs. Creates dense and "
        "MixedPrecisionOps layers with FP16 compute from the start, preserves "
        "INT8/ConvRot dispatch, and installs the exact overflow fix."
    )

    def load_model(self, unet_name):
        unet_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", unet_name
        )
        supported, reason = _supports_fp16_fix()
        if not supported:
            logging.info(
                "[Star7 H3 FP16] Native loader bypassed: %s. Using ComfyUI default loader.",
                reason,
            )
            return (comfy.sd.load_diffusion_model(unet_path),)

        logging.info("[Star7 H3 FP16] Loading at creation-time FP16 | device=%s", reason)
        return (_load_h3_native_fp16(unet_path),)


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
        "Backward-compatible MODEL patch. Automatically preserves MixedPrecisionOps "
        "INT8/ConvRot kernels instead of forcing quantized weights to FP16. The "
        "dedicated MiniMax H3 Native FP16 Loader - Star7 is preferred."
    )

    def patch(self, model, enabled=True):
        if not enabled:
            return (model,)

        supported, reason = _supports_fp16_fix()
        if not supported:
            logging.info("[Star7 H3 FP16] Patch skipped: %s.", reason)
            return (model,)

        try:
            return (_patch_h3_model(model),)
        except (ImportError, TypeError) as exc:
            logging.warning("[Star7 H3 FP16] Model left unchanged: %s.", exc)
            return (model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3FP16LoaderStar7": MiniMaxH3FP16LoaderStar7,
    "MiniMaxH3FP16ExactFixStar7": MiniMaxH3FP16ExactFixStar7,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3FP16LoaderStar7": "MiniMax H3 Native FP16 Loader - Star7",
    "MiniMaxH3FP16ExactFixStar7": "MiniMax H3 FP16 Exact Fix - Star7",
}
