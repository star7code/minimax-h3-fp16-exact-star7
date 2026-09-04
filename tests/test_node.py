import importlib.util
import copy
import gc
import json
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = ROOT.parents[1]
if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))


def load_nodes():
    spec = importlib.util.spec_from_file_location("star7_h3_nodes", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_from_source(filename, source, name, globals_dict=None):
    namespace = {} if globals_dict is None else dict(globals_dict)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def test_turing_process_patch_is_neutralized_before_star7_install():
    module = load_nodes()
    from comfy.ldm.minimax import model as minimax_module

    plugin_name = "comfyui_minimax_h3_turing_test_double"
    plugin = types.ModuleType(plugin_name)
    plugin.__file__ = "C:/custom_nodes/comfyui-minimax-h3-turing/nodes.py"
    owners = (
        (minimax_module.MiniMaxH3Model, "__init__", "_orig_model_init"),
        (minimax_module.MLP, "forward", "_orig_mlp_forward"),
        (minimax_module.DiTBlock, "forward", "_orig_block_forward"),
        (minimax_module.Attention, "forward", "_orig_attention_forward"),
    )
    saved_attributes = [
        (owner, attribute, getattr(owner, attribute))
        for owner, attribute, _saved_name in owners
    ]
    marker = getattr(minimax_module, "_star7_turing_plugin_neutralized", None)
    marker_existed = hasattr(minimax_module, "_star7_turing_plugin_neutralized")
    try:
        if marker_existed:
            delattr(minimax_module, "_star7_turing_plugin_neutralized")
        for index, (owner, attribute, saved_name) in enumerate(owners):
            original = _function_from_source(
                f"star7_original_{index}.py",
                f"def original_{index}(*args, **kwargs): return {index}\n",
                f"original_{index}",
            )
            conflicting = _function_from_source(
                f"C:/custom_nodes/comfyui-minimax-h3-turing/patch_{index}.py",
                f"def conflicting_{index}(*args, **kwargs): return -{index + 1}\n",
                f"conflicting_{index}",
            )
            setattr(plugin, saved_name, original)
            setattr(owner, attribute, conflicting)
        sys.modules[plugin_name] = plugin

        assert module._neutralize_process_wide_h3_conflicts() is True
        for owner, attribute, saved_name in owners:
            assert getattr(owner, attribute) is getattr(plugin, saved_name)
    finally:
        sys.modules.pop(plugin_name, None)
        for owner, attribute, original in saved_attributes:
            setattr(owner, attribute, original)
        if marker_existed:
            minimax_module._star7_turing_plugin_neutralized = marker
        elif hasattr(minimax_module, "_star7_turing_plugin_neutralized"):
            delattr(minimax_module, "_star7_turing_plugin_neutralized")


def test_turing_instance_forward_wrapper_is_unwrapped():
    module = load_nodes()
    wrapped = torch.nn.Identity()
    original = wrapped.forward
    conflicting = _function_from_source(
        "C:/custom_nodes/comfyui-minimax-h3-turing/instance.py",
        "def conflicting(value, _original=original): return _original(value)\n",
        "conflicting",
        {"original": original},
    )
    wrapped.forward = conflicting
    diffusion_model = torch.nn.Module()
    diffusion_model.child = wrapped
    diffusion_model.blocks = [SimpleNamespace(_h3_fp16_fix=True)]

    module._strip_conflicting_instance_forwards(diffusion_model)

    assert wrapped.forward is original
    assert diffusion_model.blocks[0]._h3_fp16_fix is False


def test_registration():
    module = load_nodes()
    assert "MiniMaxH3FP16LoaderStar7" in module.NODE_CLASS_MAPPINGS
    assert "MiniMaxH3FP16ExactFixStar7" in module.NODE_CLASS_MAPPINGS
    assert module.MiniMaxH3FP16ExactFixStar7.DEPRECATED is True
    for display_name in module.NODE_DISPLAY_NAME_MAPPINGS.values():
        assert display_name.endswith("Star7")


def test_scale_constants_are_powers_of_two():
    module = load_nodes()
    for value in (module.K_OUT_PROJ, module.K_FC2):
        integer = int(value)
        assert integer > 0 and integer & (integer - 1) == 0


def test_sm80_loader_corrects_h3_to_bf16():
    module = load_nodes()
    loaded = object()
    node = module.MiniMaxH3FP16LoaderStar7()
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
        mock.patch.object(module.folder_paths, "get_full_path_or_raise", return_value="h3.safetensors"),
        mock.patch.object(module.comfy.sd, "load_diffusion_model", return_value=loaded) as native_load,
        mock.patch.object(module, "_load_h3_native_fp16") as fp16_load,
    ):
        assert node.load_model("h3.safetensors") == (loaded,)
    native_load.assert_called_once_with(
        "h3.safetensors", model_options={"dtype": torch.bfloat16}
    )
    fp16_load.assert_not_called()


def test_sm80_loader_overrides_launcher_fp16_with_bf16():
    module = load_nodes()
    loaded = object()
    node = module.MiniMaxH3FP16LoaderStar7()
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
        mock.patch.object(module.comfy.model_management.args, "fp16_unet", True),
        mock.patch.object(module.folder_paths, "get_full_path_or_raise", return_value="h3.safetensors"),
        mock.patch.object(module.comfy.sd, "load_diffusion_model", return_value=loaded) as native_load,
        mock.patch.object(module, "_load_h3_native_fp16") as fp16_load,
    ):
        assert node.load_model("h3.safetensors") == (loaded,)
    native_load.assert_called_once_with(
        "h3.safetensors", model_options={"dtype": torch.bfloat16}
    )
    fp16_load.assert_not_called()


def make_h3_patcher(module, quantized=False):
    import comfy.ldm.minimax.model as minimax

    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = torch.nn.Linear(2, 2, bias=False)

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(2, 4, bias=False)
            self.fc2 = torch.nn.Linear(2, 2, bias=False)

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = TinyAttention()
            self.mlp = TinyMLP()

        def forward(self, *args, **kwargs):
            return args[0]

    diffusion = minimax.MiniMaxH3Model.__new__(minimax.MiniMaxH3Model)
    torch.nn.Module.__init__(diffusion)
    diffusion.condition_proj = torch.nn.Linear(2, 2, bias=False)
    diffusion.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
    if quantized:
        linear = diffusion.blocks[0].mlp.fc1
        linear.quant_format = "int8_tensorwise"
        linear.layout_type = "TensorWiseINT8Layout"
        linear.weight._params = SimpleNamespace(convrot=True)

    class FakePatcher:
        def __init__(self):
            self.diffusion = diffusion
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}
            self.compute_dtype = None
            self.force_cast_weights = False
            self.patches = {}

        def clone(self):
            cloned = copy.copy(self)
            cloned.model_options = copy.deepcopy(self.model_options)
            cloned.object_patches = self.object_patches.copy()
            return cloned

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return self.diffusion

        def set_model_compute_dtype(self, dtype):
            self.compute_dtype = dtype
            self.force_cast_weights = dtype is not None
            self.add_object_patch("manual_cast_dtype", dtype)

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            wrappers = self.model_options["transformer_options"].setdefault("wrappers", {})
            wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    return FakePatcher(), diffusion


def apply_node(module, patcher):
    node = module.MiniMaxH3FP16ExactFixStar7()
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)),
    ):
        return node.patch(patcher, enabled=True)[0]


def test_dense_model_patch_is_scoped_and_complete():
    module = load_nodes()
    patcher, diffusion = make_h3_patcher(module)
    patched = apply_node(module, patcher)
    assert patched is not patcher
    assert patched.compute_dtype is torch.float16
    assert patched.force_cast_weights is True
    assert patched.model_options["transformer_options"][module.PATCH_FLAG] == module.NODE_VERSION
    assert patched.model_options["transformer_options"][module.PATCH_MODE] == "postload-dense"
    assert len(patched.object_patches) == 2 + 3 * len(diffusion.blocks)
    wrappers = patched.model_options["transformer_options"]["wrappers"][
        module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    ]
    assert list(wrappers) == [module.TE_BOUNDARY_WRAPPER_KEY]


def test_commercial_te_boundary_promotes_only_the_context_carrier():
    module = load_nodes()
    context = torch.ones(1, 2, 3, dtype=torch.float16)
    x = torch.ones(1, dtype=torch.float16)
    runtime = object()
    seen = {}

    def executor(*args, **kwargs):
        seen["x"] = args[0]
        seen["context"] = args[2]
        seen["options"] = args[3]
        return "ok"

    options = {module.TE_RUNTIME_KEY: runtime}
    assert module._commercial_te_boundary_wrapper(executor, x, None, context, options) == "ok"
    assert seen["x"] is x
    assert seen["context"].dtype is torch.float32
    assert seen["context"].device == context.device
    assert seen["options"] is options


def test_boundary_wrapper_leaves_noncommercial_and_oss_paths_unchanged():
    module = load_nodes()
    context = torch.ones(1, 2, 3, dtype=torch.float16)
    seen = []

    def executor(*args, **kwargs):
        seen.append(args[2])
        return "ok"

    for options in ({}, {"te_speed_minimax_h3_oss_runtime": object()}, None):
        assert module._commercial_te_boundary_wrapper(executor, None, None, context, options) == "ok"
        assert seen[-1] is context


def test_fp32_te_carrier_keeps_block_compute_inputs_fp16():
    module = load_nodes()
    seen = {}

    class Identity(torch.nn.Module):
        def forward(self, value):
            return value

    class Block:
        norm1 = Identity()
        norm2 = Identity()

        def adaln_proj(self, _t_emb):
            return (torch.zeros(1),) * 6

        def attn(self, value, **_kwargs):
            seen["attention"] = value.dtype
            return value

        def mlp(self, value):
            seen["mlp"] = value.dtype
            return value

    fake_minimax = SimpleNamespace(
        _mod_scale_shift=lambda value, *_args: value,
        _mod_gate=lambda residual, _gate, update, _segments: residual + update,
    )
    protected_forward = module._block_forward(lambda *_args, **_kwargs: None, fake_minimax)
    result = protected_forward(
        Block(),
        torch.ones(1, 2, dtype=torch.float32),
        torch.zeros(1),
        [],
        None,
        {module.TE_RUNTIME_KEY: object()},
    )

    assert result.dtype is torch.float32
    assert seen == {"attention": torch.float16, "mlp": torch.float16}


def test_quantized_model_preserves_native_dispatch():
    module = load_nodes()
    patcher, _diffusion = make_h3_patcher(module, quantized=True)
    patched = apply_node(module, patcher)
    assert patched.compute_dtype is torch.float16
    assert patched.force_cast_weights is False
    assert patched.object_patches["manual_cast_dtype"] is torch.float16
    assert patched.model_options["transformer_options"][module.PATCH_MODE] == "postload-quantized"


def test_model_wrappers_are_weakly_bound_and_idempotent():
    module = load_nodes()
    patcher, diffusion = make_h3_patcher(module)
    patched = apply_node(module, patcher)

    wrapper = patched.object_patches["diffusion_model.blocks.0.forward"]
    assert isinstance(wrapper.__self__, weakref.ProxyTypes)
    assert all(
        cell.cell_contents is not diffusion
        for cell in (wrapper.__func__.__closure__ or ())
    )

    reapplied = apply_node(module, patched)
    assert len(reapplied.object_patches) == len(patched.object_patches)
    assert reapplied.model_options["transformer_options"][module.PATCH_FLAG] == module.NODE_VERSION

    diffusion_ref = weakref.ref(diffusion)
    del reapplied, patched, patcher, diffusion
    gc.collect()
    assert diffusion_ref() is None


def test_quantization_summary_reports_convrot():
    module = load_nodes()
    _patcher, diffusion = make_h3_patcher(module, quantized=True)
    assert module._quantization_summary(diffusion) == {
        "int8_tensorwise+convrot": 1,
    }


def test_native_loader_builds_fp16_operations_before_model_creation():
    module = load_nodes()
    state_dict = {"marker": object()}
    metadata = {"version": 1}
    model_config = SimpleNamespace(quant_config={"layer": {"format": "int8_tensorwise"}})
    operations = object()
    base_model = SimpleNamespace()
    patched_model = SimpleNamespace()

    with (
        mock.patch.object(
            module.comfy.utils,
            "load_torch_file",
            return_value=(state_dict, metadata),
        ),
        mock.patch.object(
            module.comfy.utils,
            "convert_old_quants",
            return_value=(state_dict, metadata),
        ),
        mock.patch.object(module, "_detect_h3_config", return_value=model_config),
        mock.patch.object(
            module.comfy.model_management,
            "get_torch_device",
            return_value=torch.device("cuda"),
        ),
        mock.patch.object(
            module.comfy.ops,
            "pick_operations",
            return_value=operations,
        ) as pick_operations,
        mock.patch.object(
            module.comfy.sd,
            "load_diffusion_model_state_dict",
            return_value=base_model,
        ) as load_state_dict,
        mock.patch.object(
            module,
            "_patch_h3_model",
            return_value=patched_model,
        ) as patch_h3,
    ):
        result = module._load_h3_native_fp16(
            "model.safetensors", disable_dynamic=True
        )

    assert result is patched_model
    assert result.cached_patcher_init == (
        module._load_h3_native_fp16,
        ("model.safetensors",),
    )
    pick_operations.assert_called_once_with(
        torch.float16,
        torch.float16,
        load_device=torch.device("cuda"),
        model_config=model_config,
    )
    options = load_state_dict.call_args.kwargs["model_options"]
    assert options == {
        "dtype": torch.float16,
        "custom_operations": operations,
    }
    assert load_state_dict.call_args.kwargs["disable_dynamic"] is True
    patch_h3.assert_called_once_with(base_model, loader_native=True)


def test_model_detection_strips_diffusion_prefix():
    module = load_nodes()
    state_dict = {"model.blocks.0.weight": object()}
    stripped = {"blocks.0.weight": state_dict["model.blocks.0.weight"]}
    config = object.__new__(module.comfy.supported_models.MiniMaxH3)

    with (
        mock.patch.object(
            module.comfy.model_detection,
            "unet_prefix_from_state_dict",
            return_value="model.",
        ),
        mock.patch.object(
            module.comfy.utils,
            "state_dict_prefix_replace",
            return_value=stripped,
        ) as replace_prefix,
        mock.patch.object(
            module.comfy.model_detection,
            "model_config_from_unet",
            return_value=config,
        ) as detect_config,
    ):
        assert module._detect_h3_config(state_dict, {}) is config

    replace_prefix.assert_called_once_with(
        state_dict, {"model.": ""}, filter_keys=True
    )
    detect_config.assert_called_once_with(stripped, "", metadata={})


def test_normalize_keeps_embedded_quantization_configs():
    module = load_nodes()
    quant = torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
    state_dict = {
        "video_patch_proj.weight": object(),
        "blocks.0.attn.qkv_proj.comfy_quant": quant,
    }

    normalized, metadata = module._normalize_h3_state_dict(state_dict, {})

    assert normalized == state_dict
    assert metadata == {}
    assert module.comfy.utils.detect_layer_quantization(normalized, "") == {
        "mixed_ops": True,
    }


def test_normalize_restores_unprefixed_legacy_quants_after_prefix_removal():
    module = load_nodes()
    prefix = "model.diffusion_model."
    state_dict = {
        f"{prefix}video_patch_proj.weight": object(),
        f"{prefix}audio_patch_proj.weight": object(),
        f"{prefix}blocks.0.attn.qkv_proj.weight": object(),
        f"{prefix}blocks.0.attn.q_norm.weight": object(),
        f"{prefix}blocks.0.attn.k_norm.weight": object(),
        f"{prefix}blocks.0.mlp.fc1.weight": object(),
    }
    layer_config = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }
    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": {
                "blocks.0.attn.qkv_proj": layer_config,
            }}
        )
    }

    normalized, normalized_metadata = module._normalize_h3_state_dict(
        state_dict, metadata
    )

    assert "video_patch_proj.weight" in normalized
    assert "audio_patch_proj.weight" in normalized
    assert "blocks.0.attn.qkv_proj.comfy_quant" in normalized
    assert not any(key.startswith(prefix) for key in normalized)
    restored = bytes(
        normalized["blocks.0.attn.qkv_proj.comfy_quant"].tolist()
    ).decode("utf-8")
    assert json.loads(restored) == layer_config
    assert normalized_metadata == metadata
    assert module.comfy.utils.detect_layer_quantization(normalized, "") == {
        "mixed_ops": True,
    }


if __name__ == "__main__":
    test_turing_process_patch_is_neutralized_before_star7_install()
    test_turing_instance_forward_wrapper_is_unwrapped()
    test_registration()
    test_scale_constants_are_powers_of_two()
    test_sm80_loader_corrects_h3_to_bf16()
    test_sm80_loader_overrides_launcher_fp16_with_bf16()
    test_dense_model_patch_is_scoped_and_complete()
    test_commercial_te_boundary_promotes_only_the_context_carrier()
    test_boundary_wrapper_leaves_noncommercial_and_oss_paths_unchanged()
    test_fp32_te_carrier_keeps_block_compute_inputs_fp16()
    test_quantized_model_preserves_native_dispatch()
    test_quantization_summary_reports_convrot()
    test_native_loader_builds_fp16_operations_before_model_creation()
    test_model_detection_strips_diffusion_prefix()
    test_normalize_keeps_embedded_quantization_configs()
    test_normalize_restores_unprefixed_legacy_quants_after_prefix_removal()
    print("MiniMax H3 FP16 Exact Fix - Star7 tests passed")
