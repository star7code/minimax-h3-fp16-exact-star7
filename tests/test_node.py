import importlib.util
import copy
import sys
from pathlib import Path

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


def test_registration():
    module = load_nodes()
    assert "MiniMaxH3FP16ExactFixStar7" in module.NODE_CLASS_MAPPINGS
    assert module.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3FP16ExactFixStar7"].endswith(
        "Star7"
    )


def test_scale_constants_are_powers_of_two():
    module = load_nodes()
    for value in (module.K_OUT_PROJ, module.K_FC2):
        integer = int(value)
        assert integer > 0 and integer & (integer - 1) == 0


def test_model_patch_is_scoped_and_complete():
    module = load_nodes()
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

    class FakePatcher:
        def __init__(self):
            self.diffusion = diffusion
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}
            self.compute_dtype = None

        def clone(self):
            return copy.copy(self)

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return self.diffusion

        def set_model_compute_dtype(self, dtype):
            self.compute_dtype = dtype

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

    patcher = FakePatcher()
    node = module.MiniMaxH3FP16ExactFixStar7()
    patched = node.patch(patcher, enabled=True)[0]
    assert patched is not patcher
    assert patched.compute_dtype is torch.float16
    assert patched.model_options["transformer_options"][module.PATCH_FLAG] == module.NODE_VERSION
    assert len(patched.object_patches) == 1 + 3 * len(diffusion.blocks)


if __name__ == "__main__":
    test_registration()
    test_scale_constants_are_powers_of_two()
    test_model_patch_is_scoped_and_complete()
    print("MiniMax H3 FP16 Exact Fix - Star7 tests passed")
