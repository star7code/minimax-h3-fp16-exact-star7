# MiniMax H3 FP16 Exact Fix - Star7

[中文说明](#中文说明) · [示例工作流](examples/workflows)

Native FP16 model loading and scoped numerical protection for ComfyUI MiniMax H3 on GPUs without native BF16 Tensor Core acceleration. Quantized checkpoints retain eligible INT8/ConvRot kernels instead of being expanded into dense FP16 weights.

The overflow-protection method is derived from the MIT-licensed [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix). This package provides a native MiniMax H3 loader, quantization-aware dispatch, architecture checks, scoped ModelPatcher integration, diagnostics, and ComfyUI workflow support.

## 中文说明

本项目为 MiniMax H3 提供原生 FP16 模型载入和数值保护，主要面向 RTX 20 系（Turing）及其他缺少原生 BF16 Tensor Core 加速的显卡。

推荐使用 `MiniMax H3 Native FP16 Loader - Star7` 载入扩散模型。它在模型创建阶段确定 FP16 计算类型，同时保留原生 MixedPrecisionOps 的 INT8/ConvRot 权重布局，并在推理前安装 MiniMax H3 的 FP16 溢出保护。

对于量化模型，保留 INT8/ConvRot 路径可避免符合条件的线性层被展开为常驻稠密 FP16 权重。实际显存占用仍由模型规模、LoRA、ComfyUI 动态卸载、参考条件和其他节点共同决定。

## 主要功能

| 功能 | 说明 |
|---|---|
| 原生 FP16 载入 | 在 MiniMax H3 模型创建阶段配置 FP16 计算，无需 `--fp16-unet` |
| FP16 数值保护 | 对残差、SwiGLU、attention `out_proj` 和 MLP `fc2` 设置对应的 FP32 计算区与溢出保护 |
| 量化路径保留 | 对原生 MixedPrecisionOps 模型保持 `force_cast_weights=false`，保留可用的 INT8/ConvRot 内核 |
| 模型格式识别 | 支持原生 H3 权重、`model.diffusion_model.` 外层前缀及文件级 `_quantization_metadata` |
| 架构检测 | 在 SM80+ 上使用 ComfyUI 默认 BF16 路径；SM61 因 FP16 吞吐较低而跳过修复 |
| 生命周期安全 | 修复仅安装在克隆后的 ModelPatcher 上，模型方法使用弱绑定，避免旧模型被补丁闭包长期持有 |
| 运行诊断 | 报告载入模式、量化格式、强制权重转换状态、权重补丁数量和 DiT block 数量 |

## 工作原理

MiniMax H3 在 FP16 下需要保护若干数值敏感位置：

- `condition_proj` 接收 FP32 输入；
- 50 个 DiT block 的残差流保持 FP32；
- attention 和 MLP 分支的归一化输出以 FP16 进入对应计算路径；
- SwiGLU 点运算使用 FP32；
- attention `out_proj` 输入先按 `64` 缩放，投影后以 FP32 恢复；
- MLP `fc2` 输入先按 `256` 缩放，投影后以 FP32 恢复；
- H3 原有的 FP32 输出区保持不变。

`Exact` 指以 2 的整数次幂进行溢出保护；这类缩放不会额外引入普通比例换算的舍入形式。它不表示 FP16、BF16、INT8、ConvRot 与 FP32 会产生逐位一致的输出。

该修复只处理模型计算精度和溢出边界，不修改采样器、sigma、latent、VAE、帧数、分辨率或注意力后端。CK、SLA、Sol 等注意力中的 INT8 QK/PV 计算不会被转换为 FP16。

## 包含的节点

| 节点 | 用途 |
|---|---|
| `MiniMax H3 Native FP16 Loader - Star7` | 推荐节点；从 `diffusion_models` 载入原生 MiniMax H3，并在模型创建阶段应用 FP16 策略和数值保护 |
| `MiniMax H3 FP16 Exact Fix (Legacy) - Star7` | 已加载 `MODEL` 的兼容入口；仅用于仍包含该 class ID 的工作流 |

在 Ampere、Ada、Blackwell 等 SM80+ 架构上使用载入节点时，会直接调用 ComfyUI 默认模型载入路径，不安装 FP16 修复，因此不会增加 Transformer block 包装或额外计算。

## 推荐连接方式

```text
MiniMax H3 Native FP16 Loader - Star7
  -> LoRA Loader（可选）
  -> Attention Patch（可选）
  -> MiniMax H3 Activation Chunk - Star7（可选）
  -> Guider / Scheduler / Sampler
```

FP16 Loader 与 [MiniMax H3 Activation Chunk - Star7](https://github.com/star7code/minimax-h3-chunk-star7) 可以分别独立使用：前者负责 FP16 载入与数值保护，后者负责 QKV/RoPE/MLP 激活分块和注意力选择。

仓库提供 RTX 20 系示例工作流：[MiniMax-H3-FP16-Chunk-RTX20-Star7.json](examples/workflows/MiniMax-H3-FP16-Chunk-RTX20-Star7.json)。示例工作流还使用以下项目：

- [MiniMax H3 Activation Chunk - Star7](https://github.com/star7code/minimax-h3-chunk-star7)
- [MiniMax H3 Audio Conditioning T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
- ComfyUI-VideoHelperSuite

这些项目是示例工作流的配套节点，不是 FP16 Loader 本身的 Python 依赖。

## INT8 / ConvRot 兼容

ComfyUI 原生量化线性层通过 MixedPrecisionOps 保存量化权重及调度信息。本载入器使用 FP16 activation compute，同时对量化模型保持 `force_cast_weights=false`，因此不会主动把受支持的 INT8/ConvRot 权重转换为稠密 FP16。

| 模型或载入路径 | 支持情况 |
|---|---|
| 原生 MiniMax H3 稠密 BF16 / FP32 safetensors | 支持；以 FP16 operations 创建模型并安装数值保护 |
| 原生 `int8_tensorwise` + ConvRot MiniMax H3 | 支持；保留量化权重与原生调度 |
| 带 `model.diffusion_model.` 前缀的原生 H3 权重 | 支持；移除外层前缀后按模型结构识别 |
| 使用文件级 `_quantization_metadata` 的原生 H3 权重 | 支持；恢复对应 MixedPrecisionOps 配置 |
| `convrot_w4a4`、`asym_w4a8_int8` | 保留原生量化调度；需由相应环境完成端到端验证 |
| GGUF、GPTQ、bitsandbytes 或自定义量化加载器 | 不支持；不属于 ComfyUI 原生 MixedPrecisionOps 契约 |
| 非原生或结构已修改的 H3 模型类 | 不支持 |
| 非 MiniMax H3 扩散模型 | 不支持 |

本插件不会把所有计算都变成 INT8。残差累积、溢出保护、归一化和调制边界等数值敏感位置会按修复要求保留 FP32 或 FP16。

## LoRA 注意事项

标准 ComfyUI LoRA 可能以运行时 `weight_function` 的形式附加在权重上。在动态或低显存加载中，受影响的量化层可能因此被临时反量化。插件检测到量化模型存在权重补丁时会输出一次警告，但不会静默合并或重新量化 LoRA。

如需最大限度保留量化内核，应使用明确支持目标 MiniMax H3 量化格式的 LoRA 载入方式，并根据控制台中的 `backend`、`force-cast` 和 `weight-patches` 信息确认实际路径。

## 支持的硬件

| GPU 架构 | 行为 |
|---|---|
| NVIDIA Turing（RTX 20、T4、Quadro RTX、Titan RTX） | 推荐使用 FP16 Loader |
| NVIDIA Volta（V100、Titan V） | 支持 FP16 路径 |
| NVIDIA P100（SM60） | 可使用 FP16 路径，需实机验证目标工作流 |
| NVIDIA P40 / GTX 10（SM61） | 自动跳过，保留 ComfyUI 默认载入路径 |
| NVIDIA Ampere 及更新架构（SM80+） | 自动跳过，使用原生 BF16 路径 |
| AMD ROCm | 实验性支持，取决于当前 PyTorch 与 ComfyUI 环境 |

## 安装与更新

ComfyUI Manager / Comfy Registry：

```text
Search: MiniMax H3 Native FP16 - Star7
Package: minimax-h3-fp16-exact-star7
```

Comfy CLI：

```bash
comfy node install minimax-h3-fp16-exact-star7
```

GitHub 手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/star7code/minimax-h3-fp16-exact-star7.git
```

更新：

```bash
cd ComfyUI/custom_nodes/minimax-h3-fp16-exact-star7
git pull
```

安装或更新后重启 ComfyUI。

## 运行诊断

量化模型的正常启用日志示例：

```text
[Star7 H3 FP16] Enabled v2.0.7 | mode=loader-quantized | backend=int8_tensorwise+convrot:200 | force-cast=False | weight-patches=0 | blocks=50
```

| `mode` | 含义 |
|---|---|
| `loader-quantized` | 原生载入器创建的量化模型 |
| `loader-dense` | 原生载入器创建的稠密 FP16 模型 |
| `postload-quantized` | 兼容节点处理的量化模型 |
| `postload-dense` | 兼容节点处理的稠密模型 |

## 兼容范围

- 目标模型类为 `comfy.ldm.minimax.model.MiniMaxH3Model`。
- 使用 ComfyUI 原生模型检测、ModelPatcher、MixedPrecisionOps、量化、动态卸载和缓存重载机制。
- 不应与另一个作用于相同 H3 block 的进程级 FP16 溢出补丁重复使用。
- ComfyUI 的 MiniMax H3 模型结构或载入接口发生变化时，可能需要同步更新。

## Attribution

FP32 数值区和 2 的整数次幂缩放方法来自 [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix)。原生载入、量化感知调度、硬件检测、ModelPatcher 集成和 ComfyUI 打包由 [Star7](https://github.com/star7code) 维护。

## License

MIT。上游版权声明保留在 [LICENSE](LICENSE) 中。
