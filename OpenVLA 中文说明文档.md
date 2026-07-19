# OpenVLA：一个开源的视觉-语言-动作模型

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**快速开始**](#getting-started) | [**预训练模型**](#pretrained-vlas) | [**安装**](#installation) | [**LoRA 微调**](#fine-tuning-openvla-via-lora) | [**全量微调**](#fully-fine-tuning-openvla) |
[**从头训练**](#training-vlas-from-scratch) | [**评估**](#evaluating-openvla) | [**项目主页**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>

<a id="latest-updates"></a>
## 最新更新
- [2025-03-03] 近期发布了 OFT（Optimized Fine-Tuning recipe for VLAs）！与“原版 OpenVLA 微调”相比，OFT 可实现 25-50× 更快的推理速度、更高的任务成功率、支持多张输入图像以及高频率的双臂机器人控制。与 FAST 不同，OFT 使用连续动作，从而获得更好的模型质量。项目网站见 [这里](https://openvla-oft.github.io/)。
- [2025-01-16] 近期发布了 FAST 动作 tokenizer！与原版 OpenVLA 风格的 256-bin 动作离散化相比，FAST 允许将动作 chunk 压缩为更少的 token，在使用离散机器人动作时可将推理速度提升最多约 15×。项目网站见 [这里](https://www.physicalintelligence.company/research/fast)。
- [2024-10-15] 在 README 中新增 [VLA 性能排障](#vla-performance-troubleshooting) 章节，提供微调后 VLA 表现不佳时的排查最佳实践。
- [2024-09-04] 在论文中增加了 LIBERO 仿真基准的微调实验（见 [arXiv](https://arxiv.org/abs/2406.09246) v2）；并在 README 的 [LIBERO Simulation Benchmark Evaluations](#libero-simulation-benchmark-evaluations) 章节中加入复现实验说明。
- [2024-09-04] 在论文中增加了 LIBERO 仿真基准的微调实验（见 [arXiv](https://arxiv.org/abs/2406.09246) v2）；并在 README 的 [LIBERO 仿真基准评估](#libero-simulation-benchmark-evaluations) 章节中加入复现实验说明。
- [2024-08-14] 新增 [评估 OpenVLA](#evaluating-openvla) 章节，包含在 BridgeData V2 WidowX 机器人上运行评估的说明。
- [2024-07-08] 新增章节：[通过 LoRA 微调 OpenVLA](#fine-tuning-openvla-via-lora)、[全量微调 OpenVLA](#fully-fine-tuning-openvla)。
- [2024-06-13] 初始发布

<hr style="border: 2px solid gray;"></hr>

一个简单且可扩展的代码库，用于训练与微调视觉-语言-动作模型（VLA），面向通用机器人操作：

- **不同的数据集混合（Dataset Mixtures）**：原生支持 RLDS 格式的任意数据集（以及任意混合方式），包括来自 [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/) 的数据。
- **易于扩展（Scaling）**：基于 PyTorch FSDP 与 Flash-Attention，可快速高效训练 1B - 34B 参数量的模型，并能方便地适配模型结构。
- **原生支持微调（Fine-Tuning）**：内置支持（含示例）多种微调方式（全量、部分、LoRA）。

基于 [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) 构建。

<a id="getting-started"></a>
## 快速开始

为了快速加载并运行 OpenVLA 模型进行推理，我们提供了一个轻量接口，利用 HuggingFace `transformers` 的 AutoClasses，并尽量减少依赖。

例如：在 [BridgeData V2 环境](https://rail-berkeley.github.io/bridgedata/) 中，使用 WidowX 机器人加载 `openvla-7b` 做零样本指令跟随：

```python
# 安装最小依赖（`torch`、`transformers`、`timm`、`tokenizers` 等）
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# 加载 Processor 与 VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
  attn_implementation="flash_attention_2",  # [可选] 需要 `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# 获取图像输入并构造 prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# 预测动作（7-DoF；为 BridgeData V2 反归一化）
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 执行...
robot.act(action, ...)
```

我们还提供了一个 [用于对新任务/新本体（embodiments）微调 OpenVLA 的示例脚本](./vla-scripts/finetune.py)；该脚本支持不同的微调模式——包括由 [HuggingFace PEFT 库](https://huggingface.co/docs/peft/en/index) 支持的（量化）低秩适配（LoRA）。

在部署方面，我们提供了一个轻量脚本，用于 [通过 REST API 的形式提供 OpenVLA 服务](./vla-scripts/deploy.py)，便于将 OpenVLA 集成到现有的机器人控制栈中，并避免对本地强算力的依赖。

<a id="pretrained-vlas"></a>
## 预训练 VLA 模型

我们发布了两款 OpenVLA 模型；其检查点、配置以及模型卡可在 [Hugging Face 页面](https://huggingface.co/openvla) 获取：
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b)：论文中的旗舰模型，从 Prismatic `prism-dinosiglip-224px` VLM 初始化（融合 DINOv2 与 SigLIP 的视觉主干，以及 Llama-2 LLM）。在覆盖约 970K 轨迹的 Open X-Embodiment 大规模混合数据上训练（[混合细节见“Open-X Magic Soup++”](./prismatic/vla/datasets/rlds/oxe/mixtures.py)）。
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01)：开发阶段的早期模型，从 Prismatic `siglip-224px` VLM 初始化（单 SigLIP 视觉主干，以及 Vicuña v1.5 LLM）。在与 [Octo](https://github.com/octo-models/octo) 相同的混合数据上训练，但使用的 GPU 小时数显著少于最终模型（[混合细节见“Open-X Magic Soup”](./prismatic/vla/datasets/rlds/oxe/mixtures.py)）。

**关于模型许可与商用的明确说明**：尽管本仓库所有代码以 MIT License 发布，但我们提供的预训练模型可能继承底座模型的限制。具体而言，上述两个模型均衍生自 Llama-2，因此受 [Llama Community License](https://ai.meta.com/llama/license/) 约束。

---

<a id="installation"></a>
## 安装

> **注意**：这些安装说明面向全量预训练（以及分布式微调）。如果你只是想运行推理（或做轻量微调），请参考上面的说明！


本仓库使用 Python 3.10 构建，但应向后兼容 Python >= 3.8。我们要求 PyTorch 2.2.*——安装说明可参考 [这里](https://pytorch.org/get-started/locally/)。本仓库最新版本开发并充分测试时使用的版本为：
  - PyTorch 2.2.0、torchvision 0.17.0、transformers 4.40.1、tokenizers 0.19.1、timm 0.9.10，以及 flash-attn 2.5.5

**[5/21/24] 注意**：根据社区反馈，在 `transformers`、`timm`、`tokenizers` 的更高版本中出现了回归与破坏性变更，我们因此明确固定（pin）了上述依赖版本。我们正在实现更完善的测试，并计划在可行时放宽这些约束。

按如下命令开始安装：

```bash
# 创建并激活 conda 环境
conda create -n openvla python=3.10 -y
conda activate openvla

# 安装 PyTorch。下面是示例命令，但你应根据你的计算平台参考以下链接：
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# 克隆并安装 openvla 仓库
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 为训练安装 Flash Attention 2（https://github.com/Dao-AILab/flash-attention）
#   =>> 若遇到安装问题，先尝试运行 `pip cache remove flash_attn`
pip install packaging ninja
ninja --version; echo $?  # 验证 Ninja ——应返回退出码 "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

如果安装过程中遇到任何问题，请提交 GitHub Issue。

**注意**：完整的 OpenVLA 训练与验证脚本在 `vla-scripts/` 中。请注意 `scripts/` 多来自原始（基础）`prismatic-vlms` 仓库，用于训练与评估“视觉条件语言模型”（VLM）；尽管你可以用本仓库训练 VLM 与 VLA，但请注意：对现有 OpenVLA 模型使用 `scripts/generate.py` 生成语言将无法工作（因为我们训练当前的 OpenVLA 模型只生成动作，且只生成动作）。

<a id="fine-tuning-openvla-via-lora"></a>
## 通过 LoRA 微调 OpenVLA

**（2025-03-03 更新：我们建议尝试新的 OFT 微调配方，使 OpenVLA 生成更快且更成功的策略。项目网站见 [这里](https://openvla-oft.github.io/)。）**

本节讨论如何通过 Hugging Face `transformers` 微调 OpenVLA（LoRA：Low-Rank Adaptation）。如果你没有足够算力对 7B 参数模型做全量微调，LoRA 是推荐方案。LoRA 微调的主脚本是 `vla-scripts/finetune.py`。（如果你希望做全量微调，请参阅 [全量微调 OpenVLA](#fully-fine-tuning-openvla)）。

下面展示如何对主要的 OpenVLA checkpoint（[`openvla-7b`](https://huggingface.co/openvla/openvla-7b)）进行 LoRA 微调。我们在单张 80GB A100 上对 [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) 进行微调。（如果 GPU 更小，只要显存至少约 27GB，也可以通过调整 batch size 来微调。）

首先，下载 BridgeData V2 数据集：

```bash
# 切换到你的数据集根目录
cd <PATH TO BASE DATASETS DIR>

# 下载完整数据集（124 GB）
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 将数据集重命名为 `bridge_orig`（注意：省略此步可能会导致后续运行时报错）
mv bridge_dataset bridge_orig
```

然后，如下启动 LoRA 微调脚本。注意 `--batch_size==16` 且 `--grad_accumulation_steps==1` 需要约 72GB 显存。如果 GPU 更小，你应降低 `--batch_size` 并提高 `--grad_accumulation_steps` 来保持足够大的有效 batch 以获得稳定训练。如果你有多张 GPU 并希望用 PyTorch DDP 训练，只需将下方 `torchrun` 的 `--nproc-per-node` 设置为可用 GPU 数量。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

注意：如果你在上述命令中将 `--image_aug==False`，你会在训练日志中看到接近 100% 的 `action_accuracy`，因为 [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) 已经在包含 BridgeData V2 的更大数据集超集上预训练过（且预训练时未使用增强）。

如果要对其他数据集做 LoRA 微调，你可以从 [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) 混合中下载（可参考 [该自定义脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) 了解如何从 OXE 下载数据集）。或者，如果你有不在 OXE 中的自定义数据集，你可以（a）将其转换为与我们的微调脚本兼容的 RLDS 格式（可参考 [此仓库](https://github.com/kpertsch/rlds_dataset_builder)），或（b）使用你自己的 PyTorch Dataset 封装（见 `vla-scripts/finetune.py` 中的注释）。我们对大多数用户推荐（a）；因为 RLDS 数据集与 dataloader 在我们的预训练/微调实验中被更充分地测试。

对于选项（a），将数据集转换为 RLDS 后，你需要在我们的数据加载器中注册它：在 [这里](prismatic/vla/datasets/rlds/oxe/configs.py#L54) 注册 dataset config，并在 [这里](prismatic/vla/datasets/rlds/oxe/transforms.py#L828) 注册 dataset transform 函数。

当你完成新数据集的接入后，就可以使用同一个 `vla-scripts/finetune.py` 脚本启动 LoRA 微调。如果遇到问题，请查看 [VLA 故障排除](#vla-troubleshooting) 或在 [OpenVLA GitHub Issues](https://github.com/openvla/openvla/issues?q=)（包括已关闭 issue）中检索是否有类似问题；如果找不到，欢迎创建新 issue。

<a id="fully-fine-tuning-openvla"></a>
## 全量微调 OpenVLA

**（2025-03-03 更新：我们建议尝试新的 OFT 微调配方，使 OpenVLA 生成更快且更成功的策略。项目网站见 [这里](https://openvla-oft.github.io/)。）**

本节讨论如何使用 [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) 的训练脚本，通过原生 PyTorch Fully Sharded Data Parallel（FSDP）对 OpenVLA 进行 <ins>全量微调</ins>（全部 7.5B 参数）。全量微调更复杂/更进阶，只有在你有足够算力（例如一整机 8 张 A100）且 LoRA 微调不满足需求（例如微调分布与预训练分布差异很大）时才推荐。否则，我们建议使用参数高效微调（LoRA），见 [通过 LoRA 微调 OpenVLA](#fine-tuning-openvla-via-lora)。

进行全量微调时，你需要下载一个与 Prismatic VLMs 代码库兼容的 OpenVLA checkpoint（我们基于该代码库开发 OpenVLA）。你需要下载 [另一个版本的 OpenVLA checkpoint](https://huggingface.co/openvla/openvla-7b-prismatic)。你可以用以下 git 命令下载这个 Prismatic 兼容的 checkpoint（或者使用 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 下载）：

```bash
# 切换到你的模型 checkpoint 根目录
cd <PATH TO BASE MODEL CHECKPOINTS DIR>

# 下载 checkpoint（30 GB）——可能需要几分钟
git clone git@hf.co:openvla/openvla-7b-prismatic

# 如果上面的命令没有下载完整的 checkpoint，
# 请通过 git Large File Storage（LFS）手动拉取
# 注意：你可能需要先配置 SSH key 才能生效
cd openvla-7b-prismatic
git lfs fetch --all
```

我们展示如何在单节点 8 GPU 上对 [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) 进行全量微调。如果你希望使用不同数量的 GPU（或节点），你可以在 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 中修改 VLA 训练配置。

下载 BridgeData V2 数据集：

```bash
# 切换到你的数据集根目录
cd <PATH TO BASE DATASETS DIR>

# 下载完整数据集（124 GB）
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 将数据集重命名为 `bridge_orig`（注意：省略此步可能会导致后续运行时报错）
mv bridge_dataset bridge_orig
```

接下来，创建一个 [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens)，并将 token 值（以 `hf_...` 开头的字符串）写入本仓库根目录的 `.hf_token` 文件（`openvla/.hf_token`）：

```bash
# 进入 openvla 仓库根目录
cd openvla

# 将 HF token 写入 token 文件。请将 "hf_..." 替换为你自己的 token！
# 参见：https://huggingface.co/docs/hub/en/security-tokens
echo hf_... >>> .hf_token
```

现在启动训练脚本。如果你希望使用不同数量的节点或 GPU，请修改 [`prismatic/conf/vla.py`](prismatic/conf/vla.py) 中的 VLA 训练配置，并相应调整下面命令的 `--nnodes` 与 `--nproc-per-node`。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <PATH TO openvla/openvla-7b-prismatic CHECKPOINT FILE: step-295000-epoch-40-loss=0.2200.pt> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --run_id <可选：用于 WANDB 日志的 RUN ID> \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_interval <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> \
  --is_resume False
```

注意：上面将 `--is_resume` 设置为 `False`，因为我们是在微调一个预训练 checkpoint，而不是恢复一个暂停的训练。

如果你的训练被暂停，且你希望从最新 checkpoint 恢复训练，请将 `--pretrained_checkpoint` 改为最新 checkpoint 路径，并将 `--is_resume==True`，同时指定 `--resume_step` 与 `--resume_epoch`（分别为 step 与 epoch 号）。例如，如果你希望从名为 `step-010000-epoch-20-loss=0.0160.pt` 的 checkpoint 恢复，应设置 `is_resume==True`、`resume_step==10000`、`resume_epoch==20`。

注意：如果你运行上面的 BridgeData V2 全量微调命令，你应在训练日志中观察到接近 100% 的 Action Token Accuracy，因为 [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) 已在包含 BridgeData V2 的更大数据集超集上预训练过。

要在其他数据集上全量微调 OpenVLA，你可以从 [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) 混合中下载数据集（见 [该自定义脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh)），或将不在 OXE 中的自定义数据集转换为与微调脚本兼容的 RLDS 格式（见 [此仓库](https://github.com/kpertsch/rlds_dataset_builder)）。下载/转换后，你需要修改以下文件：

* [`prismatic/conf/vla.py`](prismatic/conf/vla.py)：通过创建一个 experiment class 添加新的训练配置，并在文件底部的 `VLARegistry` 中注册。
  * 请确保为你的微调运行创建一个新的唯一 `vla_id`，并根据需要调整一些配置变量——例如 `expected_world_size`（GPU 数量）、`per_device_batch_size`（每卡 batch）、`global_batch_size`（总 batch）、`shuffle_buffer_size`（每卡 shuffle buffer 中样本数）等。请阅读文件顶部 `VLAConfig` 类下的注释以理解每个变量的用途。
* [`prismatic/vla/datasets/rlds/oxe/mixtures.py`](prismatic/vla/datasets/rlds/oxe/mixtures.py)：在 `OXE_NAMED_MIXTURES` 字典中为你的微调混合定义一个新 mixture。
* [`prismatic/vla/datasets/rlds/oxe/transforms.py`](prismatic/vla/datasets/rlds/oxe/transforms.py)：为你的微调数据集定义一个新的 dataset transform 函数，并将其添加到文件底部的 `OXE_STANDARDIZATION_TRANSFORMS` registry。
* [`prismatic/vla/datasets/rlds/oxe/configs.py`](prismatic/vla/datasets/rlds/oxe/configs.py)：在 `OXE_DATASET_CONFIGS` 字典中增加一个新配置，用于描述你的微调数据集的 observation 与 action 空间。

完成上述步骤后，你就可以使用 `vla-scripts/train.py` 启动全量微调。请确保 `--vla.type` 参数设置为你在 `prismatic/conf/vla.py` 中添加的新的 `vla_id`。

微调完成后，你需要将最终的模型 checkpoint 转换为与 Hugging Face `transformers` AutoClasses 兼容的版本。请参阅 [将 Prismatic 模型转换为 Hugging Face](#converting-prismatic-models-to-hugging-face)。

如遇问题，请查看 [VLA 故障排除](#vla-troubleshooting)，或在 [OpenVLA GitHub Issues](https://github.com/openvla/openvla/issues?q=)（包括已关闭 issue）中检索；如无类似问题，欢迎创建新 issue。

<a id="converting-prismatic-models-to-hugging-face"></a>
### 将 Prismatic 模型转换为 Hugging Face

如果你使用 Prismatic VLMs 代码库训练模型（例如在新数据集上对 OpenVLA 做全量微调），你需要将最终 checkpoint 转换为与 Hugging Face `transformers` AutoClasses 兼容的版本。本节说明如何转换。

假设你的训练运行目录是 `PRISMATIC_RUN_DIR`（例如 `prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7`）。在该目录下应有一个 `checkpoints` 目录，包含保存的模型 checkpoint（例如 `step-295000-epoch-40-loss=0.2200.pt`）。Prismatic-to-Hugging-Face 转换脚本（[convert_openvla_weights_to_hf.py](vla-scripts/extern/convert_openvla_weights_to_hf.py)）期望存在一个名为 `latest-checkpoint.pt` 的 checkpoint 文件。因此，你应先创建一个名为 `latest-checkpoint.pt` 的符号链接，指向你希望转换的 checkpoint 文件：

```bash
# 进入你的 Prismatic 训练运行目录下的 `checkpoints` 目录
cd PRISMATIC_RUN_DIR/checkpoints

# 创建一个符号链接，指向你想要转换的 checkpoint 文件
ln -s <YOUR CHECKPOINT FILENAME> latest-checkpoint.pt
```

然后，运行转换脚本，将 Prismatic VLMs 格式的 checkpoint 转换为 Hugging Face 格式：

```bash
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
    --output_hf_model_local_path <OUTPUT DIR FOR CONVERTED CHECKPOINT>
```

上述命令会将 HF 兼容的 checkpoint 保存到 `output_hf_model_local_path`。随后你就可以像平常一样用 HF AutoClasses 加载。需要注意：如果你加载的是本地保存的 checkpoint（而不是推送到 HF Hub），则在加载前还需要将 OpenVLA 模型注册到 HF AutoClasses（详见 [这里](https://huggingface.co/docs/transformers/en/custom_models#registering-a-model-with-custom-code-to-the-auto-classes)）。

```python
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# 将 OpenVLA 模型注册到 HF AutoClasses（如果你已将模型推送到 HF Hub，则不需要）
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# 加载 Processor 与 VLA
processor = AutoProcessor.from_pretrained("<PATH TO CONVERTED CHECKPOINT DIR>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<PATH TO CONVERTED CHECKPOINT DIR>",
  attn_implementation="flash_attention_2",  # [可选] 需要 `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

...
```

<a id="training-vlas-from-scratch"></a>
## 从头训练 VLA

我们提供了在（任意子集的）[Open X-Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/) 上从头训练 VLA 模型的完整说明与配置。如遇问题，请参阅下方 [VLA 故障排除](#vla-troubleshooting)（或提交 GitHub Issue）。

### VLA 预训练数据集

我们遵循 [这个自定义脚本](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) 以 [RLDS 格式](https://github.com/google-research/rlds) 下载并预处理 Open X-Embodiment 中的各个数据集。`openvla-7b` 训练所用的完整数据集列表（及混合权重）见 [mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py)。
- **重要**：对于 BridgeData V2 组件，OXE 中的版本较旧（截至 2023-12-20）。你应从 [官网](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) 下载数据集，并放在子目录 `bridge_orig/` 下。将 OXE 代码中对 `bridge` 的任何引用替换为 `bridge_orig`。

### VLA 配置与训练脚本

VLA 训练入口是 [`vla-scripts/train.py`](vla-scripts/train.py)。我们使用 [`draccus`](https://pypi.org/project/draccus) 提供模块化、基于 dataclass 的配置接口；现有的 VLA 配置位于 [`prismatic/conf/vla.py`](prismatic/conf/vla.py)。你可以添加自己的训练配置，并通过命令行参数 `--vla.type` 进行引用。

我们使用 PyTorch Fully Sharded Data Parallel（FSDP）在多 GPU 上分布式训练。通过 `torchrun` 启动训练：

```bash
# 在单节点（8 GPU）上，使用 Prismatic DINO-SigLIP 224px 视觉主干，在 BridgeData V2 上训练 VLA
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

<a id="vla-troubleshooting"></a>
### VLA 故障排除

以下是一些已知问题与对应修复：

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **修复**：降级 `tensorflow-datasets`：`pip install tensorflow-datasets==4.9.3`。


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **修复**：升级 `dlimp` 到最新版本。你可能需要使用 `--force-reinstall`：
`pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`

---

<a id="evaluating-openvla"></a>
## 评估 OpenVLA

### BridgeData V2 WidowX 评估

#### 环境准备

克隆 [BridgeData V2 WidowX controller repo](https://github.com/rail-berkeley/bridge_data_robot) 并安装 `widowx_envs`：

```bash
git clone https://github.com/rail-berkeley/bridge_data_robot.git
cd bridge_data_robot
pip install -e widowx_envs
```

另外安装 [`edgeml`](https://github.com/youliangtan/edgeml) 库：
```bash
git clone https://github.com/youliangtan/edgeml.git
cd edgeml
pip install -e .
```

按照 `bridge_data_robot` README 的说明创建 Bridge WidowX Docker 容器。

#### 启动 BridgeData V2 评估

运行 BridgeData V2 评估有多种方式。这里描述 server-client 方法。

在一个 Terminal 窗口中（例如 tmux），启动 WidowX Docker 容器：

```bash
cd bridge_data_robot
./generate_usb_config.sh
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

在第二个 Terminal 窗口中，运行 WidowX robot server：

```bash
cd bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"
```

在第三个 Terminal 窗口中，运行 OpenVLA 策略评估脚本：

```bash
cd openvla
python experiments/robot/bridge/run_bridgev2_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b
```

如果你遇到类似 `ModuleNotFoundError: No module named 'moviepy.editor'` 的错误，你可以通过将 `bridge_data_robot` 仓库的 requirements.txt 中的 moviepy 版本固定为较旧的 v1.0.3 来绕过该问题（位置见 [这里](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/requirements.txt)）。也就是将 `moviepy` 替换为 `moviepy==1.0.3`。然后回到第一步并重新启动 Docker 容器（它会用旧版本 moviepy 重建）。


<a id="libero-simulation-benchmark-evaluations"></a>
### LIBERO 仿真基准评估

在 [更新后的 OpenVLA 论文（v2）](https://arxiv.org/abs/2406.09246) 中，我们在附录 E 讨论了在仿真基准 [LIBERO](https://libero-project.github.io/main.html) 上对 OpenVLA 的微调。请阅读论文获取细节，例如我们如何修改提供的演示数据集以提升各方法的整体表现。

我们将结果复制到下方，并说明如何复现 OpenVLA 的结果。

#### OpenVLA 微调结果

| 方法 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | 平均 |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo fine-tuned | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA fine-tuned (ours) | **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |

每个成功率是 3 个随机种子 × 每个种子 500 次 rollout 的平均值（10 个任务 × 每个任务 50 次）。

#### LIBERO 环境准备

克隆并安装 [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO)：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

另外安装其他必需包：
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

（可选）若要下载我们在微调实验中使用的“修改版 LIBERO 数据集”，运行下面命令。这会下载 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、以及 LIBERO-10 数据集的 RLDS 格式数据（总计约 10GB）。你可以用它来微调 OpenVLA 或训练其他方法。该步骤可选，因为我们也提供了预训练的 OpenVLA checkpoint（见下）。
（你也可以在 [这里](experiments/robot/libero/regenerate_libero_dataset.py) 找到我们生成修改版数据集的脚本（原始 HDF5 格式），以及在 [这里](https://github.com/moojink/rlds_dataset_builder) 找到我们将其转换为 RLDS 格式的代码。）
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

#### 启动 LIBERO 评估

我们通过 LoRA（r=32）分别在四个 LIBERO 任务套件上独立微调 OpenVLA：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、以及 LIBERO-10（也称 LIBERO-Long）。这四个 checkpoint 可在 Hugging Face 获取：
* [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
* [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)
* [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)
* [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)

要用其中一个 checkpoint 启动评估，运行以下命令之一。每个命令都会自动下载相应 checkpoint。

```bash
# 启动 LIBERO-Spatial 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# 启动 LIBERO-Object 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# 启动 LIBERO-Goal 评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# 启动 LIBERO-10（LIBERO-Long）评估
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

注意：
* 评估脚本默认运行 500 次试验（10 个任务 × 每个任务 50 个 episode）。你可以通过 `--num_trials_per_task` 修改每个任务的试验次数；也可以通过 `--seed` 修改随机种子。
* **注意：设置 `--center_crop True` 很重要**，因为我们在训练时使用了随机裁剪增强（每个训练样本随机裁剪 90% 面积，因此测试时我们直接取中心 90% 裁剪）。
* 评估脚本会在本地记录结果。你也可以通过设置 `--use_wandb True` 并指定 `--wandb_project <PROJECT>` 与 `--wandb_entity <ENTITY>` 将结果记录到 Weights & Biases。
* 论文中报告的结果是在 **Python 3.10.13、PyTorch 2.2.0、transformers 4.40.1、flash-attn 2.5.5** 以及 **NVIDIA A100 GPU** 上得到的，并对三个随机种子取平均。请尽量使用这些版本。
  注意：如果你在评估时使用不同 GPU，由于大模型在 GPU 上的非确定性，结果可能会略有差异（但我们测试过不同 A100 机器结果一致）。

如果你遇到任何问题，请提交 GitHub Issue。

---

<a id="repository-structure"></a>
## 项目结构

仓库/项目文件树的高层概览：

+ `prismatic` - 包源码；提供模型加载、训练、数据预处理等核心工具。
+ `vla-scripts/` - VLA 训练、微调与部署的核心脚本。
+ `experiments/` - 在机器人环境中评估 OpenVLA 策略的代码。
+ `LICENSE` - 所有代码以 MIT License 发布；欢迎使用！
+ `Makefile` - 顶层 Makefile（默认支持 linting：检查与自动修复）；可按需扩展。
+ `pyproject.toml` - 完整项目配置（含依赖）以及工具配置。
+ `README.md` - 你正在读的文件。

---


<a id="vla-performance-troubleshooting"></a>
# VLA 性能排障

本节覆盖在目标域机器人数据集上微调后，VLA 表现不佳时的排查最佳实践。

**注意**：OpenVLA 通常需要在你的目标域机器人上采集一个小规模的演示数据集（约 ~100 条 demo）进行微调。开箱即用时，它只在训练数据覆盖的域上表现较好。

**基本自检（Sanity checks）**：
- 回放微调数据集中的某条演示动作，并确保机器人能成功执行任务（这能确保你的数据采集流水线是正确的）。
- 当你微调出一个模型后，将模型加载到你的推理流水线（就像你会用它控制机器人那样），但输入使用微调数据集中的图像（假装它们来自机器人相机），并验证你可以复现训练时的 token accuracy / L1 error（这能确保你的推理流水线是正确的）。

**微调数据采集最佳实践**:
如果你的设置通过了上述两个自检，那么问题可能不在模型训练，而在你用于微调的数据中。以下是一些数据采集最佳实践：
- *以约 5-10Hz 的控制频率采集。* OpenVLA 未使用 action chunking 进行训练，经验上模型在高频数据上表现不佳。如果你的机器人使用高频控制器（例如 50Hz），考虑将动作下采样到 5Hz。首先验证当使用 5Hz 动作时，机器人仍能完成任务（即用 5Hz 动作重做上面的自检（1））。
- *采集时避免停顿/小动作。* 因为 OpenVLA 未使用 action chunking 训练，它可能对微调数据中的 idle 动作很敏感。如果你的数据包含机器人几乎不动的步骤，模型在推理时可能会在这些步骤“卡住”。尽量采集连续、缓慢移动的微调演示。
- *确保足够的数据覆盖。* 如果你计划在推理时测试某些变化（例如物体不同初始位置），确保你的微调数据中也包含足够多样的这些条件（例如包含不同初始条件的演示）。
- *采集时使用一致的任务策略。* 这不是硬性约束，但可能会让你的工作更轻松。尽量以一致方式演示任务，例如从同一侧接近物体，即使子步骤可以按任意顺序执行，也尽量按相同顺序执行。保持一致能让你的微调数据集多模态更少，从而使建模问题更容易。


---

<a id="citation"></a>
#### 引用

如果你觉得我们的代码或模型对你的工作有用，请引用 [我们的论文](https://arxiv.org/abs/2406.09246)：

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
