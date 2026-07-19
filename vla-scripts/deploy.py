"""
deploy.py

该脚本提供一个极简的 OpenVLA 推理服务（仅包含服务端），通过 Hugging Face AutoClass API 加载模型并暴露 REST
接口。客户端只需依赖 numpy/json-numpy 与 requests 即可调用，下面示例展示了典型使用方式。

依赖：
    => 服务端（GPU 上运行 OpenVLA）：`pip install uvicorn fastapi json-numpy`
    => 客户端：`pip install requests json-numpy`

客户端示例（假设服务监听在 0.0.0.0:8000）：

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()
```

若服务端不对外开放，可使用 ngrok 或 SSH 端口转发：`ssh -L 8000:localhost:8000 USER@<SERVER_IP>`
"""

import os.path  # 标准库 os.path 用于判断微调目录是否存在、拼接统计文件路径

# ruff: noqa: E402  # 让 ruff 忽略「先 import 第三方模块再 import 标准库」的顺序告警
import json_numpy  # 第三方库 json-numpy，扩展原生 json 以支持 numpy.ndarray 的序列化

json_numpy.patch()  # 打补丁，让全局 json 序列化器自动处理 numpy 类型
import json  # 标准库 json，处理常规 JSON 读写
import logging  # 标准库 logging，记录服务器错误信息
import traceback  # 标准库 traceback，用于打印堆栈
from dataclasses import dataclass  # dataclass 用于快速定义配置结构体
from pathlib import Path  # Path 用于更直观地操作文件路径
from typing import Any, Dict, Optional, Union  # 类型提示，便于阅读与 IDE 检查

import draccus  # 第三方配置库，用装饰器方式解析命令行参数
import torch  # PyTorch，既用于判断 GPU 是否可用，也承载 VLA 推理
import uvicorn  # Uvicorn 是 ASGI 服务器，实现 FastAPI 应用的运行时
from fastapi import FastAPI  # FastAPI 框架，用于声明 REST 接口
from fastapi.responses import JSONResponse  # FastAPI 提供的 JSON 响应工具
from PIL import Image  # Pillow 图像库，把 numpy 数组转为 PIL.Image 以匹配 processor 期望
from transformers import AutoModelForVision2Seq, AutoProcessor  # HF Transformers 提供自动模型/处理器

# === 工具常量 ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)  # 默认系统提示词；字符串保持英文以保持与模型训练一致


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    """根据模型版本（通过路径判断）拼装对应的文本提示"""
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === 服务端主类 ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """初始化 OpenVLA 推理服务，加载 HF AutoClass 模型并准备好推理上下文"""
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation  # 记录模型来源与注意力实现
        self.device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")  # 确定推理设备

        # 通过 AutoProcessor.from_pretrained 加载图像/文本联合处理模块（需 trust_remote_code 支持自定义逻辑）
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        # 通过 AutoModelForVision2Seq.from_pretrained 加载 VLA，本质是 VisionEncoder + LLM 的 Seq2Seq 结构
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,  # 透传注意力实现，如 flash_attention_2
            torch_dtype=torch.bfloat16,  # 推理使用 BF16 以节省显存
            low_cpu_mem_usage=True,  # 允许 transformers 用更低的 CPU 内存姿势加载
            trust_remote_code=True,  # 允许使用 repo 内自定义模型类
        ).to(self.device)  # 将模型整体移动到推理设备

        # 如果传入的是本地微调目录，则尝试读取 dataset_statistics.json 以便推理时做动作反归一化
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)  # 将 mean/std 写入模型属性，predict_action 可直接使用

    def predict_action(self, payload: Dict[str, Any]) -> str:
        """FastAPI 的处理函数：接收图像+指令 JSON，返回动作向量"""
        try:
            if double_encode := "encoded" in payload:
                # 若客户端无法安装 json_numpy，会先把 numpy 数组字符串化，这里解码得到原始 JSON
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            image, instruction = payload["image"], payload["instruction"]  # 必填字段：图像与指令
            unnorm_key = payload.get("unnorm_key", None)  # 可选字段：指定动作反归一化配置

            prompt = get_openvla_prompt(instruction, self.openvla_path)  # 根据模型版本生成提示词
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            # 上面一步：processor 负责 tokenizer + 图像预处理，随后统一搬到目标设备并设定 dtype
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)  # 调用自定义 API 产出动作
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))  # 若输入是双重编码，输出也转成 JSON 字符串
            else:
                return JSONResponse(action)  # 否则直接返回 numpy 数组，json_numpy 会帮忙序列化
        except:  # noqa: E722
            logging.error(traceback.format_exc())  # 记录完整堆栈，便于排查
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )  # 额外输出告警，提醒调用者检查 payload
            return "error"  # FastAPI 会把字符串直接返回给客户端

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()  # 实例化 FastAPI 应用
        self.app.post("/act")(self.predict_action)  # 注册 POST /act 路由并绑定处理函数
        uvicorn.run(self.app, host=host, port=port)  # 用 Uvicorn 启动 ASGI 服务器


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "openvla/openvla-7b"               # 指向 HF Hub 模型或本地 checkpoint

    host: str = "0.0.0.0"                                               # FastAPI 监听 IP（0.0.0.0 表示对外开放）
    port: int = 8000                                                    # 服务监听端口

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path)  # 根据配置加载模型
    server.run(cfg.host, port=cfg.port)  # 启动服务并阻塞至进程结束


if __name__ == "__main__":
    deploy()  # 允许 python vla-scripts/deploy.py 直接运行
