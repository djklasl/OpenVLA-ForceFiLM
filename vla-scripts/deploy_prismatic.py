"""
deploy_prismatic.py

该脚本提供一个极简的 PrismaticVLM 推理服务（仅包含服务端），加载 PrismaticVLM 并暴露 REST 接口。

与 [vla-scripts/deploy.py](vla-scripts/deploy.py) 的差异：
- 输入仍为 image + instruction
- 固定做“六选一”：要求模型只在 1..6 中选择一个
- 输出为该数字对应的 tokenizer token id（一个整数）

依赖：
    => 服务端（GPU 上运行 PrismaticVLM）：`pip install uvicorn fastapi json-numpy`
    => 客户端：`pip install requests json-numpy`

客户端示例（假设服务监听在 0.0.0.0:8000）：

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

token_id = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "应该怎么移动？"}
).json()

print(token_id)  # e.g. 29896 (取决于 tokenizer)
```

若服务端不对外开放，可使用 ngrok 或 SSH 端口转发：`ssh -L 8000:localhost:8000 USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

from prismatic.models.load import load as load_prismatic
from prismatic.models.vlms import PrismaticVLM


# 与 deploy.py 保持一致的默认系统提示（具体格式仍由 PromptBuilder 决定）
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


ACTION6_CHOICES = {
    1: "前进",
    2: "后退",
    3: "向左移动",
    4: "向右移动",
    5: "向上移动",
    6: "向下移动",
}


def build_action6_prompt(vlm: PrismaticVLM, instruction: str) -> str:
    """构造一个“六选一”约束输出的 prompt，要求模型只输出 1-6 的单个数字。"""
    choices_text = "\n".join([f"选项{i}：{ACTION6_CHOICES[i]}" for i in range(1, 7)])
    message = (
        f"{instruction}\n\n"
        "你必须从下面六个选项中选择一个，并且只输出对应的序号（1,2,3,4,5,6）之一。"
        "不要输出任何解释、文字或标点。\n"
        f"{choices_text}\n\n"
        "回答："
    )
    prompt_builder = vlm.get_prompt_builder(system_prompt=SYSTEM_PROMPT)
    prompt_builder.add_turn(role="human", message=message)
    return prompt_builder.get_prompt()


class PrismaticVLMServer:
    def __init__(self, prismatic_id_or_path: Union[str, Path]) -> None:
        self.prismatic_id_or_path = prismatic_id_or_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 通过项目自带 loader 加载 PrismaticVLM（支持本地 run_dir 或 registry/HF 下载）
        self.vlm = load_prismatic(self.prismatic_id_or_path)
        self.vlm.to(self.device)
        self.vlm.eval()

    def predict_action6_choice(self, payload: Dict[str, Any]) -> Any:
        """接收图像+指令，固定做六选一，返回所选数字对应的 tokenizer token id（一个整数）。"""
        try:
            if double_encode := "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            image, instruction = payload["image"], payload["instruction"]
            return_probs = bool(payload.get("return_probs", False))

            prompt_text = build_action6_prompt(self.vlm, instruction)
            tokenizer = self.vlm.llm_backbone.tokenizer
            image_transform = self.vlm.vision_backbone.image_transform

            encoded = tokenizer(prompt_text, truncation=True, return_tensors="pt")
            input_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device) if hasattr(encoded, "attention_mask") else None

            pixel_values = image_transform(Image.fromarray(image).convert("RGB"))
            if isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values[None, ...].to(self.device)
            elif isinstance(pixel_values, dict):
                pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
            else:
                raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

            # 候选 token：数字 1..6。若某个 tokenizer 不是单 token，则退化为生成并解析。
            candidate_token_ids: Dict[int, int] = {}
            for i in range(1, 7):
                ids = tokenizer.encode(str(i), add_special_tokens=False)
                if len(ids) != 1:
                    candidate_token_ids = {}
                    break
                candidate_token_ids[i] = ids[0]

            if not candidate_token_ids:
                raise ValueError(
                    "Tokenizer does not represent digits '1'..'6' as single tokens; cannot return a single token id."
                )

            if candidate_token_ids:
                autocast_enabled = self.device.type == "cuda" and self.vlm.enable_mixed_precision_training
                if self.device.type == "cuda":
                    autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
                    autocast_ctx = torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled)
                else:
                    autocast_ctx = torch.autocast("cpu", enabled=False)

                with autocast_ctx:
                    out = self.vlm(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        use_cache=False,
                        return_dict=True,
                    )

                logits = out.logits[0, -1, :]
                token_ids = torch.tensor([candidate_token_ids[i] for i in range(1, 7)], device=logits.device)
                choice_logits = logits[token_ids]
                probs = torch.softmax(choice_logits, dim=0)
                choice_idx = int(torch.argmax(probs).item()) + 1
                token_id = int(candidate_token_ids[choice_idx])

                resp: Any = {"token_id": token_id}
                if return_probs:
                    resp["probs"] = probs.detach().cpu().tolist()

            if double_encode:
                return JSONResponse(json_numpy.dumps(resp))
            return JSONResponse(resp["token_id"] if not return_probs else resp)

        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "Optional: `return_probs: bool`."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action6_choice)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployPrismaticConfig:
    # fmt: off
    prismatic_id_or_path: Union[str, Path] = "siglip-224px+7b"  # registry 名称或本地 run 目录

    host: str = "0.0.0.0"
    port: int = 8000
    # fmt: on


@draccus.wrap()
def deploy_prismatic(cfg: DeployPrismaticConfig) -> None:
    server = PrismaticVLMServer(cfg.prismatic_id_or_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy_prismatic()
