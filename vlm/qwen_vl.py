from __future__ import annotations

import base64
from pathlib import Path
from typing import List

import requests


class QwenVLClient:
    """轻量封装 Qwen3-VL-Flash 的描述接口（DashScope 兼容模式）。"""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _file_to_b64(self, path: Path) -> str:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def describe_image(self, image_path: Path, prompt: str | None = None) -> str:
        """为单张图片生成描述，返回纯文本。"""
        img_b64 = self._file_to_b64(image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "请用中文简洁描述这张图的关键信息。"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def describe_images(self, image_paths: List[Path], prompt: str | None = None) -> List[str]:
        return [self.describe_image(p, prompt) for p in image_paths]
