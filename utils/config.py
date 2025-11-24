import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def deepseek(self) -> Dict[str, Any]:
        return {
            "model": self.raw["generation"]["model"],
            "base_url": self.raw["env"]["deepseek_base_url"],
            "api_key": os.getenv(self.raw["env"]["deepseek_api_key_env"], ""),
            "temperature": self.raw["generation"].get("temperature", 0.3),
            "system_prompt": self.raw["generation"]["system_prompt"],
        }

    @property
    def embedding(self) -> Dict[str, Any]:
        return self.raw["embedding"]

    @property
    def qdrant(self) -> Dict[str, Any]:
        return self.raw["qdrant"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def reranker(self) -> Dict[str, Any]:
        out = self.raw["reranker"].copy()
        out["api_key"] = os.getenv(self.raw["env"]["silicon_api_key_env"], "")
        return out

    @property
    def vlm(self) -> Dict[str, Any]:
        out = self.raw["vlm"].copy()
        out["api_key"] = os.getenv(self.raw["env"]["dashscope_api_key_env"], "")
        return out

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw["paths"]


def load_config(path: str | Path = "opsfusion_rag/configs/config.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)
