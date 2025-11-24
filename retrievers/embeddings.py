from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

from easyrag_langchain.utils.device import pick_device


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2归一化函数

    对向量进行L2归一化，使向量的欧几里得范数为1。
    这对于余弦相似度计算至关重要，确保所有向量在同一尺度空间中比较。

    Args:
        arr: 待归一化的numpy数组，形状为(batch_size, vector_dim)

    Returns:
        np.ndarray: 归一化后的数组，形状与输入相同

    Note:
        添加1e-12防止除零错误，提高数值稳定性
    """
    norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norm


class LocalBGEM3:
    """本地 BGE-M3 模型封装类

    封装BAAI/bge-m3多语言嵌入模型，同时支持稠密向量和稀疏向量生成。
    这是混合向量存储系统的核心组件，实现语义理解和关键词匹配的双重能力。

    BGE-M3 特性：
    - 支持多种语言（中英双语优化）
    - 生成1024维稠密向量（语义理解）
    - 生成稀疏向量（关键词匹配）
    - 支持长文本输入

    Attributes:
        model: BGE-M3 FlagEmbedding模型实例
        batch_size: 批处理大小，影响GPU内存使用和处理速度
        normalize: 是否进行L2归一化，通常为True以确保余弦相似度计算正确
    """

    def __init__(
        self,
        model_name: str,
        device_pref: str = "auto",
        batch_size: int = 8,
        normalize: bool = True,
    ) -> None:
        """初始化BGE-M3嵌入模型

        Args:
            model_name: 模型名称或路径，通常为 "BAAI/bge-m3"
            device_pref: 设备偏好，"auto"/"cpu"/"cuda"/"mps"等
            batch_size: 批处理大小，平衡内存使用和处理速度
            normalize: 是否进行L2归一化，推荐True
        """
        # 自动选择最优计算设备（CUDA优先，然后MPS，最后CPU）
        device = pick_device(device_pref)

        # 初始化BGE-M3模型
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=(device == "cuda"),  # CUDA设备启用FP16加速
            device=device,
        )
        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict[int, float]]]:
        """编码文本为稠密向量和稀疏向量

        将输入文本列表转换为混合向量表示，用于后续的混合检索。
        这是"混合向量存储"阶段的核心计算。

        Args:
            texts: 待编码的文本列表

        Returns:
            Tuple[np.ndarray, List[Dict[int, float]]]:
            - 稠密向量：形状为(len(texts), 1024)的numpy数组，用于语义相似度计算
            - 稀疏向量：长度为len(texts)的字典列表，每个字典为{token_id: weight}格式

        Example:
            >>> embedder = LocalBGEM3("BAAI/bge-m3")
            >>> dense, sparse = embedder(["如何配置服务器IP地址？"])
            >>> dense.shape  # (1, 1024)
            >>> sparse[0]    # {123: 0.5, 456: 0.3, ...}
        """
        # 调用BGE-M3模型进行编码
        outputs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_sparse=True,  # 返回稀疏向量用于关键词匹配
            return_dense=True,   # 返回稠密向量用于语义理解
        )

        # 提取并处理稠密向量
        dense = np.array(outputs["dense_vecs"], dtype=np.float32)
        if self.normalize:
            dense = _l2_normalize(dense)  # L2归一化，确保余弦相似度计算正确

        # 提取稀疏向量（词袋权重）
        sparse = outputs["lexical_weights"]

        return dense, sparse
