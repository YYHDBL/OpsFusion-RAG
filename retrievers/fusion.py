from __future__ import annotations

from typing import Dict, List

from qdrant_client.http import models as rest


def rrf_merge(
    dense_results: List[rest.ScoredPoint],
    sparse_results: List[rest.ScoredPoint],
    topk: int,
    k: int = 60,
) -> List[rest.ScoredPoint]:
    """倒数排序融合算法实现

    实现RRF(Reciprocal Rank Fusion)算法，将稠密向量检索和稀疏向量检索的结果进行融合。
    这是"双路召回"后的结果融合步骤，平衡语义理解和关键词匹配的优势。

    RRF原理：
    1. 对每个检索结果进行排名（rank）
    2. 计算倒数融合分数：score = 1 / (k + rank)
    3. 将同一文档在不同检索路径中的分数相加
    4. 按最终分数重新排序

    优势：
    - 结合语义相似性和关键词精确性
    - 降低单一检索路径的偏差
    - 对排名靠前的结果给予更高权重

    Args:
        dense_results: 稠密向量检索结果，按语义相似度排序
        sparse_results: 稀疏向量检索结果，按关键词匹配度排序
        topk: 最终返回的Top-K结果数量
        k: RRF平滑参数，通常为60，控制排名权重的衰减速度

    Returns:
        List[rest.ScoredPoint]: 融合后重新排序的检索结果列表

    Example:
        >>> dense = [doc1, doc3, doc5]  # 语义检索结果
        >>> sparse = [doc2, doc1, doc4]  # 关键词检索结果
        >>> merged = rrf_merge(dense, sparse, topk=5)
        # doc1在两个路径中都出现，获得更高融合分数

    Note:
        - k值越大，排名权重差异越小
        - 去重处理：同一文档只保留最高分数的版本
        - 自动处理结果ID冲突问题
    """
    scores: Dict[str, float] = {}  # 存储每个文档的融合分数
    payload_map: Dict[str, rest.ScoredPoint] = {}  # 存储文档详细信息

    def add_results(results: List[rest.ScoredPoint]):
        """处理单路检索结果，计算RRF分数

        Args:
            results: 单路检索结果列表，已按相关性排序
        """
        for rank, point in enumerate(results, start=1):
            pid = str(point.id)  # 统一转为字符串作为键
            payload_map[pid] = point  # 保存文档详细信息

            # RRF核心公式：score = 1 / (k + rank)
            # 排名越靠前，分数越高
            reciprocal_score = 1.0 / (k + rank)

            # 累加该文档在各检索路径中的分数
            scores[pid] = scores.get(pid, 0) + reciprocal_score

    # 处理稠密向量检索结果（语义理解）
    add_results(dense_results)

    # 处理稀疏向量检索结果（关键词匹配）
    add_results(sparse_results)

    # 按融合分数降序排序，取Top-K结果
    merged_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:topk]

    # 返回融合后的结果列表
    return [payload_map[i] for i in merged_ids]
