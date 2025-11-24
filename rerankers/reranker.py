from __future__ import annotations

import requests
from typing import List


class SiliconReranker:
    """SiliconFlow重排序服务封装类

    封装基于Cross-Encoder架构的重排序模型，实现对检索结果的精准排序。
    这是RAG流水线中"重排序"阶段的核心组件，用于提高检索精度。

    重排序原理：
    - 使用Cross-Encoder对query和document进行全交互
    - 计算查询-文档对的相关性分数
    - 相比Bi-Encoder，精度更高但计算成本更大

    使用场景：
    - 在RRF融合之后进行精准排序
    - 从Top-50中筛选出最相关的Top-6
    - 作为Top-1证据修正的前置步骤

    Attributes:
        endpoint: SiliconFlow重排序API端点
        api_key: SiliconFlow API密钥
        model: 重排序模型名称，通常为"BAAI/bge-reranker-v2-m3"
    """

    def __init__(self, endpoint: str, api_key: str, model: str) -> None:
        """初始化重排序器

        Args:
            endpoint: SiliconFlow重排序API端点
                      例如: "https://api.siliconflow.cn/v1/rerank"
            api_key: SiliconFlow API密钥，用于身份认证
            model: 重排序模型名称，推荐使用"BAAI/bge-reranker-v2-m3"

        Example:
            >>> reranker = SiliconReranker(
            ...     endpoint="https://api.siliconflow.cn/v1/rerank",
            ...     api_key="your-api-key",
            ...     model="BAAI/bge-reranker-v2-m3"
            ... )
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    def rerank(self, query: str, docs: List[str], topk: int) -> List[int]:
        """执行文档重排序

        对检索到的文档列表进行重排序，返回按相关性从高到低排列的文档索引。
        使用Cross-Encoder模型计算查询与每个文档的精确相关性分数。

        工作流程：
        1. 构建API请求载荷
        2. 发送HTTP POST请求到SiliconFlow
        3. 解析响应数据并按分数排序
        4. 返回Top-K文档的原始索引

        Args:
            query: 用户查询字符串（通常经过改写优化）
            docs: 待重排序的文档列表，来自RRF融合结果
            topk: 返回的Top-K文档数量，通常为6

        Returns:
            List[int]: 按相关性从高到低排列的文档索引列表
                       索引对应于输入docs列表的位置

        Example:
            >>> docs = ["文档1", "文档2", "文档3"]
            >>> indices = reranker.rerank("如何配置服务器", docs, topk=2)
            >>> print(indices)  # [2, 1] 表示文档3最相关，文档2次之

        Raises:
            requests.HTTPError: 当API调用失败时抛出异常
            ValueError: 当响应数据格式不正确时抛出异常

        Note:
            - API调用有30秒超时限制
            - 响应格式: {"data": [{"index": 0, "score": 0.95}, ...]}
            - 模型支持中英文混合文档
            - 计算复杂度：O(len(docs))，适合小规模精准排序
        """
        # 构建API请求载荷
        payload = {
            "model": self.model,      # 指定重排序模型
            "query": query,           # 用户查询
            "documents": docs         # 待排序文档列表
        }

        # 设置请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",  # API密钥认证
            "Content-Type": "application/json",          # JSON格式
        }

        # 发送HTTP POST请求
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=30  # 30秒超时
            )
            resp.raise_for_status()  # 检查HTTP错误
        except requests.RequestException as e:
            raise requests.HTTPError(f"重排序API调用失败: {e}")

        # 解析响应数据
        data = resp.json()

        # SiliconFlow返回格式: {"data": [{"index": 0, "score": 0.95}, ...]}
        items = data.get("data", [])
        if not items:
            raise ValueError("重排序API返回空结果")

        # 按相关性分数降序排序，取Top-K
        items = sorted(items, key=lambda x: x["score"], reverse=True)[:topk]

        # 返回文档索引列表（对应原始docs列表的顺序）
        return [it["index"] for it in items]
