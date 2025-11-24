from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from opsfusion_rag.llm.deepseek import generate_answer
from opsfusion_rag.retrievers.embeddings import LocalBGEM3
from opsfusion_rag.retrievers.fusion import rrf_merge
from opsfusion_rag.retrievers.qdrant_store import search_dense, search_sparse
from opsfusion_rag.rerankers.reranker import SiliconReranker
from opsfusion_rag.utils.config import Config


def rewrite_query_if_needed(llm: ChatOpenAI, query: str, enabled: bool) -> str:
    """查询改写函数（可选）

    使用LLM将用户原始查询改写为更适合向量检索的表述。
    这是查询优化的第一步，旨在提高检索召回率。

    改写目标：
    - 明确查询意图
    - 补充关键上下文
    - 使用更规范的术语表达
    - 保持原意不变

    Args:
        llm: 用于查询改写的大语言模型
        query: 用户原始查询
        enabled: 是否启用查询改写功能

    Returns:
        str: 改写后的查询，如果未启用则返回原查询

    Example:
        原查询: "怎么配IP"
        改写后: "如何在服务器上配置IP地址参数"
    """
    if not enabled:
        return query

    # 使用专门的提示词引导LLM进行查询改写
    prompt = f"请将以下运维问题改写为更检索友好的表述，保持语义不变：{query}"
    return llm.invoke(prompt).content


def retrieve_and_answer(cfg: Config, query: str) -> str:
    """完整的检索增强生成(RAG)流程

    实现从用户查询到最终答案的完整RAG流水线，整合了查询改写、双路召回、
    RRF融合、重排序和Top-1证据修正等高级技术。

    RAG流程步骤：
    1. 查询预处理：改写查询以提高召回率
    2. 双路召回：稠密向量(语义) + 稀疏向量(关键词)
    3. RRF融合：合并双路结果，平衡语义和精确性
    4. 重排序：使用Cross-Encoder进行精准排序
    5. 答案生成：基于检索到的上下文生成答案
    6. Top-1修正：以最高置信度文档为基准进行纠错

    Args:
        cfg: 系统配置对象
        query: 用户查询字符串

    Returns:
        str: 基于检索结果的增强答案

    Example:
        >>> answer = retrieve_and_answer(cfg, "如何配置Director产品?")
        >>> print(answer)
        # 返回基于文档的准确答案，包含具体的配置步骤和参数
    """
    # ========== 初始化所有组件 ==========

    # 嵌入模型：用于查询向量化
    embedder = LocalBGEM3(
        model_name=cfg.embedding["model_name"],  # "BAAI/bge-m3"
        device_pref=cfg.embedding.get("device", "auto"),
        batch_size=cfg.embedding.get("batch_size", 8),
        normalize=cfg.embedding.get("normalize", True),
    )

    # Qdrant客户端：用于向量检索
    client = QdrantClient(
        host=cfg.qdrant["host"],
        port=cfg.qdrant["port"],
        api_key=cfg.qdrant.get("api_key") or None,
        https=cfg.qdrant.get("https", False),
    )

    # 重排序模型：用于精准排序
    reranker = SiliconReranker(
        endpoint=cfg.reranker["endpoint"],  # SiliconFlow API
        api_key=cfg.reranker["api_key"],
        model=cfg.reranker["model"],         # "BAAI/bge-reranker-v2-m3"
    )

    # 生成模型：用于答案生成
    deepseek = ChatOpenAI(
        model=cfg.deepseek["model"],        # "deepseek-chat"
        base_url=cfg.deepseek["base_url"],  # DeepSeek API地址
        api_key=cfg.deepseek["api_key"],
        temperature=cfg.deepseek["temperature"],  # 控制生成随机性
    )

    # ========== 步骤1: 查询改写（可选）==========
    # 将口语化查询改写为更适合检索的表述
    rewritten = rewrite_query_if_needed(
        deepseek, query, cfg.retrieval.get("use_query_rewrite", False)
    )

    # ========== 步骤2: 查询向量化==========
    # 将改写后的查询转换为混合向量表示
    q_dense, q_sparse = embedder.encode([rewritten])
    q_dense_vec = q_dense[0].tolist()  # 稠密向量：1024维
    q_sparse_vec = q_sparse[0]         # 稀疏向量：词袋权重

    # ========== 步骤3: 双路召回检索==========

    # 路径1: 稠密向量语义检索（Top-50）
    # 使用HNSW索引进行高效语义相似度搜索
    dense_hits = search_dense(
        client,
        collection=cfg.qdrant["collection"],
        query=q_dense_vec,
        limit=cfg.retrieval["topk_dense"],  # 通常为50
    )

    # 路径2: 稀疏向量关键词检索（Top-50）
    # 使用倒排索引进行精确关键词匹配
    sparse_hits = search_sparse(
        client,
        collection=cfg.qdrant["collection"],
        query_sparse=q_sparse_vec,
        sparse_name=cfg.qdrant["sparse_name"],
        limit=cfg.retrieval["topk_sparse"],  # 通常为50
    )

    # ========== 步骤4: RRF融合==========
    # 使用倒数排序融合算法合并双路结果
    # 平衡语义理解和关键词精确性
    fused = rrf_merge(
        dense_hits,           # 语义检索结果
        sparse_hits,          # 关键词检索结果
        topk=cfg.retrieval["topk_final"],  # 最终保留数量，通常为6
    )

    # ========== 步骤5: 重排序==========
    # 使用Cross-Encoder对融合结果进行精准重排序
    docs_text = [hit.payload["text"] for hit in fused]
    rerank_idx = reranker.rerank(
        query=rewritten,      # 使用改写后的查询
        docs=docs_text,       # 待排序文档
        topk=len(docs_text)   # 对所有文档重新排序
    )
    ordered = [docs_text[i] for i in rerank_idx]

    # ========== 步骤6: 答案生成与Top-1修正==========
    # 构建上下文块，使用编号标记便于引用
    context_block = "\n\n".join(
        [f"[{i+1}] {text}" for i, text in enumerate(ordered[: cfg.retrieval["topk_final"]])]
    )

    # 基于检索上下文生成答案
    # system_prompt中包含Top-1证据修正指令
    answer = generate_answer(
        llm=deepseek,
        system_prompt=cfg.deepseek["system_prompt"],  # 包含"优先使用Top-1证据"指令
        query=query,           # 原始用户查询
        context=context_block, # 检索到的上下文
    )

    return answer
