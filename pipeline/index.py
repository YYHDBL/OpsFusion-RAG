from __future__ import annotations

from pathlib import Path
from typing import List

import os
from qdrant_client import QdrantClient

from opsfusion_rag.ingestion.loader import SandwichConfig, load_documents
from opsfusion_rag.retrievers.embeddings import LocalBGEM3
from opsfusion_rag.retrievers.qdrant_store import ensure_collection, upsert_points
from opsfusion_rag.utils.config import Config


def build_index(cfg: Config) -> None:
    """构建向量索引的完整流水线

    这是EasyRAG系统的核心索引构建函数，整合了从预处理文档到向量数据库的完整ETL流水线。
    实现了异构数据清洗、逻辑拓扑重构、语境感知切片、三明治注入和混合向量存储的全流程。

    流水线步骤：
    1. 初始化Qdrant客户端和集合
    2. 加载预处理文档并执行三明治注入
    3. 使用BGE-M3生成混合向量（稠密+稀疏）
    4. 批量存储到向量数据库

    Args:
        cfg: 系统配置对象，包含所有必要的参数配置

    Example:
        >>> from easyrag_langchain.utils.config import Config
        >>> cfg = Config.from_file("config.yaml")
        >>> build_index(cfg)
        # 输出：已完成入库：1234 条 chunk

    Note:
        - 处理时间取决于文档数量和硬件性能
        - 建议在GPU环境下运行以加速向量计算
        - 索引构建完成后即可进行查询操作
    """
    # 设置数据目录路径
    data_root = Path(cfg.paths["data_root"])

    # 配置本地网络环境，避免Qdrant连接被代理拦截
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    # 初始化Qdrant向量数据库客户端
    client = QdrantClient(
        host=cfg.qdrant["host"],
        port=cfg.qdrant["port"],
        api_key=cfg.qdrant.get("api_key") or None,
        https=cfg.qdrant.get("https", False),
        timeout=cfg.qdrant.get("timeout", 30),
        prefer_grpc=False,  # 使用HTTP接口提高兼容性
    )

    # 确保目标集合存在，如不存在则创建
    # 初始化混合向量存储结构：稠密向量(HNSW) + 稀疏向量(倒排索引)
    ensure_collection(
        client=client,
        name=cfg.qdrant["collection"],
        vector_size=cfg.qdrant["vector_size"],  # BGE-M3: 1024维
        sparse_name=cfg.qdrant["sparse_name"],  # "text-sparse"
        distance=cfg.qdrant["distance"],  # "Cosine"
    )

    # 步骤1-2: 加载文档并执行三明治注入
    # 整合逻辑拓扑重构、语境感知切片和按需图片挂载
    docs = load_documents(
        data_root=data_root,
        pathmap_path=Path(cfg.paths["pathmap"]),  # 物理路径→逻辑路径映射
        imgmap_path=Path(cfg.paths["imgmap"]),    # 图片描述映射
        cfg=SandwichConfig(
            chunk_size=cfg.retrieval["chunk_size"],      # 1024字符
            chunk_overlap=cfg.retrieval["chunk_overlap"], # 200字符
            img_on_demand=cfg.retrieval["images"].get("enable_on_demand", True),
            img_trigger_regex=cfg.retrieval["images"].get(
                "trigger_regex", r"(如图|见下图|参见图|图\s*\d+|图表\s*\d+|figure\s*\d+|fig\.)"
            ),
            img_max_per_chunk=cfg.retrieval["images"].get("max_per_chunk", 2),  # 限制图片数量
        ),
    )

    # 打印前4个chunk供快速查看
    print("示例 chunk（前4个）:")
    for i, d in enumerate(docs[:4]):
        preview = d.page_content[:300].replace("\n", "\\n")
        print(
            f"[{i}] source={d.metadata.get('source')} "
            f"chunk_id={d.metadata.get('chunk_id')} "
            f"logical_path={' > '.join(d.metadata.get('logical_path', []))} "
            f"content={preview}"
        )

    # 步骤3: 生成混合向量
    # 使用BGE-M3同时生成稠密向量(语义理解)和稀疏向量(关键词匹配)
    embedder = LocalBGEM3(
        model_name=cfg.embedding["model_name"],  # "BAAI/bge-m3"
        device_pref=cfg.embedding.get("device", "auto"),  # 自动选择最优设备
        batch_size=cfg.embedding.get("batch_size", 8),    # 批处理大小
        normalize=cfg.embedding.get("normalize", True),   # L2归一化
    )

    # 编码所有文档为混合向量表示
    print(f"正在编码 {len(docs)} 个文档片段...")
    dense, sparse = embedder.encode([d.page_content for d in docs])

    # 构建载荷数据：包含原始文本和元数据
    payloads: List[dict] = []
    for doc in docs:
        payloads.append(
            {
                "text": doc.page_content,  # 经过三明治注入的增强文本
                **doc.metadata,            # source, chunk_id, logical_path等
            }
        )

    # 步骤4: 批量存储到向量数据库
    # 完成混合向量存储，支持后续的双路召回检索
    print("正在存储向量到数据库...")
    upsert_points(
        client=client,
        collection=cfg.qdrant["collection"],
        dense=dense.tolist(),     # 稠密向量列表
        sparse_list=sparse,       # 稀疏向量列表
        payloads=payloads,        # 载荷数据
        sparse_name=cfg.qdrant["sparse_name"],
    )

    print(f"已完成入库：{len(docs)} 条 chunk")
