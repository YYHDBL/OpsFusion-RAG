from __future__ import annotations

from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    sparse_name: str,
    distance: str = "Cosine",
) -> None:
    """确保Qdrant集合存在，如不存在则创建

    创建支持混合向量检索的Qdrant集合，同时包含稠密向量和稀疏向量索引。
    这是"混合向量存储"阶段的数据库初始化步骤。

    集合结构：
    - 稠密向量：1024维，使用HNSW索引，适合语义相似度检索
    - 稀疏向量：词袋权重，使用倒排索引，适合关键词精确匹配
    - 距离度量：余弦相似度，适合归一化向量

    Args:
        client: Qdrant客户端实例
        name: 集合名称，如"easyrag_hybrid"
        vector_size: 稠密向量维度，BGE-M3为1024
        sparse_name: 稀疏向量名称，通常为"text-sparse"
        distance: 距离度量类型，推荐"Cosine"

    Note:
        如果集合已存在，函数直接返回，不会重复创建
    """
    # 检查现有集合
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return

    # 创建混合向量集合
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": rest.VectorParams(size=vector_size, distance=distance),
        },
        sparse_vectors_config={
            sparse_name: rest.SparseVectorParams(),
        },
    )


def upsert_points(
    client: QdrantClient,
    collection: str,
    dense: List[List[float]],
    sparse_list: List[Dict[int, float]],
    payloads: List[Dict],
    sparse_name: str = "text-sparse",
) -> None:
    """批量插入向量点到Qdrant集合

    将经过三明治注入处理的文档向量化后存储到Qdrant数据库。
    这是"混合向量存储"阶段的数据入库步骤。

    数据结构：
    - 稠密向量：用于语义检索的1024维向量
    - 稀疏向量：用于关键词检索的词袋权重
    - 载荷数据：包含原始文本、元数据等信息

    Args:
        client: Qdrant客户端实例
        collection: 目标集合名称
        dense: 稠密向量列表，每个向量为1024维float数组
        sparse_list: 稀疏向量列表，每个为{token_id: weight}字典
        payloads: 载荷数据列表，包含text、source、chunk_id等元数据
        sparse_name: 稀疏向量字段名称

    Example:
        >>> dense, sparse = embedder.encode(["增强后的文档文本"])
        >>> payloads = [{"text": "增强后的文档文本", "source": "path/to/file.txt"}]
        >>> upsert_points(client, "collection", dense.tolist(), [sparse], payloads)
    """
    points = []
    for idx, (d, s, p) in enumerate(zip(dense, sparse_list, payloads)):
        # 稠密向量转为原生 float
        dense_vec = [float(x) for x in d]
        # 稀疏向量需使用 SparseVector(indices, values)
        if s:
            items = sorted(s.items())
            sparse_vec = rest.SparseVector(
                indices=[int(k) for k, _ in items],
                values=[float(v) for _, v in items],
            )
        else:
            sparse_vec = rest.SparseVector(indices=[], values=[])
        points.append(
            rest.PointStruct(
                id=None,  # 自动生成ID
                vector={"dense": dense_vec, sparse_name: sparse_vec},  # 混合向量
                payload=p,  # 载荷数据
            )
        )

    # 批量插入到Qdrant
    client.upsert(collection_name=collection, points=points)


def search_dense(
    client: QdrantClient,
    collection: str,
    query: List[float],
    limit: int,
) -> List[rest.ScoredPoint]:
    """稠密向量相似度搜索

    使用HNSW索引进行高效的语义相似度搜索。
    这是"双路召回"中的稠密向量检索路径。

    Args:
        client: Qdrant客户端实例
        collection: 搜索的目标集合
        query: 查询的稠密向量，1024维
        limit: 返回结果数量限制

    Returns:
        List[rest.ScoredPoint]: 按相似度排序的检索结果列表
        每个结果包含分数、载荷数据和向量ID

    Note:
        使用余弦相似度计算，分数越高表示语义越相关
    """
    return client.search(
        collection_name=collection,
        query_vector=("dense", query),  # 指定稠密向量字段
        limit=limit,
        with_payload=True,  # 返回载荷数据
    )


def search_sparse(
    client: QdrantClient,
    collection: str,
    query_sparse: Dict[int, float],
    sparse_name: str,
    limit: int,
) -> List[rest.ScoredPoint]:
    """稀疏向量关键词搜索

    使用倒排索引进行精确的关键词权重匹配。
    这是"双路召回"中的稀疏向量检索路径。

    Args:
        client: Qdrant客户端实例
        collection: 搜索的目标集合
        query_sparse: 查询的稀疏向量，{token_id: weight}格式
        sparse_name: 稀疏向量字段名称
        limit: 返回结果数量限制

    Returns:
        List[rest.ScoredPoint]: 按点积分排序的检索结果列表
        每个结果包含分数、载荷数据和向量ID

    Note:
        使用点积计算相似度，适合关键词精确匹配
        可以有效处理专业术语、产品名称等精确查询
    """
    items = sorted(query_sparse.items())
    sparse_vec = rest.SparseVector(
        indices=[int(k) for k, _ in items],
        values=[float(v) for _, v in items],
    )
    return client.search(
        collection_name=collection,
        query_vector=rest.NamedSparseVector(name=sparse_name, vector=sparse_vec),
        limit=limit,
        with_payload=True,  # 返回载荷数据
    )
