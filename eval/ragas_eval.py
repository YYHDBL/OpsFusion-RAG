from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from qdrant_client import QdrantClient
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_relevance

from langchain_openai import ChatOpenAI

from opsfusion_rag.pipeline.query import rewrite_query_if_needed
from opsfusion_rag.llm.deepseek import generate_answer
from opsfusion_rag.retrievers.embeddings import LocalBGEM3
from opsfusion_rag.retrievers.fusion import rrf_merge
from opsfusion_rag.retrievers.qdrant_store import search_dense, search_sparse
from opsfusion_rag.rerankers.reranker import SiliconReranker
from opsfusion_rag.utils.config import load_config, Config


def load_val(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_clients(cfg: Config):
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    client = QdrantClient(
        host=cfg.qdrant["host"],
        port=cfg.qdrant["port"],
        api_key=cfg.qdrant.get("api_key") or None,
        https=cfg.qdrant.get("https", False),
        timeout=cfg.qdrant.get("timeout", 30),
        prefer_grpc=False,
    )
    embedder = LocalBGEM3(
        model_name=cfg.embedding["model_name"],
        device_pref=cfg.embedding.get("device", "auto"),
        batch_size=cfg.embedding.get("batch_size", 8),
        normalize=cfg.embedding.get("normalize", True),
    )
    reranker = SiliconReranker(
        endpoint=cfg.reranker["endpoint"],
        api_key=cfg.reranker["api_key"],
        model=cfg.reranker["model"],
    )
    llm = ChatOpenAI(
        model=cfg.deepseek["model"],
        base_url=cfg.deepseek["base_url"],
        api_key=cfg.deepseek["api_key"],
        temperature=cfg.deepseek["temperature"],
    )
    return client, embedder, reranker, llm


def run_one(cfg: Config, client, embedder, reranker, llm, query: str):
    # 查询改写
    rewritten = rewrite_query_if_needed(llm, query, cfg.retrieval.get("use_query_rewrite", False))

    # 编码查询
    q_dense, q_sparse = embedder.encode([rewritten])
    q_dense_vec = q_dense[0].tolist()
    q_sparse_vec = q_sparse[0]

    # 双路召回
    dense_hits = search_dense(
        client,
        collection=cfg.qdrant["collection"],
        query=q_dense_vec,
        limit=cfg.retrieval["topk_dense"],
    )
    sparse_hits = search_sparse(
        client,
        collection=cfg.qdrant["collection"],
        query_sparse=q_sparse_vec,
        sparse_name=cfg.qdrant["sparse_name"],
        limit=cfg.retrieval["topk_sparse"],
    )
    fused = rrf_merge(
        dense_hits,
        sparse_hits,
        topk=cfg.retrieval["topk_final"],
    )

    docs_text = [hit.payload["text"] for hit in fused]

    # 重排
    rerank_idx = reranker.rerank(query=rewritten, docs=docs_text, topk=len(docs_text))
    ordered = [docs_text[i] for i in rerank_idx][: cfg.retrieval["topk_final"]]

    # 生成
    context_block = "\n\n".join(
        [f"[{i+1}] {text}" for i, text in enumerate(ordered)]
    )
    answer = generate_answer(
        llm=llm,
        system_prompt=cfg.deepseek["system_prompt"],
        query=query,
        context=context_block,
    )
    return answer, ordered


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for OpsFusion-RAG")
    parser.add_argument("--config", default="opsfusion_rag/configs/config.yaml", help="配置文件路径")
    parser.add_argument("--val", default="src/data/val.json", help="ground truth 文件")
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 条")
    args = parser.parse_args()

    cfg = load_config(args.config)
    val = load_val(Path(args.val))
    if args.limit:
        val = val[: args.limit]

    client, embedder, reranker, llm = build_clients(cfg)

    rows = []
    for i, item in enumerate(val):
        q = item["query"]
        gt = item.get("answer", "")
        answer, ctxs = run_one(cfg, client, embedder, reranker, llm, q)
        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": ctxs,
                "ground_truth": gt,
            }
        )
        print(f"[{i+1}/{len(val)}] done")

    df = pd.DataFrame(rows)

    eval_res = evaluate(
        df,
        metrics=[faithfulness, answer_relevance, context_relevance],
        llm=llm,
    )

    print("RAGAS metrics:")
    for k, v in eval_res.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
