from __future__ import annotations

import argparse

from easyrag_langchain.pipeline.index import build_index
from easyrag_langchain.pipeline.query import retrieve_and_answer
from easyrag_langchain.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="EasyRAG LangChain Pipeline")
    parser.add_argument(
        "--mode",
        choices=["index", "query", "vlm"],
        required=True,
        help="index: 构建索引；query: 交互查询",
    )
    parser.add_argument("--q", "--query", dest="query", help="查询文本")
    parser.add_argument(
        "--config",
        default="opsfusion_rag/configs/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "index":
        build_index(cfg)
    elif args.mode == "query":
        if not args.query:
            raise SystemExit("缺少 --query 文本")
        answer = retrieve_and_answer(cfg, args.query)
        print(answer)
    elif args.mode == "vlm":
        # 占位：只打印配置，提醒使用 qwen_vl 客户端
        print(
            "VLM 已配置但默认不开启自动调用。\n"
            "如需生成图像描述，请在代码中使用 easyrag_langchain.vlm.qwen_vl.QwenVLClient。"
        )


if __name__ == "__main__":
    main()
