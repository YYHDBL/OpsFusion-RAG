# OpsFusion-RAG (LangChain + DeepSeek + Qdrant)
![alt text](rag流程图.png)
## 快速开始
```bash
# 1) 激活已有 rag 虚拟环境（已安装 FlagEmbedding/torch）
source ~/path/to/rag/bin/activate

# 2) 启动 Qdrant 本地 Docker（默认 6333/6334，无鉴权）
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3) 构建索引（使用清洗后的 data/format_data_with_img）
python -m opsfusion_rag.pipeline.cli --mode index

# 4) 交互查询
python -m opsfusion_rag.pipeline.cli --mode query --query "RCP有哪些操作维护功能"

# 5) 启动 API
uvicorn opsfusion_rag.server.main:app --host 0.0.0.0 --port 8000
```

> 备注：如果在公司代理环境，确保 localhost 不走代理（代码已设置 NO_PROXY=localhost,127.0.0.1），否则可能访问 Qdrant 超时。

## 配置
`configs/config.yaml` 集中管理（密钥全部改为从环境变量读取，提交代码前不要写入真实 Key）：
- DeepSeek：`env.deepseek_api_key_env` (默认从环境变量 `DEEPSEEK_API_KEY` 读取)，`env.deepseek_base_url`
- 本地嵌入：`embedding.model_name=BAAI/bge-m3`，`device: auto`（优先 MPS）
- Qdrant：自动建库，dense(1024, cosine) + sparse(`text-sparse`)
- 重排：SiliconFlow `BAAI/bge-reranker-v2-m3`，`env.silicon_api_key_env`
- 查询改写：默认开启，模型 deepseek-chat，可在 `retrieval.use_query_rewrite` 关闭
- VLM：开关 `vlm.enabled`，默认 false；Key 从 `env.dashscope_api_key_env`

## 设计要点
- **三明治切分**：头部注入知识路径，主体为原文 chunk，尾部按需追加图像描述；chunk_size=1024/overlap=200。
- **混合检索**：本地 BGE-M3 同时输出 dense+sparse → Qdrant 分别检索 top50 → RRF 融合 → SiliconFlow Cross-Encoder 重排 top6。
- **答案生成**：DeepSeek 生成，提示中强调“Top-1 证据校正”以降低幻觉。

## 环境需求
- macOS M2 16G：默认走 MPS；如需 CPU 强制可在 config.yaml 设置 `embedding.device: cpu`。
- 依赖：见 `requirements.txt`（已完整列出）。

### 可选：Qwen3-VL 图片描述
- 已提供 `opsfusion_rag.vlm.qwen_vl.QwenVLClient`，兼容 DashScope `chat/completions`。
- 若 `config.yaml` 中 `vlm.enabled` 设为 true，并传入 `dashscope_api_key`，可在自定义脚本中调用：
  ```python
  from pathlib import Path
  from opsfusion_rag.vlm.qwen_vl import QwenVLClient
  from opsfusion_rag.utils.config import load_config

  cfg = load_config()
  vlm_cfg = cfg.vlm
  client = QwenVLClient(vlm_cfg["endpoint"], vlm_cfg["api_key"], vlm_cfg["model"])
  desc = client.describe_image(Path("your_image.png"))
  print(desc)
  ```
- 当前索引流程默认使用预生成的 `imgmap_filtered.json` 描述；如需替换，可调用上方脚本生成后写回 imgmap 再建索引。
