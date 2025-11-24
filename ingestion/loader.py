from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class SandwichConfig:
    """三明治注入配置类

    用于配置文档切分和图片注入的参数，实现语境感知的文档增强策略。
    这是整个ETL流水线中"语境感知切片与三明治注入"阶段的核心配置。

    Attributes:
        chunk_size: 切片大小，默认1024字符，确保切片包含完整的语义信息
        chunk_overlap: 切片重叠大小，默认200字符，保证上下文连续性
        img_on_demand: 是否启用按需图片挂载，True表示只在有图片引用时才注入图片描述
        img_max_per_chunk: 每个chunk最大图片数量，防止上下文污染
        img_trigger_regex: 图片引用触发正则表达式，用于检测文本中的图片引用
    """
    chunk_size: int
    chunk_overlap: int
    img_on_demand: bool = True
    img_max_per_chunk: int = 2
    img_trigger_regex: str = r"(如图|见下图|参见图|图\s*\d+|图表\s*\d+|figure\s*\d+|fig\.)"


def load_pathmap(path: Path) -> Dict[str, List[str]]:
    """加载逻辑路径映射文件

    这是"逻辑拓扑重构"阶段的核心函数，用于建立物理文件路径到业务知识路径的映射。
    通过这个映射，可以解决不同服务器文档中相同章节名称的语义歧义问题。

    Args:
        path: pathmap.json文件路径，包含物理路径→逻辑路径的映射关系

    Returns:
        Dict[str, List[str]]: 键为相对文件路径，值为逻辑路径列表
        示例: {"node_123.html": ["Director产品", "产品描述", "主要功能", "运维管理"]}

    Raises:
        FileNotFoundError: 当pathmap文件不存在时抛出异常
    """
    if not path.exists():
        raise FileNotFoundError(f"pathmap 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_imgmap(path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """加载图片描述映射文件

    加载经过多模态大模型生成的图片语义描述，用于实现图片的按需挂载。
    这是"多模态信息融合"的关键组件。

    Args:
        path: imgmap.json文件路径，包含图片路径和描述信息

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: 三层嵌套字典结构
        - 第一层键: 文件路径
        - 第二层键: 图片标识（如"图1"、"图2"）
        - 第三层键: 图片元数据（img_path, title, content）

    Example:
        {
            "director/产品描述.txt": {
                "图1": {
                    "img_path": "director/产品描述/topics/images/arch.png",
                    "title": "产品架构图",
                    "content": "这是一个展示产品整体架构的图片..."
                }
            }
        }

    Raises:
        FileNotFoundError: 当imgmap文件不存在时抛出异常
    """
    if not path.exists():
        raise FileNotFoundError(f"imgmap 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_sandwich(
    body: str,
    logical_path: List[str],
    img_refs: Optional[Dict[str, Dict[str, str]]] = None,
    cfg: Optional[SandwichConfig] = None,
) -> str:
    """构建三明治注入文档

    实现"三明治注入策略"，是ETL流水线中的核心增强技术。
    采用"先切分、后注入"的策略，确保切片稳定性与语义完整性。

    三明治结构：
    1. 头部(Head)：注入知识路径，为Embedding模型提供全局导航语义
    2. 中部(Body)：保留原始文档切片
    3. 尾部(Tail)：按需挂载图片描述，实现多模态召回

    Args:
        body: 原始文档切片内容
        logical_path: 逻辑路径列表，如["Director产品", "产品描述", "主要功能"]
        img_refs: 图片引用字典，包含图片描述信息
        cfg: 三明治注入配置，用于控制图片挂载行为

    Returns:
        str: 经过三明治注入的增强文档字符串

    Example:
        输入:
            body = "TECS Director产品在NFV架构中的位置如图1所示"
            logical_path = ["Director产品", "产品描述", "产品定位"]
            img_refs = {"图1": {"content": "这是一个网络虚拟化基础设施的架构图..."}}

        输出:
            "[路径] Director产品 > 产品描述 > 产品定位
            TECS Director产品在NFV架构中的位置如图1所示
            [图像描述] 这是一个网络虚拟化基础设施的架构图..."
    """
    # 构建头部：注入逻辑路径，为嵌入模型提供全局导航语义
    head = " > ".join(logical_path)
    injected = f"[路径] {head}\n{body}"

    # 处理图片引用：实现按需挂载，避免上下文污染
    if img_refs:
        need_imgs = True
        max_imgs = len(img_refs)

        # 如果启用按需挂载，检测文本中是否包含图片引用
        if cfg and cfg.img_on_demand:
            # 编译正则表达式，忽略大小写
            trigger = re.compile(cfg.img_trigger_regex, re.I)
            # 检测正文是否包含图片引用（如"如图1"、"见图2"等）
            need_imgs = bool(trigger.search(body))
            # 限制每个chunk的图片数量
            max_imgs = cfg.img_max_per_chunk

        # 如果需要挂载图片，构建尾部内容
        if need_imgs:
            tail_parts = []
            # 最多取max_imgs个图片描述，按顺序处理
            for _, meta in list(img_refs.items())[:max_imgs]:
                # 优先使用content字段，其次使用text字段
                desc = meta.get("content") or meta.get("text") or ""
                if desc:
                    tail_parts.append(f"[图像描述] {desc}")

            # 将图片描述追加到文档尾部
            if tail_parts:
                injected = injected + "\n" + "\n".join(tail_parts)

    return injected


def iter_raw_files(root: Path) -> Iterable[Path]:
    """遍历原始文档文件

    递归遍历数据目录，查找所有.txt格式的预处理文档。
    这些文档应该已经经过HTML清洗和标准化处理。

    Args:
        root: 数据根目录路径

    Yields:
        Path: .txt文件的路径，用于后续处理
    """
    for path in root.rglob("*.txt"):
        yield path


def load_documents(
    data_root: Path,
    pathmap_path: Path,
    imgmap_path: Path,
    cfg: SandwichConfig,
) -> List[Document]:
    """加载并处理所有文档

    这是ETL流水线的主要入口函数，整合了逻辑拓扑重构、语境感知切片和三明治注入。
    完成从原始预处理文本到可索引文档的完整转换流程。

    处理流程：
    1. 加载逻辑路径映射（解决语义歧义）
    2. 加载图片描述映射（多模态信息）
    3. 鲁棒性文本切分（确保一致性）
    4. 三明治注入（语义增强）
    5. 构建LangChain Document对象

    Args:
        data_root: 预处理数据根目录，包含.txt文件
        pathmap_path: 逻辑路径映射文件路径
        imgmap_path: 图片描述映射文件路径
        cfg: 三明治注入配置参数

    Returns:
        List[Document]: 经过增强处理的文档列表，可直接用于向量化存储

    Example:
        返回的Document结构：
        Document(
            page_content="[路径] Director产品 > 产品描述 > 产品定位\nTECS Director在NFV架构中...\n[图像描述] 架构图描述...",
            metadata={
                "source": "director/产品描述/topics/产品定位.txt",
                "chunk_id": 0,
                "logical_path": ["Director产品", "产品描述", "产品定位"]
            }
        )
    """
    # 加载映射文件
    pathmap = load_pathmap(pathmap_path)  # 逻辑拓扑重构
    imgmap = load_imgmap(imgmap_path)    # 图片描述映射

    # 配置文本切分器：使用鲁棒性切分确保不同环境下结果一致
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,       # 切片大小：1024字符
        chunk_overlap=cfg.chunk_overlap,  # 重叠大小：200字符
        separators=["\n\n", "\n", "。", "；", "，", " "],  # 智能分隔符
    )

    docs: List[Document] = []

    # 遍历所有预处理文档
    for txt_path in iter_raw_files(data_root):
        # 获取相对路径作为文档标识
        key = str(txt_path.relative_to(data_root))

        # 获取逻辑路径和图片信息
        logical_path = pathmap.get(key, [])    # 解决"这是什么"的问题
        images = imgmap.get(key, {})           # 多模态信息

        # 读取原始文本内容
        with txt_path.open("r", encoding="utf-8") as f:
            raw = f.read()

        # 执行鲁棒性切分
        chunks = splitter.split_text(raw)

        # 对每个切片进行三明治注入
        for idx, chunk in enumerate(chunks):
            # 构建增强文档：路径+正文+图片描述
            text = build_sandwich(
                chunk,
                logical_path,
                images if images else None,
                cfg=cfg,
            )

            # 构建元数据
            metadata = {
                "source": key,                    # 原始文件路径
                "chunk_id": idx,                  # 切片ID
                "logical_path": logical_path,     # 逻辑路径
            }

            # 创建LangChain Document对象
            docs.append(Document(page_content=text, metadata=metadata))

    return docs
