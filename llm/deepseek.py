from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def build_deepseek(model: str, base_url: str, api_key: str, temperature: float):
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


def generate_answer(llm: ChatOpenAI, system_prompt: str, query: str, context: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"用户问题：{query}\n\n检索到的证据（已按相关性排序，第一条最可信）：\n{context}\n\n请基于证据作答，并使用第一条证据进行校正。"
        ),
    ]
    resp = llm.invoke(messages)
    return resp.content
