"""MiniMax-M2.7 + `tool.py` 工具 + ReAct 文本协议（任务由运行时输入）。"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tool import ALL_TOOLS

REACT_SYSTEM = """按 **ReAct** 行动。可用工具（名称须完全一致）：
{tools_block}

格式（二选一）：
Thought: …
Action: <工具名>
Action Input: <JSON 对象，单行；无参写 {{}}>

或：
Thought: …
Final Answer: <给用户的完整回复>

一次只写一个 Action；完成后用 Final Answer 结束。"""


def _text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            x if isinstance(x, str) else str((x or {}).get("text", "")) for x in c
        )
    return str(c)


def _action_json(text: str) -> str | None:
    """取 `Action Input:` 后第一个配平的 `{{...}}`。"""
    i = text.find("Action Input:")
    if i < 0:
        return None
    j = text.find("{", i + 13)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            d += 1
        elif text[k] == "}":
            d -= 1
            if d == 0:
                return text[j : k + 1]
    return None


def _parse(text: str) -> dict[str, Any]:
    if "Final Answer:" in text:
        return {"k": "f", "a": text.split("Final Answer:", 1)[-1].strip()}
    ma = re.search(r"Action:\s*([A-Za-z0-9_]+)", text)
    js = _action_json(text)
    if not ma or not js:
        return {"k": "e", "a": "缺少 Action / Action Input（JSON 对象）。"}
    try:
        obj = json.loads(js)
    except json.JSONDecodeError as e:
        return {"k": "e", "a": f"JSON 错误: {e}"}
    if not isinstance(obj, dict):
        return {"k": "e", "a": "Action Input 须为 JSON 对象。"}
    return {"k": "x", "n": ma.group(1), "o": obj}


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("MINIMAX_API_KEY", "").strip()
    if not key:
        print("缺少 MINIMAX_API_KEY（写在 `.env` 并保存）。", file=sys.stderr)
        sys.exit(1)

    llm = ChatOpenAI(
        model=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7").strip(),
        api_key=key,
        base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1").strip(),
        temperature=float(os.getenv("MINIMAX_TEMPERATURE", "0.2")),
    )
    tools = "\n".join(
        f"- `{t.name}`：{(t.description or '').replace(chr(10), ' ')}"
        for t in ALL_TOOLS
    )
    tmap = {t.name: t for t in ALL_TOOLS}
    sys_msg = REACT_SYSTEM.format(tools_block=tools)

    print("任务（quit 退出）：")
    while True:
        try:
            u = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u or u.lower() in ("q", "quit", "exit"):
            break

        msgs: list[BaseMessage] = [SystemMessage(sys_msg), HumanMessage(u)]
        for _ in range(48):
            r = _text(llm.invoke(msgs))
            msgs.append(AIMessage(r))
            p = _parse(r)
            if p["k"] == "f":
                print(p["a"])
                break
            if p["k"] == "e":
                msgs.append(HumanMessage("Observation: " + p["a"]))
                continue
            nm, pl = p["n"], p["o"]
            if nm not in tmap:
                o = f"未知工具 `{nm}`。"
            else:
                try:
                    out = tmap[nm].invoke(pl)
                    o = out if isinstance(out, str) else str(out)
                except Exception as e:  # noqa: BLE001
                    o = str(e)
            msgs.append(HumanMessage("Observation: " + o))
        else:
            print("超过 48 步仍未 Final Answer。")


if __name__ == "__main__":
    main()
