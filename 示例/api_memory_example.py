#!/usr/bin/env python3
"""不依赖第三方包的长期记忆 API 示例。"""

import json
import urllib.request


URL = "http://127.0.0.1:11435/v1/chat/completions"
MODEL = "huihui_ai/qwen3.5-abliterated:9b-16k"
PERSONA = "aili"


def chat(user_text, conversation_id=None):
    body = {
        "model": MODEL,
        "persona": PERSONA,
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
    }
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    request = urllib.request.Request(
        URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


first = chat("请记住：我喜欢翡翠色。")
conversation_id = first["conversation_id"]
print("第一轮：", first["choices"][0]["message"]["content"])
print("会话 ID：", conversation_id)

second = chat("我喜欢什么颜色？", conversation_id)
print("第二轮：", second["choices"][0]["message"]["content"])

# 不传 conversation_id，会新建一个空白上下文；艾莉的共享记忆和历史语义检索仍有效。
new_session = chat("这是新对话。你还记得我喜欢什么颜色吗？")
print("跨会话：", new_session["choices"][0]["message"]["content"])
