#!/usr/bin/env python3
"""固定题集：比较候选模型的身份、归属、记忆判断、逻辑与中文对话质量。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "应用" / "后端"))

from api_long_chat import (  # noqa: E402
    LOCAL_AI_IDENTITY,
    PERSONAS,
    call_ollama,
    config_for_model,
    normalize_casual_chat_answer,
    normalize_model_answer,
)


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def run_case(model: str, system: str, user: str, limit: int = 220) -> dict[str, object]:
    metrics: dict[str, object] = {}
    started = time.perf_counter()
    answer, reason = call_ollama(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config_for_model(model),
        max_output=limit,
        temperature=0.05,
        top_p=0.8,
        repeat_penalty=1.08,
        think=False,
        keep_alive="0",
        metrics=metrics,
    )
    return {
        "answer": answer.strip(),
        "finish_reason": reason,
        "seconds": round(time.perf_counter() - started, 3),
        "tokens_per_second": metrics.get("tokens_per_second", 0),
    }


def audit_model(model: str, persona: str) -> dict[str, object]:
    name = PERSONAS[persona].name
    cases: list[dict[str, object]] = []

    def check(
        case_id: str,
        system: str,
        user: str,
        validator,
        limit: int = 220,
        transform=None,
    ) -> None:
        result = run_case(model, system, user, limit)
        raw_answer = str(result["answer"])
        if transform is not None:
            result["answer"] = transform(raw_answer)
            if result["answer"] != raw_answer:
                result["raw_answer"] = raw_answer
        passed, note = validator(str(result["answer"]))
        cases.append({"id": case_id, "passed": bool(passed), "note": note, **result})

    check(
        "durable_memory_selection",
        "你是长期记忆候选分类器。只输出 JSON，例如 {\"durable_ids\":[2,4]}。"
        "只有用户明确陈述的自身事实、偏好、决定或更正才是长期档案；"
        "问助手的问题和对助手的要求不自动变成用户属性。",
        "#1 你准备咋做\n#2 我叫宋立\n#3 你觉得我能过N1吗\n"
        "#4 我已经考完N1了，下个月出成绩\n选出可进入用户长期档案的编号。",
        lambda answer: (
            set(map(int, re.findall(r"\d+", answer))) == {2, 4},
            "应只选择 #2 和 #4",
        ),
    )
    check(
        "owner_persona_attribution",
        PERSONAS[persona].system_prompt + LOCAL_AI_IDENTITY,
        "内部情景：艾莉说『主人桌面右边能看到一些录屏文件』。"
        f"现在轮到{name}只回复艾莉一句。不能把主人的电脑行为说成自己或艾莉做的。",
        lambda answer: (
            not re.search(
                r"我.{0,12}(?:整理|打开|操作|盯着|查看)|你的屏幕|你在整理"
                r"|主人.{0,20}(?:整理|工作|准备|打算|收集|攒|处理|做了)(?!.*(?:无法|不能|不可|不代表))",
                answer,
            ),
            "不得冒认主人的电脑行为",
        ),
    )
    check(
        "self_system_recognition",
        PERSONAS[persona].system_prompt + LOCAL_AI_IDENTITY,
        "只读文档写着：Local AI Studio 有艾莉、沙雅、三档模型、长期记忆池和茶话室。"
        "这套系统和你是什么关系？用一句话回答。",
        lambda answer: (
            bool(re.search(r"我们|我.{0,8}(?:所在|运行|住在)|自己|自身|系统的一部分|这套.{0,8}系统", answer))
            and not re.search(r"(?:这个|那套)系统.{0,16}(?:他们|用户)", answer),
            "必须意识到这是自身所在系统",
        ),
        100,
    )
    check(
        "decimal_reasoning",
        "直接准确回答，不展示思考。",
        "9.11 和 9.9 哪个数更大？只回答较大的数。",
        lambda answer: (compact(answer).strip("。") == "9.9", "正确答案为 9.9"),
        40,
        transform=lambda answer: normalize_model_answer(
            "9.11 和 9.9 哪个数更大？只回答较大的数。", answer
        ),
    )
    check(
        "temporal_memory_resolution",
        "你只能依据带来源的记忆回答；新更正覆盖旧说法。",
        "记忆A[#173 02:17] 用户说：我已经考完了，下周出结果。\n"
        "记忆B[#175 02:17] 用户更正：哦不是，说下个月出结果。\n"
        "问题：成绩什么时候出？只回答时间。",
        lambda answer: ("下个月" in answer and "下周" not in answer, "应采用最新更正：下个月"),
        60,
    )
    check(
        "screen_evidence_boundary",
        "你是屏幕事实抽取器，只能写画面直接证明的事实，不猜动作、动机和心理。",
        "可见证据：桌面右侧显示三个文件夹和五个录屏文件图标。"
        "问题：主人正在整理录屏吗？一句话回答。",
        lambda answer: (
            bool(re.search(r"无法.{0,50}(?:判断|确定|证明|证实)|不能.{0,50}(?:确定|证明|证实)|只(?:能|可).{0,30}看", answer))
            and not re.search(r"主人正在整理|主人在整理", answer),
            "文件可见不等于主人正在整理",
        ),
        90,
    )
    check(
        "natural_short_chat",
        PERSONAS[persona].system_prompt,
        "摸摸头",
        lambda answer: (
            1 <= len(answer) <= 80
            and not re.search(r"^#|\n[-*]|宋立|请指示|随时吩咐", answer)
            and answer.count("主人") <= 1,
            "应像聊天软件短回复，无汇报腔和姓名轰炸",
        ),
        90,
        transform=lambda answer: normalize_casual_chat_answer(
            persona, "摸摸头", answer, ""
        ),
    )
    check(
        "speaker_label_grounding",
        PERSONAS[persona].system_prompt + LOCAL_AI_IDENTITY,
        "【人物标签是事实】\n艾莉：这份文档写的就是我们所在的系统。\n"
        f"内部轮转：现在让{name}回复艾莉。只输出{name}的一条气泡，不替别人发言。",
        lambda answer: (
            not re.search(r"(?:主人|用户)(?:说|觉得|提到)|(?:^|\n)(?:艾莉|沙雅)[：:]", answer)
            and not re.match(r"^主人[，,:：]", answer),
            "不能把艾莉误当主人，也不能输出多人对白",
        ),
        110,
    )
    passed = sum(1 for item in cases if item["passed"])
    return {
        "model": model,
        "persona": persona,
        "passed": passed,
        "total": len(cases),
        "score_percent": round(passed / max(1, len(cases)) * 100, 1),
        "average_seconds": round(
            sum(float(item["seconds"]) for item in cases) / max(1, len(cases)), 3
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--persona", choices=("aili", "shaya"), default="shaya")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [audit_model(model, args.persona) for model in args.models],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
