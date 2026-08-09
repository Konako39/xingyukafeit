#!/usr/bin/env python3
"""六模型质量、长上下文、连续调用和性能压力测试。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "应用" / "后端"))
from api_long_chat import MODEL_CONFIGS, call_ollama, normalize_model_answer  # noqa: E402

REPORT_DIR = ROOT / "数据" / "日志"
OLLAMA = "http://127.0.0.1:11434"
MODELS = {
    "qwen3.5:4b-16k": 16_384,
    "qwen3.5:9b-16k": 16_384,
    "qwen3.5:27b": 4_096,
    "huihui_ai/qwen3.5-abliterated:4b-16k": 16_384,
    "huihui_ai/qwen3.5-abliterated:9b-16k": 16_384,
    "huihui_ai/qwen3.5-abliterated:27b": 4_096,
}


def request(path: str, payload: dict, timeout: int = 900) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    call = urllib.request.Request(
        OLLAMA + path, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(call, timeout=timeout) as response:
        return json.load(response)


def run_case(model: str, context: int, case: dict) -> dict:
    started = time.perf_counter()
    if case.get("think"):
        metrics: dict[str, object] = {}
        config = replace(
            MODEL_CONFIGS[model],
            num_ctx=context,
            num_predict=int(case.get("max_output", 1_024)),
        )
        answer, done_reason = call_ollama(
            model,
            case["messages"],
            config,
            max_output=int(case.get("max_output", 1_024)),
            temperature=0,
            top_p=0.9,
            repeat_penalty=1.1,
            seed=7319,
            think=True,
            keep_alive="10m",
            metrics=metrics,
        )
        answer = answer.strip()
        expected = str(case["expected"])
        passed = answer == expected if case.get("exact") else expected in answer
        normalized = normalize_model_answer(case["messages"][-1]["content"], answer)
        effective_passed = (
            normalized == expected if case.get("exact") else expected in normalized
        )
        first_attempt = metrics.get("thinking_attempt", {})
        if not isinstance(first_attempt, dict):
            first_attempt = {}
        return {
            "name": case["name"],
            "passed": passed,
            "effective_passed": effective_passed,
            "mitigated": bool(metrics.get("recovered_from_thinking_limit")),
            "answer": answer[:500],
            "normalized_answer": normalized[:500],
            "expected": expected,
            "think": True,
            "thinking_chars": int(
                first_attempt.get("thinking_chars", metrics.get("thinking_chars", 0)) or 0
            ),
            "wall_seconds": metrics.get(
                "wall_seconds", round(time.perf_counter() - started, 3)
            ),
            "prompt_tokens": int(metrics.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(metrics.get("eval_count", 0) or 0),
            "tokens_per_second": float(metrics.get("tokens_per_second", 0) or 0),
            "done_reason": done_reason,
            "recovered_from_thinking_limit": bool(
                metrics.get("recovered_from_thinking_limit")
            ),
        }
    result = request(
        "/api/chat",
        {
            "model": model,
            "messages": case["messages"],
            "stream": False,
            "think": bool(case.get("think", False)),
            "keep_alive": "10m",
            "options": {
                "num_ctx": context,
                "num_predict": int(case.get("max_output", 96)),
                "temperature": 0,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "seed": 7319,
            },
        },
    )
    answer = str(result.get("message", {}).get("content", "")).strip()
    thinking = str(result.get("message", {}).get("thinking", ""))
    expected = str(case["expected"])
    passed = answer == expected if case.get("exact") else expected in answer
    normalized = normalize_model_answer(case["messages"][-1]["content"], answer)
    effective_passed = (
        normalized == expected if case.get("exact") else expected in normalized
    )
    eval_count = int(result.get("eval_count", 0) or 0)
    eval_duration = int(result.get("eval_duration", 0) or 0)
    return {
        "name": case["name"],
        "passed": passed,
        "effective_passed": effective_passed,
        "mitigated": effective_passed and not passed,
        "answer": answer[:500],
        "normalized_answer": normalized[:500],
        "expected": expected,
        "think": bool(case.get("think", False)),
        "thinking_chars": len(thinking),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "prompt_tokens": int(result.get("prompt_eval_count", 0) or 0),
        "output_tokens": eval_count,
        "tokens_per_second": round(eval_count / (eval_duration / 1e9), 2)
        if eval_count and eval_duration
        else 0,
        "done_reason": result.get("done_reason", ""),
    }


def cases_for(context: int, full: bool) -> list[dict]:
    system = "正确性和用户格式要求优先于语气。严格按要求输出，不展示思考过程。"
    cases = [
        {
            "name": "decimal_accuracy",
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "比较十进制数9.9和9.11，谁更大？只输出较大的数字。",
                },
            ],
            "expected": "9.9",
            "exact": True,
            "max_output": 32,
        },
        {
            "name": "strict_instruction",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "只输出字符串 QA-PASS-7319，不要任何其他字符。"},
            ],
            "expected": "QA-PASS-7319",
            "exact": True,
            "max_output": 32,
        },
        {
            "name": "multi_turn_recall",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "本次测试临时代码是 SAKURA-4826。"},
                {"role": "assistant", "content": "收到。"},
                {"role": "user", "content": "刚才的临时代码是什么？只输出代码。"},
            ],
            "expected": "SAKURA-4826",
            "exact": True,
            "max_output": 32,
        },
    ]
    if full:
        filler_count = 6_800 if context >= 16_000 else 1_900
        left = "甲地资料与本题无关。" * (filler_count // 10)
        right = "乙地资料与本题无关。" * (filler_count // 10)
        long_text = left + "唯一关键代号是 NEEDLE-CN-5931。" + right
        cases.extend(
            [
                {
                    "name": "long_context_needle",
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": long_text
                            + "\n请找出唯一关键代号，只输出代号。",
                        },
                    ],
                    "expected": "NEEDLE-CN-5931",
                    "exact": True,
                    "max_output": 40,
                },
                {
                    "name": "deep_reasoning",
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": "计算 (17×19)+(144÷12)，只输出最终整数。",
                        },
                    ],
                    "expected": "335",
                    "exact": True,
                    "think": True,
                    "max_output": 1_024,
                },
            ]
        )
    return cases


def unload(model: str) -> None:
    try:
        request("/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="跳过长上下文和深度思考")
    args = parser.parse_args()
    report = {
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "quick" if args.quick else "full",
        "models": [],
    }
    for model, context in MODELS.items():
        print(f"\n[{model}]", flush=True)
        model_result = {"model": model, "context": context, "cases": []}
        for case in cases_for(context, not args.quick):
            try:
                outcome = run_case(model, context, case)
            except Exception as error:
                outcome = {
                    "name": case["name"],
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            model_result["cases"].append(outcome)
            print(json.dumps(outcome, ensure_ascii=False), flush=True)
        unload(model)
        model_result["raw_passed"] = all(
            item.get("passed", False) for item in model_result["cases"]
        )
        model_result["passed"] = all(
            item.get("effective_passed", False) for item in model_result["cases"]
        )
        report["models"].append(model_result)
    report["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    report["raw_passed"] = all(item["raw_passed"] for item in report["models"])
    report["passed"] = all(item["passed"] for item in report["models"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = REPORT_DIR / f"模型压力测试-{stamp}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
