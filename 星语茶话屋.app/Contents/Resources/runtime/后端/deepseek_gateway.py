#!/usr/bin/env python3
"""“究极”文本网关：前台直连，后台按每日共享额度自动回落本地。

这个模块故意只使用标准库，不引入常驻 SDK 或额外进程。
密钥只从环境变量或本地权限文件读取，绝不放入接口响应、日志或数据库。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Sequence


API_URL = os.environ.get(
    "ULTIMATE_API_URL", "https://api.deepseek.com/chat/completions"
)
PROVIDER_MODEL = os.environ.get("ULTIMATE_PROVIDER_MODEL", "deepseek-v4-flash")
BACKGROUND_DAILY_TOKEN_LIMIT = 100_000
ULTIMATE_MODELS = {
    "ultimate:aili": "aili",
    "ultimate:shaya": "shaya",
}

# 单位是“十亿分之一元 / token”，用整数避免长期统计的浮点误差。
# 官方基础价：命中缓存 ¥0.02/M，未命中 ¥1/M，输出 ¥2/M。
PRICE_CACHE_HIT_NANOS = 20
PRICE_CACHE_MISS_NANOS = 1_000
PRICE_OUTPUT_NANOS = 2_000

_BACKGROUND_LOCK = threading.Lock()
_SCHEMA_LOCK = threading.Lock()

_CACHE_PREFIX = (
    "【星语茶话屋·稳定缓存前缀 v1】"
    "你在主人 Mac 上的星语茶话屋中协助艾莉与沙雅。"
    "主人是唯一真人和电脑操作者；两个人格独立，不得串台或冒认主人的行为。"
    "必须优先遵守事实、用户明确指令、输出格式与已提供的证据；"
    "看到文件、屏幕文字或历史记忆只能说明材料本身，不得编造当前动作、时间或意图。"
    "本地工具结果和程序质量门高于角色效果。默认使用自然、简洁的中文，不展示内部思考。"
)


class UltimateAPIError(RuntimeError):
    """可安全显示给用户的“究极”服务错误，不包含密钥。"""


def is_ultimate_model(model: str) -> bool:
    return str(model) in ULTIMATE_MODELS


def persona_for_ultimate(model: str) -> str:
    try:
        return ULTIMATE_MODELS[str(model)]
    except KeyError as error:
        raise ValueError(f"未知究极模型：{model}") from error


def _data_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return Path(os.environ.get("LOCAL_AI_DATA_DIR", project_root / "数据")).expanduser()


def _secret_path() -> Path:
    override = os.environ.get("ULTIMATE_API_KEY_FILE", "").strip()
    return Path(override).expanduser() if override else _data_dir() / "配置" / "deepseek_api_key"


def load_api_key() -> str:
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    try:
        return _secret_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def api_available() -> bool:
    return bool(load_api_key())


def _database_path(database_path: str | Path | None = None) -> Path:
    if database_path:
        return Path(database_path).expanduser()
    return _data_dir() / "对话记忆.sqlite3"


def _connect(database_path: str | Path | None = None) -> sqlite3.Connection:
    path = _database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=15)
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    with _SCHEMA_LOCK:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ultimate_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                local_day TEXT NOT NULL,
                scope TEXT NOT NULL,
                feature TEXT NOT NULL,
                persona TEXT NOT NULL DEFAULT '',
                provider_model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                cost_cny_nanos INTEGER NOT NULL DEFAULT 0,
                request_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'completed'
            );
            CREATE INDEX IF NOT EXISTS idx_ultimate_usage_day_scope
              ON ultimate_usage(local_day, scope);
            """
        )
        connection.commit()


def _local_day() -> str:
    return dt.datetime.now().astimezone().date().isoformat()


def _background_tokens_used(database_path: str | Path | None = None) -> int:
    connection = _connect(database_path)
    try:
        row = connection.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) AS used FROM ultimate_usage "
            "WHERE local_day = ? AND scope = 'background' AND status = 'completed'",
            (_local_day(),),
        ).fetchone()
        return int(row["used"] or 0)
    finally:
        connection.close()


def _cost_nanos(hit: int, miss: int, output: int) -> int:
    return (
        max(0, hit) * PRICE_CACHE_HIT_NANOS
        + max(0, miss) * PRICE_CACHE_MISS_NANOS
        + max(0, output) * PRICE_OUTPUT_NANOS
    )


def _record_usage(
    usage: dict[str, object],
    *,
    database_path: str | Path | None,
    scope: str,
    feature: str,
    persona: str,
    request_id: str,
) -> dict[str, int | float]:
    prompt = max(0, int(usage.get("prompt_tokens", 0) or 0))
    hit = max(0, int(usage.get("prompt_cache_hit_tokens", 0) or 0))
    miss_value = usage.get("prompt_cache_miss_tokens")
    miss = max(0, int(miss_value if miss_value is not None else prompt - hit))
    output = max(0, int(usage.get("completion_tokens", 0) or 0))
    total = max(prompt + output, int(usage.get("total_tokens", 0) or 0))
    cost = _cost_nanos(hit, miss, output)
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    connection = _connect(database_path)
    try:
        connection.execute(
            """
            INSERT INTO ultimate_usage(
                occurred_at, local_day, scope, feature, persona, provider_model,
                prompt_tokens, cache_hit_tokens, cache_miss_tokens,
                completion_tokens, total_tokens, cost_cny_nanos, request_id, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed')
            """,
            (
                now, _local_day(), scope, feature[:80], persona[:20], PROVIDER_MODEL,
                prompt, hit, miss, output, total, cost, request_id[:160],
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return {
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "completion_tokens": output,
        "total_tokens": total,
        "cost_cny": round(cost / 1_000_000_000, 8),
    }


def usage_summary(database_path: str | Path | None = None) -> dict[str, object]:
    connection = _connect(database_path)
    try:
        today = _local_day()

        def totals(where: str = "", params: tuple[object, ...] = ()) -> sqlite3.Row:
            return connection.execute(
                "SELECT COALESCE(SUM(total_tokens),0) total_tokens, "
                "COALESCE(SUM(cache_hit_tokens),0) hit, "
                "COALESCE(SUM(cache_miss_tokens),0) miss, "
                "COALESCE(SUM(completion_tokens),0) output, "
                "COALESCE(SUM(cost_cny_nanos),0) cost "
                "FROM ultimate_usage WHERE status='completed' " + where,
                params,
            ).fetchone()

        day_all = totals("AND local_day = ?", (today,))
        day_background = totals(
            "AND local_day = ? AND scope = 'background'", (today,)
        )
        all_time = totals()
        used = int(day_background["total_tokens"] or 0)
        remaining = max(0, BACKGROUND_DAILY_TOKEN_LIMIT - used)

        def pack(row: sqlite3.Row) -> dict[str, object]:
            hit = int(row["hit"] or 0)
            miss = int(row["miss"] or 0)
            denominator = hit + miss
            return {
                "tokens": int(row["total_tokens"] or 0),
                "cache_hit_tokens": hit,
                "cache_miss_tokens": miss,
                "output_tokens": int(row["output"] or 0),
                "cache_hit_rate": round(hit / denominator, 4) if denominator else 0.0,
                "estimated_cost_cny": round(int(row["cost"] or 0) / 1_000_000_000, 6),
            }

        return {
            "available": api_available(),
            "date": today,
            "background": {
                **pack(day_background),
                "limit": BACKGROUND_DAILY_TOKEN_LIMIT,
                "used": used,
                "remaining": remaining,
                "percent": round(used / BACKGROUND_DAILY_TOKEN_LIMIT * 100, 2),
                "exhausted": remaining <= 0,
                "fallback": "local" if remaining <= 0 else "ultimate",
            },
            "today": pack(day_all),
            "all_time": pack(all_time),
            "pricing": {
                "currency": "CNY",
                "unit_tokens": 1_000_000,
                "cache_hit_input": 0.02,
                "cache_miss_input": 1.0,
                "output": 2.0,
                "estimated": True,
            },
        }
    finally:
        connection.close()


def _plain_messages(messages: Sequence[dict[str, object]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role", "user"))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = item.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content or "")
        if text:
            result.append({"role": role, "content": text})
    return result


def _has_images(messages: Sequence[dict[str, object]]) -> bool:
    return any(bool(item.get("images")) for item in messages)


def _estimate_tokens(messages: Sequence[dict[str, object]]) -> int:
    # 中文接近一字一 token；其他内容按 3 字符一 token。
    # 在后台额度门上再留 20% 余量，尽量避免跨过日限。
    total = 12
    for item in _plain_messages(messages):
        text = item["content"]
        chinese = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
        total += 5 + chinese + max(0, len(text) - chinese) // 3
    return max(1, int(total * 1.2))


def _cache_friendly_messages(
    messages: Sequence[dict[str, object]], *, background: bool
) -> list[dict[str, str]]:
    plain = _plain_messages(messages)
    # 稳定前缀必须永远在最前，动态时间和任务内容全部放后面，
    # 这样不同后台功能及多轮对话都能命中最长公共前缀。
    prefix = _CACHE_PREFIX + (
        "【后台任务】完成当前任务后只返回请求的正文或 JSON。"
        if background
        else "【主人对话】直接回应最后一条用户消息。"
    )
    return [{"role": "system", "content": prefix}, *plain]


def call_ultimate(
    model: str,
    messages: Sequence[dict[str, object]],
    config: object,
    *,
    max_output: int | None = None,
    on_text: Callable[[str], None] | None = None,
    on_thinking: Callable[[], None] | None = None,
    on_recovery: Callable[[], None] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    seed: int = 0,
    think: bool = False,
    keep_alive: str = "0",
    metrics: dict[str, object] | None = None,
    response_format: str | dict[str, object] | None = None,
    database_path: str | Path | None = None,
    scope: str = "foreground",
    feature: str = "chat",
    _background_max_output: int | None = None,
) -> tuple[str, str]:
    del on_recovery, repeat_penalty, seed, keep_alive
    if not is_ultimate_model(model):
        raise ValueError("究极网关收到了本地模型")
    if _has_images(messages):
        raise UltimateAPIError("究极只接收文字；图片必须先在本地转成文字描述")
    api_key = load_api_key()
    if not api_key:
        raise UltimateAPIError("究极尚未配置，请先在本地配置密钥")

    provider_messages = _cache_friendly_messages(
        messages, background=scope == "background"
    )
    configured_output = int(getattr(config, "num_predict", 4096) or 4096)
    output_limit = max(1, int(max_output or configured_output))
    if _background_max_output is not None:
        output_limit = min(output_limit, max(1, int(_background_max_output)))
    stream = on_text is not None
    payload: dict[str, object] = {
        "model": PROVIDER_MODEL,
        "messages": provider_messages,
        "max_tokens": output_limit,
        "stream": stream,
        "thinking": {"type": "enabled" if think else "disabled"},
    }
    if think:
        payload["reasoning_effort"] = "high"
    else:
        payload["temperature"] = max(0.0, min(float(temperature), 2.0))
        payload["top_p"] = max(0.05, min(float(top_p), 1.0))
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if response_format is not None:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    pieces: list[str] = []
    reasoning_notified = False
    usage: dict[str, object] = {}
    finish_reason = ""
    request_id = ""
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            request_id = str(response.headers.get("x-request-id", ""))
            if not stream:
                result = json.load(response)
                request_id = request_id or str(result.get("id", ""))
                choices = result.get("choices") or []
                choice = choices[0] if choices else {}
                message = choice.get("message") or {}
                answer = str(message.get("content") or "")
                if message.get("reasoning_content") and on_thinking is not None:
                    on_thinking()
                finish_reason = str(choice.get("finish_reason") or "")
                usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            else:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    request_id = request_id or str(event.get("id", ""))
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    reasoning = str(delta.get("reasoning_content") or "")
                    if reasoning and not reasoning_notified:
                        reasoning_notified = True
                        if on_thinking is not None:
                            on_thinking()
                    chunk = str(delta.get("content") or "")
                    if chunk:
                        if metrics is not None and "first_token_seconds" not in metrics:
                            metrics["first_token_seconds"] = round(
                                time.perf_counter() - started, 3
                            )
                        pieces.append(chunk)
                        if on_text is not None:
                            on_text(chunk)
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                answer = "".join(pieces)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")[:1_000]
        try:
            parsed = json.loads(details)
            details = str((parsed.get("error") or {}).get("message") or "")[:500]
        except (json.JSONDecodeError, AttributeError):
            details = ""
        suffix = f"：{details}" if details else ""
        raise UltimateAPIError(f"究极服务返回 HTTP {error.code}{suffix}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise UltimateAPIError("暂时连不上究极服务，请检查网络后重试") from error

    recorded = _record_usage(
        usage,
        database_path=database_path,
        scope=scope,
        feature=feature,
        persona=persona_for_ultimate(model),
        request_id=request_id or uuid.uuid4().hex,
    )
    if metrics is not None:
        metrics.update(recorded)
        metrics["wall_seconds"] = round(time.perf_counter() - started, 3)
        metrics["ultimate"] = True
        metrics["cost_estimated"] = True
        denominator = int(recorded["cache_hit_tokens"]) + int(
            recorded["cache_miss_tokens"]
        )
        metrics["cache_hit_rate"] = (
            round(int(recorded["cache_hit_tokens"]) / denominator, 4)
            if denominator
            else 0.0
        )
    return answer, finish_reason


def call_background_preferred(
    local_callable: Callable[..., tuple[str, str]],
    model: str,
    messages: Sequence[dict[str, object]],
    config: object,
    *,
    database_path: str | Path | None = None,
    feature: str = "background",
    **kwargs: object,
) -> tuple[str, str]:
    """额度内尝试究极，额度不足、图像或网络异常时自动回落本地。"""
    if _has_images(messages) or not api_available():
        return local_callable(model, messages, config, **kwargs)
    persona = "shaya" if "shaya" in model or not model.startswith("huihui_ai/") else "aili"
    ultimate_model = f"ultimate:{persona}"
    metrics = kwargs.get("metrics")
    with _BACKGROUND_LOCK:
        used = _background_tokens_used(database_path)
        remaining = BACKGROUND_DAILY_TOKEN_LIMIT - used
        estimated_prompt = _estimate_tokens(
            _cache_friendly_messages(messages, background=True)
        )
        if remaining <= estimated_prompt + 64:
            if isinstance(metrics, dict):
                metrics["background_fallback"] = "daily_quota"
            return local_callable(model, messages, config, **kwargs)
        configured = int(
            kwargs.get("max_output") or getattr(config, "num_predict", 1024) or 1024
        )
        allowed_output = max(1, min(configured, remaining - estimated_prompt))
        try:
            return call_ultimate(
                ultimate_model,
                messages,
                config,
                database_path=database_path,
                scope="background",
                feature=feature,
                _background_max_output=allowed_output,
                **kwargs,
            )
        except UltimateAPIError:
            if isinstance(metrics, dict):
                metrics["background_fallback"] = "ultimate_unavailable"
            return local_callable(model, messages, config, **kwargs)
