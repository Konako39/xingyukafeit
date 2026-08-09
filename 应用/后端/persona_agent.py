#!/usr/bin/env python3
"""人格 Agent：把自然语言变成可验证的 Mac 本地工具动作。

流程：正则快筛（零成本跳过普通聊天）→ 当前会话模型按 JSON schema 抽取意图
→ 调用系统应用或专用助手执行 → 把结果作为
【本地工具结果】注入上下文，由人格自然转述 → 动作写入人格记忆池。

删除是不可逆动作，只在恰好匹配到一条日程时执行；多条匹配会把候选列表
交还给人格，请主人挑选后再删。
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from api_long_chat import (
    ModelConfig,
    PERSONAS,
    call_ollama,
    now_text,
)
from persona_memory_pool import add_persona_experience


_WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# 快筛：出现工具语汇才进入意图抽取，普通聊天零开销、零模型调用。
_CALENDAR_HINT = re.compile(
    r"日程|日历|行程|档期|安排|预约|会议|开会|约了|约在|"
    r"calendar|schedule|议程"
)
_REMINDER_HINT = re.compile(r"提醒|待办|备忘|记得.{0,8}(买|做|交|发|带|去|还)|todo|to-?do")
_FILE_HINT = re.compile(
    r"(找|搜|查).{0,10}(文件|文档|图片|照片|截图|表格|PPT|PDF|视频)|"
    r"(文件|文档|截图).{0,6}(在哪|去哪|找不到)"
)
_OPEN_HINT = re.compile(
    r"打开|启动|帮我开|访问.{0,20}(网站|网页|链接)|https?://|www\."
)
_NOTE_HINT = re.compile(r"备忘录|记个?笔记|记下来|帮我记[一下个]")
_MUSIC_HINT = re.compile(
    r"音乐|放歌|切歌|下一首|上一首|什么歌|(播放|暂停)[^器]?"
)
_TIME_HINT = re.compile(
    r"\d{1,2}[点:：时]|\d{1,2}月|今天|明天|后天|大后天|昨天|上午|下午|晚上|中午|凌晨|"
    r"周[一二三四五六日天末]|星期|礼拜|下个?[周月]|这个?[周月]|号|全天"
)
_ACTION_HINT = re.compile(
    r"加|添|建|创建|安排|定|订|记|设|预约|提醒|删|取消|去掉|移除|清掉|完成|做完|勾|划掉|"
    r"改|挪|调整|查|看|有什么|有没有|哪些|列出"
)


def looks_like_agent_request(text: str) -> bool:
    value = str(text or "").strip()
    if not value or len(value) > 400:
        return False
    if (
        _FILE_HINT.search(value)
        or _OPEN_HINT.search(value)
        or _NOTE_HINT.search(value)
        or _MUSIC_HINT.search(value)
    ):
        return True
    if _CALENDAR_HINT.search(value) or _REMINDER_HINT.search(value):
        return bool(_ACTION_HINT.search(value) or _TIME_HINT.search(value))
    return False


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def calendar_helper_path() -> Path:
    """优先用 .app 包内的可执行文件；TCC 权限是按这个 .app 身份授予的。"""
    client_dir = _project_root() / "数据" / "客户端"
    bundled = client_dir / "日历助手.app" / "Contents" / "MacOS" / "日历助手"
    return bundled if bundled.exists() else client_dir / "日历助手"


def calendar_helper_app_path() -> Path:
    return _project_root() / "数据" / "客户端" / "日历助手.app"


class CalendarToolError(RuntimeError):
    """日历助手执行失败；message 可直接展示给人格与主人。"""


def _parse_helper_output(text: str) -> dict[str, object]:
    value = str(text).strip()
    if not value:
        return {}
    try:
        return json.loads(value.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}


def _run_via_open(
    app_path: Path, arguments: list[str], timeout: int
) -> dict[str, object]:
    """以独立 app 身份运行：macOS 的 TCC 权限是按责任进程归属的，
    直接当子进程调用会被归到祖先进程而拿不到授权。open 会让它自己负责。
    结果拿不到 stdout，所以让助手写进临时文件再读回来。"""
    with tempfile.TemporaryDirectory() as workspace:
        result_file = Path(workspace) / "result.json"
        try:
            subprocess.run(
                [
                    "open", "-n", "-W", "-a", str(app_path), "--args",
                    *arguments, "--out", str(result_file),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise CalendarToolError("日历助手执行超时") from error
        try:
            return _parse_helper_output(result_file.read_text(encoding="utf-8"))
        except OSError:
            return {}


def _run_calendar_helper(arguments: list[str], *, timeout: int = 120) -> dict[str, object]:
    binary = calendar_helper_path()
    app_path = calendar_helper_app_path()
    if not binary.exists():
        build_script = _project_root() / "工具" / "构建日历助手.sh"
        raise CalendarToolError(
            f"日历助手尚未编译；请先运行 {build_script}"
        )
    payload: dict[str, object] = {}
    # 先直连（快、无 Dock 闪动）；被 TCC 挡住时再以独立 app 身份重试。
    try:
        completed = subprocess.run(
            [str(binary), *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        payload = _parse_helper_output(
            (completed.stdout or "").strip() or (completed.stderr or "").strip()
        )
    except subprocess.TimeoutExpired as error:
        raise CalendarToolError(
            "日历助手执行超时；如果系统正在弹出日历授权窗口，请先允许访问"
        ) from error
    if not payload.get("ok") and "授权" in str(payload.get("error", "")):
        if app_path.exists():
            payload = _run_via_open(app_path, arguments, timeout) or payload
    if not payload.get("ok"):
        message = str(payload.get("error") or "未知错误")[:300]
        if "未授权" in message or "授权" in message:
            # 主人用的是 App，不该被要求去 Finder 翻路径：直接把授权窗口弹出来。
            _launch_authorization_wizard()
            raise CalendarToolError(
                "系统还没放行日历和提醒事项的权限，所以这次什么都没有写进去。"
                "我已经把授权窗口调出来了，在上面点「允许」，然后再跟我说一次就行"
            )
        raise CalendarToolError(f"日历操作失败：{message}")
    return payload


_WIZARD_LOCK = threading.Lock()
_WIZARD_LAUNCHED_AT = 0.0
WIZARD_COOLDOWN_SECONDS = 120


def _launch_authorization_wizard() -> None:
    """把授权向导弹到主人面前；两分钟内不重复打扰。"""
    global _WIZARD_LAUNCHED_AT
    app_path = calendar_helper_app_path()
    if not app_path.exists():
        return
    with _WIZARD_LOCK:
        if time.monotonic() - _WIZARD_LAUNCHED_AT < WIZARD_COOLDOWN_SECONDS:
            return
        _WIZARD_LAUNCHED_AT = time.monotonic()
    try:
        subprocess.Popen(
            ["open", "-a", str(app_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 时间解析


_LOCAL_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_WEEKDAY_CN = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_RELATIVE_DAYS = {"今天": 0, "今晚": 0, "明天": 1, "明晚": 1, "后天": 2, "大后天": 3}
# 日期与时间的词法必须只写一处：解析、剥离标题、判断“这句有没有日期”共用，
# 否则会出现“能解析却没从标题里剥掉”这类不一致。
_NUMBER = r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3})"
_MONTH_DAY = rf"{_NUMBER}\s*月\s*{_NUMBER}\s*[日号]"
_RELATIVE_WORD = r"大后天|后天|明晚|明天|今晚|今天"
_WEEKDAY_WORD = r"(?:下+)?(?:周|星期|礼拜)[一二三四五六日天]"
_DATE_PATTERN = rf"{_RELATIVE_WORD}|{_WEEKDAY_WORD}|{_MONTH_DAY}"
_CLOCK_PATTERN = (
    r"\d{1,2}[:：]\d{2}|"
    r"(?:凌晨|清晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里)?\s*"
    rf"{_NUMBER}\s*[点时]"
    r"(?:半|一刻|三刻|[0-5]?\d\s*分|[0-5]\d(?!\d)|"
    r"[一二两三四五六七八九十]{1,3}\s*分|[二三四五]?十[一二三四五六七八九]?)?"
)


def _cn_number(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if not text:
        return None
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    total = 0
    for char in text:
        if char not in _CN_DIGITS:
            return None
        total = total * 10 + _CN_DIGITS[char]
    return total


def parse_chinese_datetime(
    text: str, base: dt.datetime | None = None
) -> tuple[dt.datetime | None, bool, str]:
    """确定性解析中文相对时间（零模型成本）。

    返回 (时间, 是否含具体钟点, 命中的原文片段)。解析不出返回 (None, False, "")；
    宁可返回 None 交给模型，也不猜测——质量第一。
    """
    now = base or dt.datetime.now()
    value = str(text)
    matched_parts: list[str] = []
    date_value: dt.date | None = None

    relative = re.search(_RELATIVE_WORD, value)
    weekday = re.search(r"(下+)?(?:周|星期|礼拜)([一二三四五六日天])", value)
    monthday = re.search(rf"({_NUMBER})\s*月\s*({_NUMBER})\s*[日号]", value)
    if relative:
        date_value = (now + dt.timedelta(days=_RELATIVE_DAYS[relative.group()])).date()
        matched_parts.append(relative.group())
    elif monthday:
        # 「八月七号」与「8月7号」都要认。
        month = _cn_number(monthday.group(1))
        day = _cn_number(monthday.group(2))
        if month is None or day is None:
            month, day = 0, 0
        try:
            candidate = dt.date(now.year, month, day)
            if candidate < now.date():
                candidate = dt.date(now.year + 1, month, day)
            date_value = candidate
            matched_parts.append(monthday.group())
        except ValueError:
            pass
    elif weekday:
        weeks_ahead = len(weekday.group(1) or "")
        target = _WEEKDAY_CN[weekday.group(2)]
        if weeks_ahead:
            # 下周X＝下一个自然周的周X（再多一个"下"顺延一周）。
            days = (7 - now.weekday()) + target + (weeks_ahead - 1) * 7
        else:
            days = (target - now.weekday()) % 7
        date_value = (now + dt.timedelta(days=days)).date()
        matched_parts.append(weekday.group())

    hour: int | None = None
    minute = 0
    # 时间部分在剔除日期片段后的剩余文本里找，
    # 否则"下周三十点"会被误读成 30 点。
    time_source = value
    for part in matched_parts:
        time_source = time_source.replace(part, "⌚", 1)
    clock = re.search(r"(\d{1,2})[:：](\d{2})", time_source)
    # 分钟支持：半 / 一刻三刻 / 15分 / 九点15 / 九点十五 / 九点四十五分
    spoken = re.search(
        r"(凌晨|清晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里)?\s*"
        rf"({_NUMBER})\s*[点时]"
        r"(半|一刻|三刻|[0-5]?\d\s*分|[0-5]\d(?!\d)|"
        r"[一二两三四五六七八九十]{1,3}\s*分|[二三四五]?十[一二三四五六七八九]?)?",
        time_source,
    )
    period = ""
    if clock:
        hour, minute = int(clock.group(1)), int(clock.group(2))
        matched_parts.append(clock.group())
        around = time_source[: clock.start()]
        period_match = re.search(r"(凌晨|上午|中午|下午|傍晚|晚上|夜里)\s*$", around)
        period = period_match.group(1) if period_match else ""
    elif spoken:
        parsed_hour = _cn_number(spoken.group(2))
        if parsed_hour is not None and 0 <= parsed_hour <= 24:
            hour = parsed_hour
            period = spoken.group(1) or ""
            suffix = (spoken.group(3) or "").strip()
            if suffix == "半":
                minute = 30
            elif suffix == "一刻":
                minute = 15
            elif suffix == "三刻":
                minute = 45
            elif suffix:
                parsed_minute = _cn_number(suffix.rstrip("分").strip())
                if parsed_minute is not None and 0 <= parsed_minute < 60:
                    minute = parsed_minute
            matched_parts.append(spoken.group())
    if hour is not None:
        if period in ("下午", "傍晚", "晚上", "夜里") and hour < 12:
            hour += 12
        elif period == "中午" and hour < 11:
            hour += 12
        elif not period and "晚" in "".join(matched_parts) and hour < 12:
            hour += 12
        if hour >= 24:
            hour -= 24

    if date_value is None and hour is None:
        return None, False, ""
    if date_value is None:
        date_value = now.date()
        implicit_date = True
    else:
        implicit_date = False
    if hour is None:
        return (
            dt.datetime.combine(date_value, dt.time(0, 0)),
            False,
            "".join(matched_parts),
        )
    result = dt.datetime.combine(date_value, dt.time(hour, minute))
    # 没说日期且钟点已过：按人的习惯理解为明天（"晚上8点提醒我"在22点说）。
    if implicit_date and result <= now:
        result += dt.timedelta(days=1)
    return result, True, "".join(matched_parts)


def _strip_time_words(text: str, matched: str = "") -> str:
    """从标题里剥掉所有日期与钟点词；词法与解析器共用，保证不会漏剥。"""
    value = str(text)
    for part in re.findall(rf"{_DATE_PATTERN}|{_CLOCK_PATTERN}", value):
        if part:
            value = value.replace(part, "")
    del matched
    return value.strip(" ，。,、的在于")


def has_explicit_date(text: str) -> bool:
    return bool(re.search(_DATE_PATTERN, str(text or "")))


def _parse_local_datetime(text: str) -> dt.datetime | None:
    value = str(text or "").strip()
    if not value:
        return None
    for fmt in _LOCAL_DATE_FORMATS:
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        parsed = dt.datetime.fromisoformat(value)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


def _format_for_helper(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _friendly_time(iso_text: str) -> str:
    moment = _parse_local_datetime(iso_text.split("+")[0].split("Z")[0])
    if moment is None:
        return iso_text
    weekday = _WEEKDAY_NAMES[moment.weekday()]
    return f"{moment.month}月{moment.day}日({weekday}) {moment.strftime('%H:%M')}"


# ---------------------------------------------------------------------------
# 意图抽取


_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "add_event", "delete_event", "query_events",
                "add_reminder", "query_reminders",
                "complete_reminder", "delete_reminder",
                "search_files", "open_target",
                "note_add", "music_control", "none",
            ],
        },
        "title": {"type": "string"},
        "start": {"type": "string"},
        "end": {"type": "string"},
        "all_day": {"type": "boolean"},
        "keywords": {"type": "string"},
        "window_start": {"type": "string"},
        "window_end": {"type": "string"},
        "location": {"type": "string"},
        "notes": {"type": "string"},
        "alarm_minutes": {"type": "number"},
        "due": {"type": "string"},
        "target": {"type": "string"},
        "target_kind": {"type": "string", "enum": ["app", "url", "path", ""]},
        "missing": {"type": "string"},
    },
    "required": ["intent"],
}


def _intent_messages(user_text: str) -> list[dict[str, object]]:
    now = dt.datetime.now()
    today = now.strftime("%Y-%m-%d")
    weekday = _WEEKDAY_NAMES[now.weekday()]
    system = (
        "你是本地工具指令解析器。判断用户这句话是否要求执行 Mac 上的本地操作，"
        "并抽取参数。只输出 JSON。规则："
        f"1) 现在是 {today}（{weekday}）{now.strftime('%H:%M')}，"
        "所有相对时间（明天、后天、下周三、晚上八点等）都换算成"
        "『YYYY-MM-DDTHH:MM』格式的本地时间；『下周X』指下一个自然周的周X。"
        "2) intent 可选："
        "add_event=新建日历日程；delete_event=取消日程；query_events=查询日程；"
        "add_reminder=添加提醒/待办（『提醒我做X』『记得买X』）；"
        "query_reminders=查询待办；complete_reminder=完成/勾掉某待办；"
        "delete_reminder=删除某待办；"
        "search_files=在电脑上找文件；open_target=打开某个应用/网址/文件；"
        "note_add=记到系统备忘录（明确说记备忘录/笔记时，notes 填内容）；"
        "music_control=控制音乐播放（target 填原话里的动作）；"
        "其余一律 none（普通聊天、单纯问时间、让你记住某事的都是 none）。"
        "有明确时刻的约会用 add_event，无时刻或强调『提醒我』的用 add_reminder。"
        "3) add_event 必须给 title（简洁名称，不含时间词）和 start；"
        "用户没给具体时间时把 missing 设为『开始时间』，start 留空。"
        "只说了日期没说几点、也不是全天事件时，missing 设为『具体几点』。"
        "全天事件 all_day=true 且 start 给当天 00:00。"
        "提前提醒填 alarm_minutes；没给结束时间 end 留空。"
        "4) add_reminder：title 必填，有截止/提醒时间填 due。"
        "5) delete/complete/query 类：keywords 填标题匹配关键词（没有留空）；"
        "window_start/window_end 给时间范围，指明日期用那天 00:00 到次日 00:00。"
        "6) search_files：keywords 填文件名或内容关键词（去掉『帮我找』等废话）。"
        "7) open_target：target 填应用名/完整网址/文件路径，"
        "target_kind 对应 app/url/path。"
        "8) 不要编造用户没说的信息。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(user_text)},
    ]


def extract_calendar_intent(
    user_text: str, model: str, config: ModelConfig
) -> dict[str, object]:
    answer, _ = call_ollama(
        model,
        _intent_messages(user_text),
        config,
        temperature=0.1,
        top_p=0.9,
        think=False,
        keep_alive="1m",
        response_format=_INTENT_SCHEMA,
        max_output=400,
    )
    value = str(answer).strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.S)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return {"intent": "none"}
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return {"intent": "none"}
    return parsed if isinstance(parsed, dict) else {"intent": "none"}


# ---------------------------------------------------------------------------
# 动作执行


def _search_events(
    keywords: str, window_start: dt.datetime, window_end: dt.datetime
) -> list[dict[str, object]]:
    arguments = [
        "events",
        "--from",
        _format_for_helper(window_start),
        "--to",
        _format_for_helper(window_end),
    ]
    if keywords.strip():
        arguments += ["--query", keywords.strip()]
    payload = _run_calendar_helper(arguments)
    events = payload.get("events")
    return list(events) if isinstance(events, list) else []


def _describe_event(event: dict[str, object]) -> str:
    title = str(event.get("title") or "（无标题）")
    start = _friendly_time(str(event.get("start") or ""))
    calendar = str(event.get("calendar") or "")
    location = str(event.get("location") or "")
    parts = [f"「{title}」{start}"]
    if location:
        parts.append(f"地点 {location}")
    if calendar:
        parts.append(f"日历「{calendar}」")
    return "，".join(parts)


def _do_add(intent: dict[str, object]) -> dict[str, object]:
    # 标题一律再清一次：模型有时会把日期原样塞进标题（“八月七号去成都”）。
    title = _sanitize_title(str(intent.get("title") or ""))
    missing = str(intent.get("missing") or "").strip()
    start = _parse_local_datetime(str(intent.get("start") or ""))
    if not title:
        return {"status": "clarify", "question": "日程要叫什么名字"}
    if missing or start is None:
        return {"status": "clarify", "question": missing or "开始时间"}
    # 模型给出的钟点不算授权：主人只说“明天下午”时，
    # 即使模型自作主张填了 15:00，也必须先问清具体几点。
    user_text = str(intent.get("_user_text") or "")
    all_day = bool(intent.get("all_day")) and bool(
        _ALL_DAY_WORDS.search(user_text)
    )
    stated_moment, stated_has_time, _ = parse_chinese_datetime(user_text)
    if not all_day and not stated_has_time:
        return {
            "status": "clarify",
            "question": "具体几点",
            "known_date": (
                stated_moment.date().isoformat()
                if stated_moment is not None
                else start.date().isoformat()
            ),
        }
    arguments = ["add", "--title", title, "--start", _format_for_helper(start)]
    end = _parse_local_datetime(str(intent.get("end") or ""))
    if end is not None and end > start:
        arguments += ["--end", _format_for_helper(end)]
    if all_day:
        arguments.append("--all-day")
    location = str(intent.get("location") or "").strip()
    if location:
        arguments += ["--location", location]
    notes = str(intent.get("notes") or "").strip()
    if notes:
        arguments += ["--notes", notes]
    try:
        alarm = float(intent.get("alarm_minutes") or 0)
    except (TypeError, ValueError):
        alarm = 0
    if alarm > 0:
        arguments += ["--alarm", str(int(alarm))]
    payload = _run_calendar_helper(arguments)
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    return {"status": "done", "action": "add", "event": event}


def _resolve_window(
    intent: dict[str, object], *, default_days_back: int, default_days_forward: int
) -> tuple[dt.datetime, dt.datetime]:
    now = dt.datetime.now()
    window_start = _parse_local_datetime(str(intent.get("window_start") or ""))
    window_end = _parse_local_datetime(str(intent.get("window_end") or ""))
    if window_start is None:
        window_start = (now - dt.timedelta(days=default_days_back)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if window_end is None or window_end <= window_start:
        window_end = window_start + dt.timedelta(
            days=max(1, default_days_back + default_days_forward)
        )
    return window_start, window_end


def _destructive_target_is_grounded(
    intent: dict[str, object], keywords: str
) -> bool:
    """删除/完成类动作的关键词必须真的出现在主人原话里。

    模型的 JSON 只用来结构化原话，不能凭空挑选一个真实日程或待办。
    去掉空白和常见标点后再比较，允许模型将“买 牛奶”规整为“买牛奶”。
    """
    source = str(intent.get("_user_text") or "").casefold()
    target = str(keywords or "").casefold()
    compact = lambda value: re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)
    source_compact = compact(source)
    target_compact = compact(target)
    return bool(target_compact and target_compact in source_compact)


def _do_delete(intent: dict[str, object]) -> dict[str, object]:
    keywords = str(intent.get("keywords") or intent.get("title") or "").strip()
    if not keywords or not _destructive_target_is_grounded(intent, keywords):
        return {"status": "clarify", "question": "要删哪个日程"}
    window_start, window_end = _resolve_window(
        intent, default_days_back=1, default_days_forward=60
    )
    matches = _search_events(keywords, window_start, window_end)
    if not matches:
        return {"status": "not_found", "action": "delete", "keywords": keywords}
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "action": "delete",
            "candidates": matches[:6],
        }
    target = matches[0]
    payload = _run_calendar_helper(["delete", "--id", str(target.get("id"))])
    deleted = payload.get("deleted") if isinstance(payload.get("deleted"), dict) else target
    return {"status": "done", "action": "delete", "event": deleted}


def _do_query(intent: dict[str, object]) -> dict[str, object]:
    keywords = str(intent.get("keywords") or "").strip()
    window_start, window_end = _resolve_window(
        intent, default_days_back=0, default_days_forward=7
    )
    matches = _search_events(keywords, window_start, window_end)
    return {
        "status": "done",
        "action": "query",
        "events": matches[:12],
        "window": (
            f"{_friendly_time(_format_for_helper(window_start))} 到 "
            f"{_friendly_time(_format_for_helper(window_end))}"
        ),
    }


# ---------------------------------------------------------------------------
# 提醒事项


def _list_reminders(keywords: str) -> list[dict[str, object]]:
    arguments = ["reminders"]
    if keywords.strip():
        arguments += ["--query", keywords.strip()]
    payload = _run_calendar_helper(arguments)
    reminders = payload.get("reminders")
    return list(reminders) if isinstance(reminders, list) else []


def _describe_reminder(reminder: dict[str, object]) -> str:
    title = str(reminder.get("title") or "（无标题）")
    due = str(reminder.get("due") or "")
    parts = [f"「{title}」"]
    if due:
        parts.append(f"截止 {_friendly_time(due)}")
    list_name = str(reminder.get("list") or "")
    if list_name:
        parts.append(f"列表「{list_name}」")
    return "，".join(parts)


def _do_reminder_add(intent: dict[str, object]) -> dict[str, object]:
    title = _sanitize_title(str(intent.get("title") or ""))
    if not title:
        return {"status": "clarify", "question": "要提醒的内容是什么"}
    arguments = ["reminder-add", "--title", title]
    due = _parse_local_datetime(str(intent.get("due") or intent.get("start") or ""))
    user_text = str(intent.get("_user_text") or "")
    stated_moment, stated_has_time, _ = parse_chinese_datetime(user_text)
    vague_period = re.search(r"凌晨|清晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里", user_text)
    if not stated_has_time and (stated_moment is not None or vague_period):
        # 用户说了日期/时段却没说钟点时，不允许模型自作主张填
        # 09:00 或 15:00。完全没说时间则可以创建无截止时间的待办。
        return {
            "status": "clarify",
            "question": "具体几点提醒",
            "known_date": (
                stated_moment.date().isoformat()
                if stated_moment is not None
                else ""
            ),
        }
    if due is not None and not stated_has_time:
        # 原话完全没有时间，模型却填了 due：忽略幻觉值，只建普通待办。
        due = None
    if due is not None:
        arguments += ["--due", _format_for_helper(due)]
    notes = str(intent.get("notes") or "").strip()
    if notes:
        arguments += ["--notes", notes]
    payload = _run_calendar_helper(arguments)
    reminder = (
        payload.get("reminder") if isinstance(payload.get("reminder"), dict) else {}
    )
    return {"status": "done", "action": "reminder_add", "reminder": reminder}


def _do_reminder_query(intent: dict[str, object]) -> dict[str, object]:
    keywords = str(intent.get("keywords") or "").strip()
    matches = _list_reminders(keywords)
    return {
        "status": "done",
        "action": "reminder_query",
        "reminders": matches[:12],
    }


def _do_reminder_mutation(intent: dict[str, object], command: str) -> dict[str, object]:
    action = "reminder_done" if command == "reminder-done" else "reminder_delete"
    keywords = str(intent.get("keywords") or intent.get("title") or "").strip()
    if not keywords or not _destructive_target_is_grounded(intent, keywords):
        question = "要完成哪条提醒" if command == "reminder-done" else "要删哪条提醒"
        return {"status": "clarify", "question": question}
    matches = _list_reminders(keywords)
    if not matches:
        return {"status": "not_found", "action": action, "keywords": keywords}
    if len(matches) > 1:
        return {"status": "ambiguous", "action": action, "candidates": matches[:6]}
    target = matches[0]
    payload = _run_calendar_helper([command, "--id", str(target.get("id"))])
    key = "completed" if command == "reminder-done" else "deleted"
    done = payload.get(key) if isinstance(payload.get(key), dict) else target
    return {"status": "done", "action": action, "reminder": done}


# ---------------------------------------------------------------------------
# 文件搜索（Spotlight，只读）


def _do_search_files(intent: dict[str, object]) -> dict[str, object]:
    keywords = str(intent.get("keywords") or intent.get("title") or "").strip()
    if not keywords:
        return {"status": "clarify", "question": "要找的文件叫什么或包含什么内容"}
    home = str(Path.home())
    try:
        completed = subprocess.run(
            ["mdfind", "-onlyin", home, keywords],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        raise CalendarToolError(f"Spotlight 搜索失败：{error}") from error
    if completed.returncode != 0:
        message = (completed.stderr or "").strip()[:200]
        raise CalendarToolError(
            f"Spotlight 搜索失败：{message or '未知错误'}"
        )
    paths = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    # 文件名直接命中的排最前；系统库和缓存噪音沉底。
    needle = keywords.lower()

    def rank(path: str) -> tuple[int, int]:
        name_hit = 0 if needle in Path(path).name.lower() else 1
        noise = 1 if "/Library/" in path or "/node_modules/" in path else 0
        return (noise, name_hit)

    paths.sort(key=rank)
    results = []
    for path in paths[:8]:
        display = path.replace(home, "~", 1)
        results.append({"path": path, "display": display})
    return {
        "status": "done",
        "action": "search_files",
        "keywords": keywords,
        "results": results,
        "total": len(paths),
    }


# ---------------------------------------------------------------------------
# 打开应用 / 网址 / 文件（只经 open 命令，参数列表传递，无 shell）


_URL_PATTERN = re.compile(r"^(https?://)?[\w.-]+\.[a-zA-Z]{2,}(/\S*)?$")


def _do_open(intent: dict[str, object]) -> dict[str, object]:
    target = str(intent.get("target") or "").strip()
    kind = str(intent.get("target_kind") or "").strip()
    if not target:
        return {"status": "clarify", "question": "要打开什么"}
    if kind == "url" or _URL_PATTERN.match(target):
        url = target if target.startswith(("http://", "https://")) else "https://" + target
        if not _URL_PATTERN.match(target):
            return {"status": "clarify", "question": "网址似乎不完整，能再说一遍吗"}
        completed = subprocess.run(
            ["open", url], capture_output=True, text=True, timeout=15
        )
        if completed.returncode != 0:
            raise CalendarToolError(f"打不开网址 {url}")
        return {"status": "done", "action": "open", "kind": "url", "target": url}
    if kind == "path" or target.startswith(("/", "~")):
        path = Path(target).expanduser()
        if not path.exists():
            return {"status": "not_found", "action": "open", "keywords": target}
        completed = subprocess.run(
            ["open", "-R", str(path)], capture_output=True, text=True, timeout=15
        )
        if completed.returncode != 0:
            raise CalendarToolError(f"无法在访达中显示 {path}")
        return {
            "status": "done", "action": "open", "kind": "path", "target": str(path),
        }
    # 默认按应用处理：先用 open -Ra 验证应用确实存在，再真正启动。
    check = subprocess.run(
        ["open", "-Ra", target], capture_output=True, text=True, timeout=15
    )
    if check.returncode != 0:
        return {"status": "not_found", "action": "open", "keywords": target}
    completed = subprocess.run(
        ["open", "-a", target], capture_output=True, text=True, timeout=15
    )
    if completed.returncode != 0:
        raise CalendarToolError(f"启动应用「{target}」失败")
    return {"status": "done", "action": "open", "kind": "app", "target": target}


# ---------------------------------------------------------------------------
# 备忘录（osascript 对 Notes 快而稳，不同于 Calendar 的 AppleEvent 超时问题）


def _run_osascript(lines: list[str], arguments: list[str], *, timeout: int = 30) -> str:
    command = ["/usr/bin/osascript"]
    for line in lines:
        command += ["-e", line]
    command += arguments
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        raise CalendarToolError(f"系统脚本执行失败：{error}") from error
    if completed.returncode != 0:
        message = (completed.stderr or "").strip()[:200]
        raise CalendarToolError(f"系统脚本执行失败：{message or '未知错误'}")
    return (completed.stdout or "").strip()


def _do_note(intent: dict[str, object]) -> dict[str, object]:
    content = str(
        intent.get("notes") or intent.get("title") or intent.get("keywords") or ""
    ).strip()
    if not content:
        return {"status": "clarify", "question": "要记什么内容"}
    title = content.splitlines()[0][:24]
    _run_osascript(
        [
            "on run argv",
            'tell application "Notes" to tell account 1 to make new note '
            "with properties {name:item 1 of argv, body:item 2 of argv}",
            "end run",
        ],
        [title, content],
    )
    return {"status": "done", "action": "note_add", "note": {"title": title, "content": content}}


# ---------------------------------------------------------------------------
# 音乐控制（Music.app）


_MUSIC_ACTIONS = {
    "暂停": "pause", "停": "pause", "别放了": "pause",
    "继续": "play", "播放": "play", "放歌": "play", "来点音乐": "play",
    "下一首": "next track", "换一首": "next track", "切歌": "next track",
    "上一首": "previous track",
}


def _do_music(intent: dict[str, object]) -> dict[str, object]:
    request = str(
        intent.get("target") or intent.get("keywords") or intent.get("title") or ""
    ).strip()
    action = ""
    for keyword, mapped in _MUSIC_ACTIONS.items():
        if keyword in request:
            action = mapped
            break
    if not action and re.search(r"什么歌|哪首|正在放|现在放", request):
        action = "current"
    if not action:
        return {"status": "clarify", "question": "想让音乐做什么（播放/暂停/下一首/在放什么）"}
    if action == "current":
        output = _run_osascript(
            [
                'tell application "System Events" to set musicRunning to '
                '(name of processes) contains "Music"',
                "if musicRunning then",
                'tell application "Music" to if player state is playing then '
                'return (get name of current track) & " — " & '
                "(get artist of current track)",
                'return "没有在播放"',
                "else",
                'return "音乐应用没有打开"',
                "end if",
            ],
            [],
        )
        return {"status": "done", "action": "music", "music": output or "没有在播放"}
    _run_osascript([f'tell application "Music" to {action}'], [])
    labels = {"play": "开始播放", "pause": "已暂停",
              "next track": "已切到下一首", "previous track": "已回到上一首"}
    return {"status": "done", "action": "music", "music": labels.get(action, action)}


# ---------------------------------------------------------------------------
# /命令：GUI 面板可点选，零模型成本直接结构化执行


QUICK_COMMANDS: dict[str, dict[str, str]] = {
    "日程": {"usage": "/日程 [今天|明天|本周|关键词]", "description": "查询日历日程"},
    "加日程": {"usage": "/加日程 标题 时间（如 明天下午3点）", "description": "添加日程"},
    "删日程": {"usage": "/删日程 关键词", "description": "取消日程（唯一匹配才删）"},
    "待办": {"usage": "/待办 [关键词]", "description": "查询未完成提醒"},
    "提醒": {"usage": "/提醒 内容 [时间]", "description": "添加提醒事项"},
    "完成": {"usage": "/完成 关键词", "description": "勾掉一条提醒"},
    "删提醒": {"usage": "/删提醒 关键词", "description": "删除一条提醒（唯一匹配才删）"},
    "找": {"usage": "/找 文件关键词", "description": "聚焦搜索找文件"},
    "打开": {"usage": "/打开 应用名或网址", "description": "打开应用/网址/文件"},
    "记": {"usage": "/记 内容", "description": "记到系统备忘录"},
    "音乐": {"usage": "/音乐 播放|暂停|下一首|在放什么", "description": "控制音乐"},
}


def _window_for_dateword(word: str) -> tuple[str, str]:
    now = dt.datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if word in ("今天", ""):
        return _format_for_helper(today), _format_for_helper(today + dt.timedelta(days=1))
    if word == "明天":
        start = today + dt.timedelta(days=1)
        return _format_for_helper(start), _format_for_helper(start + dt.timedelta(days=1))
    if word in ("本周", "这周"):
        start = today - dt.timedelta(days=now.weekday())
        return _format_for_helper(start), _format_for_helper(start + dt.timedelta(days=7))
    if word == "下周":
        start = today + dt.timedelta(days=7 - now.weekday())
        return _format_for_helper(start), _format_for_helper(start + dt.timedelta(days=7))
    return "", ""


def parse_slash_command(text: str) -> dict[str, object] | None:
    """把 /命令 直接转成结构化意图；非 / 开头返回 None。"""
    value = str(text or "").strip()
    if not value.startswith("/"):
        return None
    parts = value[1:].split(None, 1)
    command = parts[0] if parts else ""
    args = parts[1].strip() if len(parts) > 1 else ""
    if command not in QUICK_COMMANDS:
        listing = "\n".join(
            f"{info['usage']} —— {info['description']}"
            for info in QUICK_COMMANDS.values()
        )
        return {
            "intent": "help",
            "help_text": f"可用命令：\n{listing}",
        }
    if command == "日程":
        fixed = ("今天", "明天", "本周", "这周", "下周", "")
        if args in fixed:
            window_start, window_end = _window_for_dateword(args)
            keywords = ""
        else:
            # 「/日程 8月22号」「/日程 下周三」这类具体日期也要能查。
            moment, _has_time, matched = parse_chinese_datetime(args)
            keywords = _strip_time_words(args, matched)
            if moment is not None:
                day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
                window_start = _format_for_helper(day)
                window_end = _format_for_helper(day + dt.timedelta(days=1))
            else:
                window_start, window_end = _window_for_dateword("本周")
        return {
            "intent": "query_events", "keywords": keywords,
            "window_start": window_start, "window_end": window_end,
        }
    if command == "加日程":
        moment, has_time, matched = parse_chinese_datetime(args)
        title = _clean_title(_strip_time_words(args, matched))
        if not title:
            return {"intent": "add_event", "title": "", "missing": "日程名称"}
        if moment is None or not has_time:
            pending: dict[str, object] = {
                "intent": "add_event", "title": title,
                "missing": "具体几点" if moment is not None else "开始时间",
            }
            # 已经知道日期就记住，等主人补钟点时直接合上，不能丢。
            if moment is not None:
                pending["known_date"] = moment.date().isoformat()
            return pending
        return {
            "intent": "add_event", "title": title,
            "start": _format_for_helper(moment), "missing": "",
        }
    if command == "删日程":
        if not args:
            return {"intent": "delete_event", "keywords": "", "missing": "要删哪个日程"}
        return {"intent": "delete_event", "keywords": args}
    if command == "待办":
        return {"intent": "query_reminders", "keywords": args}
    if command == "提醒":
        moment, has_time, matched = parse_chinese_datetime(args)
        title = _clean_title(_strip_time_words(args, matched))
        if not title:
            return {"intent": "add_reminder", "title": "", "missing": "提醒内容"}
        intent: dict[str, object] = {"intent": "add_reminder", "title": title}
        if moment is not None:
            due = moment if has_time else moment.replace(hour=9, minute=0)
            intent["due"] = _format_for_helper(due)
        return intent
    if command == "完成":
        return {"intent": "complete_reminder", "keywords": args}
    if command == "删提醒":
        return {"intent": "delete_reminder", "keywords": args}
    if command == "找":
        return {"intent": "search_files", "keywords": args}
    if command == "打开":
        return {"intent": "open_target", "target": args, "target_kind": ""}
    if command == "记":
        return {"intent": "note_add", "notes": args}
    if command == "音乐":
        return {"intent": "music_control", "target": args or "播放"}
    return None


# ---------------------------------------------------------------------------
# 待澄清意图：人格问“几点？”之后，主人的回答本身没有日程关键词，
# 必须靠这里把补充信息接回原意图，否则工具永远不会真正执行。


PENDING_TTL_SECONDS = 600
_PENDING_LOCK = threading.Lock()
_PENDING_INTENTS: dict[str, dict[str, object]] = {}
_CANCEL_PATTERN = re.compile(r"算了|不用了|不用啦|取消|不加了|不要了|没事了|下次吧")
# 全天事件必须由主人明说，不能由模型替他决定。
_ALL_DAY_WORDS = re.compile(r"全天|一整天|整天|一天|全日|放假|休假|请假")


def remember_pending_intent(
    persona: str, conversation_id: int, intent: dict[str, object], question: str
) -> None:
    with _PENDING_LOCK:
        _PENDING_INTENTS[f"{persona}:{conversation_id}"] = {
            "intent": dict(intent),
            "question": str(question),
            "at": time.monotonic(),
        }


def take_pending_intent(persona: str, conversation_id: int) -> dict[str, object] | None:
    key = f"{persona}:{conversation_id}"
    with _PENDING_LOCK:
        entry = _PENDING_INTENTS.pop(key, None)
    if entry is None:
        return None
    if time.monotonic() - float(entry["at"]) > PENDING_TTL_SECONDS:
        return None
    return entry


def clear_pending_intent(persona: str, conversation_id: int) -> None:
    with _PENDING_LOCK:
        _PENDING_INTENTS.pop(f"{persona}:{conversation_id}", None)


def merge_pending_intent(
    pending: dict[str, object], user_text: str
) -> dict[str, object] | None:
    """把主人的补充信息合并回待澄清意图；判断不出补充内容就放弃。"""
    value = str(user_text).strip()
    if not value or _CANCEL_PATTERN.search(value):
        return None
    intent = dict(pending.get("intent") or {})
    name = str(intent.get("intent") or "")
    moment, has_time, matched = parse_chinese_datetime(value)
    known_date = str(intent.get("known_date") or "")
    if moment is not None and known_date and not has_explicit_date(value):
        # 主人这句只补了钟点，日期沿用上一轮已经说好的那天。
        try:
            moment = dt.datetime.combine(
                dt.date.fromisoformat(known_date), moment.time()
            )
        except ValueError:
            pass
    changed = False
    if name == "add_event":
        if moment is not None and has_time:
            intent["start"] = _format_for_helper(moment)
            intent["missing"] = ""
            intent.pop("known_date", None)
            changed = True
        elif moment is not None and not str(intent.get("start") or ""):
            # 只补到日期还差钟点，继续问。
            intent["start"] = ""
            intent["missing"] = "具体几点"
            intent["known_date"] = moment.date().isoformat()
            changed = True
        if not str(intent.get("title") or "").strip():
            title = _clean_title(_strip_time_words(value, matched))
            if title:
                intent["title"] = title
                changed = True
    elif name == "add_reminder":
        if moment is not None:
            due = moment if has_time else moment.replace(hour=9, minute=0)
            intent["due"] = _format_for_helper(due)
            changed = True
        if not str(intent.get("title") or "").strip():
            title = _clean_title(_strip_time_words(value, matched))
            if title:
                intent["title"] = title
                changed = True
    elif name in ("delete_event", "complete_reminder", "delete_reminder",
                  "search_files", "query_reminders", "query_events"):
        if not str(intent.get("keywords") or "").strip() and len(value) <= 40:
            intent["keywords"] = value
            changed = True
    elif name == "open_target":
        if not str(intent.get("target") or "").strip() and len(value) <= 60:
            intent["target"] = value
            changed = True
    elif name == "note_add":
        if not str(intent.get("notes") or "").strip():
            intent["notes"] = value
            changed = True
    return intent if changed else None


def _sanitize_title(text: str) -> str:
    """标题的最终防线：剥掉日期钟点与祈使残留；剥空了就退回原文。"""
    value = str(text).strip()
    if not value:
        return ""
    cleaned = _clean_title(_strip_time_words(value))
    return cleaned or value


def _clean_title(text: str) -> str:
    """去掉“帮我加一个日程”这类祈使残留，只留下真正的事由。"""
    value = str(text).strip()
    value = re.sub(
        r"^(?:帮我|给我|替我|麻烦|请)?(?:再)?(?:加|添加|建|创建|安排|记|订|定)"
        r"(?:一?[个条下])?",
        "",
        value,
    )
    value = re.sub(
        r"(?:的)?(?:日程|行程|安排|提醒|待办|事项)?"
        r"(?:再)?(?:加|添加|建|创建|记|安排)?(?:一?[个条下])?[吧呀啊呢]?$",
        "",
        value,
    )
    return value.strip(" ，。,、的")


# ---------------------------------------------------------------------------
# 高置信自然语言模板：模式明确就不花模型钱；有一点歧义就交给模型


def deterministic_intent(text: str) -> dict[str, object] | None:
    value = str(text or "").strip()
    if (
        not value
        or len(value) > 60
        or re.search(r"吗|[?？]|怎么样$|如何$|好不好$|行不行$|呢$", value)
    ):
        # 疑问语气可能不是指令（"你能提醒我吗？"），交给模型判断。
        return None
    reminder = re.search(r"提醒我(.{1,50})$", value)
    if reminder:
        moment, has_time, matched = parse_chinese_datetime(value)
        content = _strip_time_words(reminder.group(1), matched)
        content = content.strip(" ，。,、要去")
        if len(content) >= 2:
            intent: dict[str, object] = {"intent": "add_reminder", "title": content}
            if moment is not None:
                due = moment if has_time else moment.replace(hour=9, minute=0)
                intent["due"] = _format_for_helper(due)
            return intent
    query = re.fullmatch(
        r"(今天|明天|本周|这周|下周)(?:有什么|有啥|都有什么|有没有)?"
        r"(?:日程|安排|行程)[呀啊呢]?", value)
    if query:
        window_start, window_end = _window_for_dateword(query.group(1))
        return {
            "intent": "query_events", "keywords": "",
            "window_start": window_start, "window_end": window_end,
        }
    todo = re.fullmatch(
        r"(?:我)?(?:有什么|有哪些|查查?|看看?)(?:没完成的)?(?:待办|提醒|todo)[呀啊呢]?",
        value,
    )
    if todo:
        return {"intent": "query_reminders", "keywords": ""}
    return None


# ---------------------------------------------------------------------------
# 结果 → 工具上下文 + 记忆


def _tool_context_text(outcome: dict[str, object]) -> str:
    prefix = (
        "本地 Agent 工具刚刚真实执行或查询了以下操作，结果可靠，"
        "请用自己的语气自然告诉主人，不要复述原始数据格式：\n"
    )
    status = str(outcome.get("status"))
    action = str(outcome.get("action") or "")
    if status == "clarify":
        return (
            "本地工具判断主人想执行本地操作，但缺少必要信息："
            f"{outcome.get('question')}。请自然地向主人问清这一点。"
            "注意：现在什么都还没有执行，"
            "绝对不许说“记下了、加上了、搞定”之类的完成语；"
            "主人补充之后工具会自动接着执行。"
        )
    if status == "not_found":
        keywords = str(outcome.get("keywords") or "")
        hint = f"（关键词：{keywords}）" if keywords else ""
        if action == "open":
            return prefix + f"没有找到要打开的目标{hint}。"
        return prefix + f"没有找到匹配的条目{hint}。"
    if status == "ambiguous":
        candidates = outcome.get("candidates") or []
        describe = (
            _describe_reminder
            if action.startswith("reminder")
            else _describe_event
        )
        lines = "\n".join(
            f"{index + 1}. {describe(item)}"
            for index, item in enumerate(candidates)
        )
        return (
            prefix
            + "匹配到了多条，为安全起见没有改动任何一条。"
            + "请把这些候选念给主人，让主人选：\n"
            + lines
        )
    if action == "add":
        return prefix + "已成功添加日程：" + _describe_event(outcome.get("event") or {})
    if action == "delete":
        return prefix + "已成功删除日程：" + _describe_event(outcome.get("event") or {})
    if action == "query":
        events = outcome.get("events") or []
        window = str(outcome.get("window") or "")
        if not events:
            return prefix + f"查询范围（{window}）内没有任何日程。"
        lines = "\n".join(f"- {_describe_event(event)}" for event in events)
        return prefix + f"查询范围（{window}）内的日程：\n{lines}"
    if action == "reminder_add":
        return prefix + "已添加提醒：" + _describe_reminder(outcome.get("reminder") or {})
    if action == "reminder_done":
        return prefix + "已完成并勾掉提醒：" + _describe_reminder(
            outcome.get("reminder") or {}
        )
    if action == "reminder_delete":
        return prefix + "已删除提醒：" + _describe_reminder(
            outcome.get("reminder") or {}
        )
    if action == "reminder_query":
        reminders = outcome.get("reminders") or []
        if not reminders:
            return prefix + "当前没有未完成的匹配待办。"
        lines = "\n".join(f"- {_describe_reminder(item)}" for item in reminders)
        return prefix + f"未完成的待办：\n{lines}"
    if action == "search_files":
        results = outcome.get("results") or []
        total = int(outcome.get("total") or 0)
        if not results:
            return prefix + f"没搜到与「{outcome.get('keywords')}」相关的文件。"
        lines = "\n".join(f"- {item.get('display')}" for item in results)
        more = f"（共 {total} 个，只列前 {len(results)} 个）" if total > len(results) else ""
        return prefix + f"找到这些文件{more}：\n{lines}"
    if action == "open":
        kind_names = {"app": "应用", "url": "网页", "path": "文件位置"}
        return (
            prefix
            + f"已打开{kind_names.get(str(outcome.get('kind')), '目标')}："
            + str(outcome.get("target"))
        )
    if action == "note_add":
        note = outcome.get("note") or {}
        return prefix + f"已记到系统备忘录：「{note.get('title')}」。"
    if action == "music":
        return prefix + f"音乐控制结果：{outcome.get('music')}"
    return prefix + json.dumps(outcome, ensure_ascii=False)[:500]


def _tool_fallback_text(outcome: dict[str, object]) -> str:
    """人格模型在工具之后失败时仍可直接显示的确定性结果。

    这里不含任何给模型的内部指令，避免出现“动作已执行，但最后
    措辞生成失败”后只报错、诱导主人重复执行的情况。
    """
    status = str(outcome.get("status") or "")
    action = str(outcome.get("action") or "")
    if status == "clarify":
        return f"还没有执行，需要先确认：{outcome.get('question')}？"
    if status == "not_found":
        keywords = str(outcome.get("keywords") or "")
        suffix = f"（关键词：{keywords}）" if keywords else ""
        return f"没找到匹配的目标{suffix}，什么都没有改动。"
    if status == "ambiguous":
        candidates = list(outcome.get("candidates") or [])
        describe = _describe_reminder if action.startswith("reminder") else _describe_event
        lines = "\n".join(
            f"{index + 1}. {describe(item)}"
            for index, item in enumerate(candidates)
            if isinstance(item, dict)
        )
        return "匹配到多条，为了避免误操作，这次没有改动：\n" + lines
    if status != "done":
        return "这次本地操作没有执行成功。"
    if action == "add":
        return "已成功添加日程：" + _describe_event(outcome.get("event") or {})
    if action == "delete":
        return "已成功删除日程：" + _describe_event(outcome.get("event") or {})
    if action == "query":
        events = list(outcome.get("events") or [])
        if not events:
            return f"查询范围（{outcome.get('window')}）内没有日程。"
        return "查到的日程：\n" + "\n".join(
            f"- {_describe_event(item)}" for item in events if isinstance(item, dict)
        )
    if action == "reminder_add":
        return "已添加提醒：" + _describe_reminder(outcome.get("reminder") or {})
    if action == "reminder_done":
        return "已完成并勾掉提醒：" + _describe_reminder(outcome.get("reminder") or {})
    if action == "reminder_delete":
        return "已删除提醒：" + _describe_reminder(outcome.get("reminder") or {})
    if action == "reminder_query":
        reminders = list(outcome.get("reminders") or [])
        if not reminders:
            return "当前没有未完成的匹配待办。"
        return "未完成的待办：\n" + "\n".join(
            f"- {_describe_reminder(item)}"
            for item in reminders if isinstance(item, dict)
        )
    if action == "search_files":
        results = list(outcome.get("results") or [])
        if not results:
            return f"没搜到与「{outcome.get('keywords')}」相关的文件。"
        return "找到这些文件：\n" + "\n".join(
            f"- {item.get('display')}" for item in results if isinstance(item, dict)
        )
    if action == "open":
        return f"已打开：{outcome.get('target')}"
    if action == "note_add":
        note = outcome.get("note") or {}
        return f"已记到系统备忘录：「{note.get('title')}」。"
    if action == "music":
        return f"音乐控制结果：{outcome.get('music')}"
    return "本地工具已执行完成。"


_MUTATION_VERBS = {
    "add": ("calendar_action", "添加日程"),
    "delete": ("calendar_action", "删除日程"),
    "reminder_add": ("reminder_action", "添加提醒"),
    "reminder_done": ("reminder_action", "完成提醒"),
    "reminder_delete": ("reminder_action", "删除提醒"),
    "note_add": ("note_action", "记备忘录"),
}


def _record_calendar_experience(
    connection: sqlite3.Connection,
    persona: str,
    user_text: str,
    outcome: dict[str, object],
) -> None:
    """真实改动过日历/提醒的动作写入人格记忆池；查询和打开不记。"""
    status = str(outcome.get("status"))
    action = str(outcome.get("action") or "")
    if status != "done" or action not in _MUTATION_VERBS:
        return
    source_type, verb = _MUTATION_VERBS[action]
    item = (
        outcome.get("event") or outcome.get("reminder") or outcome.get("note") or {}
    )
    title = str(item.get("title") or "")
    describe = _describe_event if action in ("add", "delete") else _describe_reminder
    content = (
        f"主人说：“{str(user_text)[:120]}”。我通过本地工具为主人{verb}："
        f"{describe(item)}。"
    )
    try:
        add_persona_experience(
            connection,
            persona,
            source_type,
            f"{action}:{item.get('id') or now_text()}",
            f"{verb}：{title}"[:60],
            content,
            importance=0.6,
            metadata={"action": action, "item_id": str(item.get("id") or "")},
        )
    except (sqlite3.Error, ValueError):
        pass


# ---------------------------------------------------------------------------
# 诚实护栏：小模型有时不听“如实告知失败”的指令，这里做确定性兜底，
# 保证主人绝不会被一句“搞定了”骗过去。


_COMPLETION_CLAIM = re.compile(
    r"记下了|记好了|已?记上|加上了|加好了|已经?加|添加好|搞定|办好|设好|"
    r"定住|安排好|已安排|已设置|设置好|已创建|建好|存好|已保存|完成了|"
    r"加进|写进|加到|排上"
)
# 疑问和提议不是完成声明：“要不要帮你加进日历？”不能算已经做了。
_OFFER_OR_QUESTION = re.compile(r"要不要|需不需要|用不用|好不好|可以吗|行吗|吗[？?]")


def _asserted_sentences(answer: str) -> str:
    """只保留陈述句，去掉提议和疑问，避免把“要不要加”误判成“已经加了”。"""
    kept: list[str] = []
    for part in re.split(r"[。！!\n]|(?<=[？?])", str(answer)):
        piece = part.strip()
        if not piece or piece.endswith(("？", "?")):
            continue
        if _OFFER_OR_QUESTION.search(piece):
            continue
        kept.append(piece)
    return "。".join(kept)


_SYSTEM_ARTIFACT = re.compile(
    r"日程|日历|行程|提醒|待办|备忘|闹钟|加进|写进|排上|安排上"
)


def honesty_correction(
    agent_outcome: dict[str, object] | None, answer: str
) -> str:
    """回复里出现完成语但工具其实没成功时，返回必须追加的澄清句。"""
    text = _asserted_sentences(answer)
    if agent_outcome is None:
        # 工具根本没被触发，人格却宣称把日程/提醒办好了——必须点破。
        if not _COMPLETION_CLAIM.search(text):
            return ""
        if not (_SYSTEM_ARTIFACT.search(text) or has_explicit_date(text)):
            return ""
        return (
            "\n\n（说明：这只是我自己记住了，并没有写进系统日历或提醒事项。"
            "想真的加进去的话，说一声「加个日程」我就去办。）"
        )
    if agent_outcome.get("performed"):
        return ""
    if not _COMPLETION_CLAIM.search(text):
        return ""
    status = str(agent_outcome.get("status") or "")
    error = str(agent_outcome.get("error") or "")
    del text
    if error:
        return f"\n\n（说明：这条其实没能真的写进去——{error}）"
    if status == "clarify":
        return "\n\n（说明：这条还没有真的记下，等你把信息补全我才会真正写进去。）"
    if status == "ambiguous":
        return "\n\n（说明：匹配到多条，我没有动任何一条，你选一个我再执行。）"
    if status == "not_found":
        return "\n\n（说明：没找到对应的条目，什么都没有改动。）"
    return "\n\n（说明：这条其实没有真的执行成功。）"


# ---------------------------------------------------------------------------
# 对外入口


def handle_agent_request(
    connection: sqlite3.Connection,
    persona: str,
    model: str,
    config: ModelConfig,
    user_text: str,
    *,
    conversation_id: int = 0,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, object] | None:
    """尝试把用户消息当作本地工具指令处理。

    返回 None 表示这不是工具请求，走普通聊天；否则返回
    {"tool_context": 注入上下文的文本, "performed": 是否真的执行了工具, ...}。
    执行失败也返回上下文，让人格如实告诉主人失败原因。
    """
    # 三级解析：/命令与高置信模板零模型成本，剩下的才花一次小模型调用。
    parse_source = "slash"
    intent = parse_slash_command(user_text)
    if intent is None:
        # 上一轮问过“几点/叫什么”时，主人的回答本身没有日程关键词，
        # 必须先把补充信息接回待澄清意图，否则工具永远不会真正执行。
        pending = take_pending_intent(persona, conversation_id)
        fresh_request = looks_like_agent_request(user_text)
        # 主人可以在澄清期间直接改发一条完整的新指令。这种情况
        # 应抛弃旧指令，否则“要删哪个日程？”后说“完成一条待办”
        # 会被错当成旧删除指令的关键词。只有钟点、标题等非完整片段
        # 才合并回待澄清意图。
        if pending is not None and not fresh_request:
            merged = merge_pending_intent(pending, user_text)
            if merged is not None:
                intent = merged
                parse_source = "pending"
        if intent is None:
            if not fresh_request:
                return None
            intent = deterministic_intent(user_text)
            parse_source = "deterministic"
            if intent is None:
                if on_status is not None:
                    on_status("正在解析指令")
                intent = extract_calendar_intent(user_text, model, config)
                parse_source = "model"
    intent_name = str(intent.get("intent") or "none")
    if intent_name == "none":
        return None
    if intent_name == "help":
        help_text = str(intent.get("help_text", ""))
        return {
            "tool_context": (
                "主人输入了一个未知的 / 命令。请把下面的可用命令自然地告诉主人：\n"
                + help_text
            ),
            "fallback_text": help_text,
            "performed": False,
            "intent": "help",
            "parse_source": parse_source,
        }
    if on_status is not None:
        labels = {
            "add_event": "通过系统日历添加日程",
            "delete_event": "通过系统日历取消日程",
            "query_events": "查询系统日历",
            "add_reminder": "添加系统提醒",
            "query_reminders": "查询待办提醒",
            "complete_reminder": "勾掉已完成的提醒",
            "delete_reminder": "删除提醒",
            "search_files": "用聚焦搜索找文件",
            "open_target": "打开目标",
            "note_add": "记到系统备忘录",
            "music_control": "控制音乐",
        }
        on_status(f"正在{labels.get(intent_name, '执行本地操作')}")
    executors = {
        "add_event": _do_add,
        "delete_event": _do_delete,
        "query_events": _do_query,
        "add_reminder": _do_reminder_add,
        "query_reminders": _do_reminder_query,
        "complete_reminder": lambda item: _do_reminder_mutation(item, "reminder-done"),
        "delete_reminder": lambda item: _do_reminder_mutation(item, "reminder-delete"),
        "search_files": _do_search_files,
        "open_target": _do_open,
        "note_add": _do_note,
        "music_control": _do_music,
    }
    if intent_name not in executors:
        # 模型 JSON 即使越出 schema，也不能让未知动作掉到
        # KeyError 后被客户端误当成普通聊天。
        return {
            "tool_context": "本地 Agent 没有识别出可执行的工具动作，什么都没有改动。",
            "fallback_text": "这条指令还不在本地 Agent 的可执行范围内，什么都没有改动。",
            "performed": False,
            "mutated": False,
            "intent": intent_name,
            "status": "unsupported",
            "parse_source": parse_source,
        }
    # 执行层需要看主人的原话，才能判断“全天”这类决定是不是主人自己说的。
    intent["_user_text"] = str(user_text)
    try:
        outcome = executors[intent_name](intent)
    except CalendarToolError as error:
        fallback_text = f"这次本地操作没做成：{error}"
        return {
            "tool_context": (
                "【重要】本地工具执行失败了，什么都没有写进系统。"
                "你必须如实、直接地告诉主人这件事没做成以及原因，"
                "绝对不许说“记下了、加上了、搞定、定住了、安排好了”之类的话。"
                f"失败原因：{error}"
            ),
            "fallback_text": fallback_text,
            "performed": False,
            "intent": intent_name,
            "status": "failed",
            "error": str(error),
            "parse_source": parse_source,
        }
    # 信息不全时把意图挂起，等主人下一句补充后自动接上继续执行。
    if str(outcome.get("status")) == "clarify":
        pending_intent = dict(intent)
        # 执行层若已经算出日期（例如模型给了零点），直接沿用它。
        if outcome.get("known_date"):
            pending_intent["known_date"] = str(outcome["known_date"])
        # 不管意图来自 /命令、模板还是模型，只要原话里说了日期就必须记住，
        # 否则主人只补钟点时会掉到今天——这正是“八月七号”被记成今天的原因。
        if not pending_intent.get("known_date") and has_explicit_date(user_text):
            known_moment, _has_time, _matched = parse_chinese_datetime(user_text)
            if known_moment is not None:
                pending_intent["known_date"] = known_moment.date().isoformat()
        remember_pending_intent(
            persona, conversation_id, pending_intent,
            str(outcome.get("question") or ""),
        )
    else:
        clear_pending_intent(persona, conversation_id)
    _record_calendar_experience(connection, persona, user_text, outcome)
    status = str(outcome.get("status"))
    action = str(outcome.get("action") or "")
    executed = status == "done"
    return {
        "tool_context": _tool_context_text(outcome),
        "fallback_text": _tool_fallback_text(outcome),
        "performed": executed,
        "mutated": executed and action in _MUTATION_VERBS,
        "intent": intent_name,
        "status": status,
        "parse_source": parse_source,
    }
