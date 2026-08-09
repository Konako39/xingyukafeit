#!/usr/bin/env python3
"""艾莉与沙雅的后台茶话室。

这一层只在 macOS 长时间无操作、内存和负载都宽裕时工作。
它只读观察非系统文件，并把来源、时间和模型一起存入独立日志；
茶话室从不直接改写用户的亲口档案。
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import math
import json
import os
import random
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from api_long_chat import (
    EMBED_MODEL,
    LOCAL_AI_IDENTITY,
    MODEL_CONFIGS,
    PERSONAS,
    QUALITY_HELPER_MODELS,
    _pack_embedding,
    _unpack_embedding,
    call_embeddings,
    call_ollama as call_model,
    config_for_model,
    format_retrieved_history,
    get_persona_memory,
    memory_text_overlap,
    now_text,
    open_database,
    retrieve_persona_history,
    semantic_excerpt,
    update_persona_self_profile,
)
from deepseek_gateway import (
    api_available as ultimate_available,
    call_background_preferred,
    usage_summary as ultimate_usage_summary,
)
from persona_memory_pool import (
    add_persona_experience,
    append_screen_daily_digest,
    ensure_persona_memory_pool_schema,
    format_persona_experiences,
    index_persona_experiences,
    memory_pool_stats,
    recent_persona_experiences,
    retrieve_persona_experiences,
)
from persona_growth import growth_identity_prompt


OLLAMA_BASE = "http://127.0.0.1:11434"
RUN_LOCK = threading.Lock()
ACTIVITY_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
LAST_USER_ACTIVITY = time.monotonic()
LAST_USER_CHAT_ACTIVITY = time.monotonic()
SCREEN_REQUEST_LOCK = threading.Lock()
SCREEN_PENDING_REQUEST: dict[str, object] = {}
SCREEN_REQUEST_TTL_SECONDS = 180
SCHEDULER_POLL_SECONDS = 180
MODEL_TAG_CACHE: tuple[float, set[str]] = (0.0, set())
FOREGROUND_STATE_LOCK = threading.Lock()
FOREGROUND_STATE_CACHE: tuple[float, dict[str, object]] = (0.0, {})
FOREGROUND_STATE_TTL_SECONDS = 6.0
FRONTMOST_STATE_SCRIPT = Path(__file__).with_name("frontmost_state.js")
MIN_LOUNGE_COMPLETE_ROUNDS = 5
MIN_LOUNGE_MESSAGES_BEFORE_DECISION = MIN_LOUNGE_COMPLETE_ROUNDS * 2
SCREEN_REPEAT_OVERLAP_LIMIT = 0.76


def call_ollama(
    model: str,
    messages: Sequence[dict[str, object]],
    config: object,
    **kwargs: object,
) -> tuple[str, str]:
    """茶话室的统一后台路由：日额度内用究极，否则原样回落本地。"""
    return call_background_preferred(
        call_model,
        model,
        messages,
        config,
        feature="lounge",
        **kwargs,
    )

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".html", ".htm", ".css", ".scss", ".sql", ".env",
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".vue", ".svelte", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp",
    ".swift", ".java", ".kt", ".kts", ".rs", ".go", ".rb", ".php",
    ".sh", ".zsh", ".fish", ".ps1", ".lua", ".shader", ".glsl",
}
DOCUMENT_EXTENSIONS = {".pdf", ".rtf", ".doc", ".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IGNORED_PARTS = {
    ".git", ".svn", "node_modules", "__pycache__", ".Trash", "Caches",
    "DerivedData", ".cache", ".npm", ".gradle", ".ollama", "System",
}

# 自由闲聊也给 4B 模型一个很轻的落点，避免它为了“找话题”编造刚去过的
# 店、刚追过的作品或不存在的实时新闻。种子只提供观点，不提供外部事实。
FREE_TOPIC_CUES = (
    "比起完美结局，带一点遗憾的结局是不是更容易让人记住",
    "玩游戏时完全盲玩和提前看攻略，各自有哪里更舒服",
    "如果房间只能留一种背景声，会选雨声、风声还是彻底安静",
    "两个人熟起来之后，默契更像少说话还是更敢说废话",
    "一件作品先打动情绪和先说服逻辑，哪种更容易留下来",
    "如果能给日常生活加一条游戏规则，什么规则会最有趣",
    "认真做事时被打断很烦，但有时意外插曲反而会带来灵感",
    "收藏东西的快乐更偏向拥有本身，还是整理和回看的过程",
    "聊天里秒回、慢慢回和想到了再回，各自给人的感觉",
    "比起能力很强但难相处的人，可靠又普通的人是不是更耐看",
    "虚拟角色最重要的是设定够特别，还是相处久了形成的细节",
    "游戏里的高难度是让胜利更有分量，还是容易把乐趣磨掉",
    "一个界面是第一眼惊艳更重要，还是每天用都不烦更重要",
    "创作时保留一点粗糙的个人味道，会不会比过度打磨更耐看",
    "小尴尬应该很快忘掉，还是偶尔拿出来互相逗一下更有意思",
)

# 这是茶话室和屏幕观察共同使用的“世界模型”。小参数模型如果只看到一段
# 文件/屏幕文本，很容易把证据来源错认成另一个人格，所以这些关系不依赖
# 模型自行推断，而是作为每轮最高优先级事实反复锚定。
LOUNGE_WORLD_MODEL = (
    "【不可改写的现实关系】主人是唯一的真人、Mac 操作者和本机文件所有者。"
    "屏幕、桌面、Safari/其他应用、项目、README、截图和文件操作都属于主人；"
    "艾莉与沙雅不会亲手打开浏览器、整理桌面或操作这些文件。"
    "艾莉和沙雅是星语茶话屋里的两个独立人格，共同服务主人，彼此是聊天对象。"
    "程序提供的同一份文件或截图证据视为两人共同看见，不能说成‘你的屏幕’、"
    "‘你刚打开的文件’，也不能由另一人格冒认主人的动作。"
    "当证据出现星语茶话屋、旧称 Local AI Studio 或本地AI、双人格、艾莉、沙雅、"
    "4B/9B/27B、记忆池、茶话室、屏幕观察或 9B 记忆整理器时，那是在描述你们"
    "自身所在的系统和功能；应以‘我们这套系统/我们所在的应用’理解，不能当成"
    "与自己无关的第三方产品。"
)


def _grounding_violation(text: str, topic_mode: str = "") -> str:
    """识别不能进入长期检索的角色/所有权错位。"""
    value = re.sub(r"\s+", "", str(text or ""))
    if not value:
        return ""
    if re.search(r"主人不在(?:场)?", value):
        return "把主人缺席当成固定开场，破坏自然对话"
    if re.search(r"等主人(?:回来|上线)|主人回来再|等主人来", value):
        return "把主人是否在场变成了话题，而不是和另一个人格自然聊天"
    if re.search(
        r"你(?:刚才|刚刚|刚|正在|在|昨天).{0,16}"
        r"(?:打开|点开|浏览|使用Safari|用Safari|整理|操作|查看|盯着|截屏|截图)",
        value,
    ):
        return "把主人的电脑操作错归给另一个人格"
    if re.search(r"你的(?:屏幕|桌面|浏览器|Safari|截图|截屏|README|文件夹)", value):
        return "把主人的屏幕或文件错认成另一个人格所有"
    if re.search(
        r"我(?:在|正在|昨天|刚好|准备|打算).{0,18}"
        r"(?:整理|操作|使用Safari|用Safari|打开浏览器|点开文件|整理截屏|整理截图)",
        value,
    ):
        return "人格冒认了主人的现实电脑操作"
    if re.search(r"如果我是(?:个)?(?:机器人|AI|人工智能)", value):
        return "人格忘记自己本来就是本地 AI 人格"
    if topic_mode in {"memory", "resume"} and re.search(
        r"(?:今天|刚才|刚刚|现在).{0,80}"
        r"(?:屏幕|桌面|Safari|截图|截屏|进度条|新动向|新变化)"
        r"|(?:屏幕|桌面|Safari|截图|截屏|进度条).{0,80}"
        r"(?:今天|刚才|刚刚|现在|正在|正盯着)",
        value,
    ):
        return "把旧记忆说成了此刻新观察，属于编造现实经历"
    if topic_mode == "file" and re.search(
        r"主人(?:刚才|刚刚|现在|正在|真会|正好).{0,18}"
        r"(?:看这个|看这|查看|打开|点开|浏览|挑时间看)",
        value,
    ):
        return "文件是程序自动选出的共同证据，不能编造主人此刻正在查看它"
    if topic_mode in {"file", "screen"} and re.search(
        r"(?:你)?别(?:光)?在那.{0,12}(?:分析|查看|看|整理|操作|调试|打开|浏览)"
        r"|你(?:还|正|正在)?在.{0,12}(?:分析|整理|操作|调试)(?:屏幕|代码|文件|项目)?",
        value,
    ):
        return "把主人或程序正在进行的观察与分析动作错归给另一个人格"
    return ""


def _screen_record_session_id(metadata_text: str) -> int:
    try:
        metadata = json.loads(metadata_text or "{}")
    except json.JSONDecodeError:
        return 0
    if not isinstance(metadata, dict):
        return 0
    discussion = metadata.get("discussion")
    if not isinstance(discussion, dict):
        return 0
    try:
        return int(discussion.get("session_id") or 0)
    except (TypeError, ValueError):
        return 0


def _quarantine_legacy_screen_errors(
    connection: sqlite3.Connection,
) -> tuple[dict[int, str], dict[int, str]]:
    """隔离旧屏幕复读和无证据推测，原始日志仍留在数据库供审计。"""
    try:
        rows = connection.execute(
            "SELECT id, status, quality_status, quality_reason, metadata, "
            "aili_observation, shaya_observation FROM screen_observations "
            "ORDER BY id"
        ).fetchall()
    except sqlite3.Error:
        return {}, {}
    bad_records: dict[int, str] = {}
    bad_sessions: dict[int, str] = {}
    accepted_texts: dict[str, list[str]] = {"aili": [], "shaya": []}
    for row in rows:
        record_id = int(row["id"])
        quality_status = str(row["quality_status"] or "accepted")
        if quality_status == "quarantined":
            reason = str(row["quality_reason"] or "旧屏幕质量审计已隔离")
            bad_records[record_id] = reason
            session_id = _screen_record_session_id(str(row["metadata"] or "{}"))
            if session_id:
                bad_sessions[session_id] = reason
            continue
        if str(row["status"]) != "completed" or quality_status != "accepted":
            continue
        reasons: list[str] = []
        current_texts: dict[str, str] = {}
        for persona, column in (
            ("aili", "aili_observation"),
            ("shaya", "shaya_observation"),
        ):
            answer = str(row[column] or "").strip()
            current_texts[persona] = answer
            issue = _screen_observation_issue(answer)
            if not issue:
                overlap = max(
                    (
                        memory_text_overlap(answer, previous)
                        for previous in accepted_texts[persona][-8:]
                    ),
                    default=0.0,
                )
                if overlap >= SCREEN_REPEAT_OVERLAP_LIMIT:
                    issue = f"与较早屏幕观察重复度过高（{overlap:.0%}）"
            if issue:
                reasons.append(f"{PERSONAS[persona].name}：{issue}")
        if reasons:
            reason = "；".join(reasons)[:1_000]
            bad_records[record_id] = reason
            connection.execute(
                "UPDATE screen_observations SET quality_status = 'quarantined', "
                "quality_reason = ? WHERE id = ?",
                (reason, record_id),
            )
            session_id = _screen_record_session_id(str(row["metadata"] or "{}"))
            if session_id:
                bad_sessions[session_id] = reason
        else:
            for persona, answer in current_texts.items():
                if answer:
                    accepted_texts[persona].append(answer)
    return bad_records, bad_sessions


def _quarantine_legacy_grounding_errors(connection: sqlite3.Connection) -> int:
    """保留旧日志用于审计，但把明显错误历史移出所有记忆检索。"""
    bad_screen_records, bad_screen_sessions = _quarantine_legacy_screen_errors(
        connection
    )
    try:
        existing_bad = connection.execute(
            "SELECT id, quality_reason FROM lounge_sessions "
            "WHERE quality_status = 'quarantined'"
        ).fetchall()
        sessions = connection.execute(
            "SELECT id, topic_mode FROM lounge_sessions "
            "WHERE quality_status = 'accepted'"
        ).fetchall()
    except sqlite3.Error:
        return 0
    # 既处理本次新发现的错误，也持续同步过去已经隔离的会话，避免旧迁移逻辑
    # 重新写入同一个 source_key 时把经历状态改回 active。
    quarantined: dict[int, str] = {
        int(row["id"]): str(row["quality_reason"] or "历史质量审计已隔离")
        for row in existing_bad
    }
    quarantined.update(bad_screen_sessions)
    screen_session_ids = {
        int(row["id"])
        for row in connection.execute(
            "SELECT id FROM lounge_sessions WHERE trigger_type = 'screen'"
        ).fetchall()
    }
    newly_quarantined = 0
    for session in sessions:
        messages = connection.execute(
            "SELECT speaker, content FROM lounge_messages "
            "WHERE lounge_session_id = ? ORDER BY id",
            (session["id"],),
        ).fetchall()
        for message in messages:
            content = str(message["content"])
            issue = _grounding_violation(content, str(session["topic_mode"]))
            speaker = str(message["speaker"])
            other_name = "沙雅" if speaker == "aili" else "艾莉"
            if not issue and str(session["topic_mode"]) == "screen" and re.search(
                rf"{other_name}.{{0,28}}(?:盯着|整理|操作|打开|点开|查看|发呆)",
                content,
            ):
                issue = "把屏幕里的主人行为或无证据动作编到另一人格身上"
            if not issue and len(content.strip()) > 130:
                issue = "旧发言超过当前聊天质量上限，容易形成会议式小作文"
            if not issue and re.search(r"(?:^|\n)\s*(?:[-*#]|\d+[.)、])\s*", content):
                issue = "旧发言使用列表或 Markdown，不符合自然私聊"
            if issue:
                quarantined[int(session["id"])] = issue
                newly_quarantined += 1
                break
    if quarantined:
        for session_id, issue in quarantined.items():
            connection.execute(
                "UPDATE lounge_sessions SET quality_status = 'quarantined', "
                "quality_reason = ? WHERE id = ?",
                (issue, session_id),
            )
        message_rows = connection.execute(
            "SELECT id, lounge_session_id FROM lounge_messages"
        ).fetchall()
        bad_message_ids = [
            int(row["id"])
            for row in message_rows
            if int(row["lounge_session_id"]) in quarantined
        ]
        for message_id in bad_message_ids:
            connection.execute(
                "DELETE FROM lounge_embeddings WHERE source_type = 'message' AND source_id = ?",
                (message_id,),
            )
    experiences = connection.execute(
        "SELECT id, source_type, source_key, content, metadata "
        "FROM persona_experiences WHERE status = 'active'"
    ).fetchall()
    for row in experiences:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        source_type = str(row["source_type"])
        content = str(row["content"] or "")
        linked_session_id = int(metadata.get("lounge_session_id") or 0)
        if not linked_session_id and source_type in {
            "lounge_message", "lounge_conversation", "file_observation"
        }:
            match = re.match(r"(\d+)", str(row["source_key"]))
            linked_session_id = int(match.group(1)) if match else 0
        experience_issue = ""
        if linked_session_id in quarantined:
            experience_issue = quarantined[linked_session_id]
        elif source_type == "lounge_message" and linked_session_id in screen_session_ids:
            experience_issue = "屏幕茶话逐条副本已停用，仅保留整轮交流和原始视觉事实"
        elif (
            source_type == "screen_observation"
            and str(row["source_key"]).isdigit()
            and int(str(row["source_key"])) in bad_screen_records
        ):
            experience_issue = bad_screen_records[int(str(row["source_key"]))]
        elif source_type.startswith("screen_") and _screen_observation_issue(content):
            experience_issue = "旧屏幕记忆含有复读、人格叙事或截图无法证明的推测"
        elif re.search(r"如果我是(?:个)?(?:机器人|AI|人工智能)|用户\s*konako", content):
            experience_issue = "旧经历存在身份错位或把程序推测写成主人事实"
        if experience_issue:
            connection.execute(
                "UPDATE persona_experiences SET status = 'quarantined', updated_at = ? "
                "WHERE id = ?",
                (now_text(), int(row["id"])),
            )
            connection.execute(
                "DELETE FROM persona_experience_embeddings WHERE experience_id = ?",
                (int(row["id"]),),
            )
    notes = connection.execute(
        "SELECT id, content, lounge_session_id FROM lounge_notes "
        "WHERE quality_status = 'accepted'"
    ).fetchall()
    for note in notes:
        linked_bad = int(note["lounge_session_id"] or 0) in quarantined
        content = str(note["content"] or "")
        note_bad = bool(
            re.search(r"如果我是(?:个)?(?:机器人|AI|人工智能)|用户\s*konako", content)
            or _grounding_violation(content)
        )
        if linked_bad or note_bad:
            connection.execute(
                "UPDATE lounge_notes SET quality_status = 'quarantined' WHERE id = ?",
                (int(note["id"]),),
            )
            connection.execute(
                "DELETE FROM lounge_embeddings WHERE source_type = 'note' AND source_id = ?",
                (int(note["id"]),),
            )
    return newly_quarantined


def record_user_activity() -> None:
    global LAST_USER_ACTIVITY
    with ACTIVITY_LOCK:
        LAST_USER_ACTIVITY = time.monotonic()


def record_user_chat_activity() -> None:
    """记录主人真正发出聊天请求；普通鼠标键盘活动不会更新这个标记。"""
    global LAST_USER_CHAT_ACTIVITY
    timestamp = time.monotonic()
    with ACTIVITY_LOCK:
        global LAST_USER_ACTIVITY
        LAST_USER_ACTIVITY = timestamp
        LAST_USER_CHAT_ACTIVITY = timestamp


def seconds_since_user_activity() -> float:
    with ACTIVITY_LOCK:
        return max(0.0, time.monotonic() - LAST_USER_ACTIVITY)


def user_activity_marker() -> float:
    with ACTIVITY_LOCK:
        return LAST_USER_ACTIVITY


def user_chat_activity_marker() -> float:
    with ACTIVITY_LOCK:
        return LAST_USER_CHAT_ACTIVITY


def default_scan_roots() -> list[str]:
    roots = [str(Path.home()), "/Applications"]
    volumes = Path("/Volumes")
    if volumes.is_dir():
        for item in sorted(volumes.iterdir()):
            if item.name not in {"Macintosh HD", "Preboot", "Recovery", "VM"}:
                roots.append(str(item))
    return roots


def ensure_lounge_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS lounge_config (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            enabled INTEGER NOT NULL DEFAULT 1,
            idle_minutes INTEGER NOT NULL DEFAULT 15,
            min_interval_minutes INTEGER NOT NULL DEFAULT 180,
            max_interval_minutes INTEGER NOT NULL DEFAULT 360,
            max_daily_rounds INTEGER NOT NULL DEFAULT 4,
            model_strategy TEXT NOT NULL DEFAULT 'auto',
            inspect_files INTEGER NOT NULL DEFAULT 1,
            scan_roots TEXT NOT NULL DEFAULT '[]',
            screen_watch_enabled INTEGER NOT NULL DEFAULT 1,
            screen_min_interval_minutes INTEGER NOT NULL DEFAULT 60,
            screen_max_interval_minutes INTEGER NOT NULL DEFAULT 180,
            screen_max_daily INTEGER NOT NULL DEFAULT 6,
            screen_next_run_after TEXT NOT NULL DEFAULT '',
            screen_last_run_at TEXT NOT NULL DEFAULT '',
            screen_last_status TEXT NOT NULL DEFAULT '等待首次屏幕观察',
            screen_last_error TEXT NOT NULL DEFAULT '',
            screen_manual_requested INTEGER NOT NULL DEFAULT 0,
            screen_diagnostic_requested INTEGER NOT NULL DEFAULT 0,
            screen_diagnostic_status TEXT NOT NULL DEFAULT '',
            screen_diagnostic_detail TEXT NOT NULL DEFAULT '',
            screen_diagnostic_updated_at TEXT NOT NULL DEFAULT '',
            next_run_after TEXT NOT NULL DEFAULT '',
            last_run_at TEXT NOT NULL DEFAULT '',
            last_status TEXT NOT NULL DEFAULT '等待系统空闲',
            last_error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lounge_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_type TEXT NOT NULL,
            model_tier TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            resource_snapshot TEXT NOT NULL DEFAULT '{}',
            termination_reason TEXT NOT NULL DEFAULT '',
            continuation_decisions TEXT NOT NULL DEFAULT '[]',
            resume_source_session_id INTEGER NOT NULL DEFAULT 0,
            quality_status TEXT NOT NULL DEFAULT 'accepted',
            quality_reason TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS lounge_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lounge_session_id INTEGER NOT NULL
                REFERENCES lounge_sessions(id) ON DELETE CASCADE,
            speaker TEXT NOT NULL CHECK(speaker IN ('aili', 'shaya', 'system')),
            content TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS lounge_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lounge_session_id INTEGER NOT NULL
                REFERENCES lounge_sessions(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            excerpt TEXT NOT NULL DEFAULT '',
            fingerprint TEXT NOT NULL UNIQUE,
            error TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS lounge_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source_paths TEXT NOT NULL DEFAULT '[]',
            confidence TEXT NOT NULL DEFAULT 'observation',
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL DEFAULT '',
            lounge_session_id INTEGER NOT NULL DEFAULT 0,
            quality_status TEXT NOT NULL DEFAULT 'accepted'
        );

        CREATE TABLE IF NOT EXISTS lounge_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL CHECK(source_type IN ('message', 'note')),
            source_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(source_type, source_id, model)
        );

        CREATE TABLE IF NOT EXISTS screen_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running',
            model_tier TEXT NOT NULL DEFAULT '4b',
            aili_observation TEXT NOT NULL DEFAULT '',
            shaya_observation TEXT NOT NULL DEFAULT '',
            image_retained INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            quality_status TEXT NOT NULL DEFAULT 'accepted',
            quality_reason TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_lounge_messages_session
            ON lounge_messages(lounge_session_id, id);
        CREATE INDEX IF NOT EXISTS idx_lounge_sessions_started
            ON lounge_sessions(started_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_lounge_notes_created
            ON lounge_notes(created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_lounge_embeddings_source
            ON lounge_embeddings(source_type, source_id, model);
        CREATE INDEX IF NOT EXISTS idx_screen_observations_captured
            ON screen_observations(captured_at DESC, id DESC);
        """
    )
    session_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(lounge_sessions)")
    }
    if "topic_mode" not in session_columns:
        connection.execute(
            "ALTER TABLE lounge_sessions "
            "ADD COLUMN topic_mode TEXT NOT NULL DEFAULT 'file'"
        )
    session_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(lounge_sessions)")
    }
    session_extensions = {
        "termination_reason": "TEXT NOT NULL DEFAULT ''",
        "continuation_decisions": "TEXT NOT NULL DEFAULT '[]'",
        "resume_source_session_id": "INTEGER NOT NULL DEFAULT 0",
        "quality_status": "TEXT NOT NULL DEFAULT 'accepted'",
        "quality_reason": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in session_extensions.items():
        if name not in session_columns:
            connection.execute(
                f"ALTER TABLE lounge_sessions ADD COLUMN {name} {definition}"
            )
    note_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(lounge_notes)")
    }
    note_extensions = {
        "lounge_session_id": "INTEGER NOT NULL DEFAULT 0",
        "quality_status": "TEXT NOT NULL DEFAULT 'accepted'",
    }
    for name, definition in note_extensions.items():
        if name not in note_columns:
            connection.execute(f"ALTER TABLE lounge_notes ADD COLUMN {name} {definition}")
    observation_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(screen_observations)")
    }
    observation_extensions = {
        "quality_status": "TEXT NOT NULL DEFAULT 'accepted'",
        "quality_reason": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in observation_extensions.items():
        if name not in observation_columns:
            connection.execute(
                f"ALTER TABLE screen_observations ADD COLUMN {name} {definition}"
            )
    config_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(lounge_config)")
    }
    screen_columns = {
        "screen_watch_enabled": "INTEGER NOT NULL DEFAULT 1",
        "screen_min_interval_minutes": "INTEGER NOT NULL DEFAULT 60",
        "screen_max_interval_minutes": "INTEGER NOT NULL DEFAULT 180",
        "screen_max_daily": "INTEGER NOT NULL DEFAULT 6",
        "screen_next_run_after": "TEXT NOT NULL DEFAULT ''",
        "screen_last_run_at": "TEXT NOT NULL DEFAULT ''",
        "screen_last_status": "TEXT NOT NULL DEFAULT '等待首次屏幕观察'",
        "screen_last_error": "TEXT NOT NULL DEFAULT ''",
        "screen_manual_requested": "INTEGER NOT NULL DEFAULT 0",
        "screen_diagnostic_requested": "INTEGER NOT NULL DEFAULT 0",
        "screen_diagnostic_status": "TEXT NOT NULL DEFAULT ''",
        "screen_diagnostic_detail": "TEXT NOT NULL DEFAULT ''",
        "screen_diagnostic_updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for name, definition in screen_columns.items():
        if name not in config_columns:
            connection.execute(
                f"ALTER TABLE lounge_config ADD COLUMN {name} {definition}"
            )
    timestamp = now_text()
    connection.execute(
        """
        INSERT OR IGNORE INTO lounge_config(
            id, scan_roots, updated_at
        ) VALUES (1, ?, ?)
        """,
        (json.dumps(default_scan_roots(), ensure_ascii=False), timestamp),
    )
    row = connection.execute(
        "SELECT screen_next_run_after FROM lounge_config WHERE id = 1"
    ).fetchone()
    if row and not str(row["screen_next_run_after"] or ""):
        first_due = (
            dt.datetime.now().astimezone()
            + dt.timedelta(minutes=random.randint(60, 120))
        ).isoformat(timespec="seconds")
        connection.execute(
            "UPDATE lounge_config SET screen_next_run_after = ? WHERE id = 1",
            (first_due,),
        )
    ensure_persona_memory_pool_schema(connection)
    _quarantine_legacy_grounding_errors(connection)
    connection.commit()


def _config_dict(row: sqlite3.Row) -> dict[str, object]:
    try:
        roots = json.loads(row["scan_roots"] or "[]")
    except json.JSONDecodeError:
        roots = []
    return {
        "enabled": bool(row["enabled"]),
        "idle_minutes": int(row["idle_minutes"]),
        "min_interval_minutes": int(row["min_interval_minutes"]),
        "max_interval_minutes": int(row["max_interval_minutes"]),
        "max_daily_rounds": int(row["max_daily_rounds"]),
        "model_strategy": str(row["model_strategy"]),
        "inspect_files": bool(row["inspect_files"]),
        "scan_roots": roots if isinstance(roots, list) else [],
        "screen_watch_enabled": bool(row["screen_watch_enabled"]),
        "screen_min_interval_minutes": int(row["screen_min_interval_minutes"]),
        "screen_max_interval_minutes": int(row["screen_max_interval_minutes"]),
        "screen_max_daily": int(row["screen_max_daily"]),
        "screen_next_run_after": str(row["screen_next_run_after"]),
        "screen_last_run_at": str(row["screen_last_run_at"]),
        "screen_last_status": str(row["screen_last_status"]),
        "screen_last_error": str(row["screen_last_error"]),
        "screen_diagnostic_status": str(row["screen_diagnostic_status"]),
        "screen_diagnostic_detail": str(row["screen_diagnostic_detail"]),
        "screen_diagnostic_updated_at": str(row["screen_diagnostic_updated_at"]),
        "next_run_after": str(row["next_run_after"]),
        "last_run_at": str(row["last_run_at"]),
        "last_status": str(row["last_status"]),
        "last_error": str(row["last_error"]),
        "updated_at": str(row["updated_at"]),
    }


def get_config(connection: sqlite3.Connection) -> dict[str, object]:
    ensure_lounge_schema(connection)
    row = connection.execute("SELECT * FROM lounge_config WHERE id = 1").fetchone()
    if row is None:
        raise RuntimeError("茶话室配置未初始化")
    return _config_dict(row)


def update_config(
    connection: sqlite3.Connection, values: dict[str, object]
) -> dict[str, object]:
    current = get_config(connection)
    enabled = bool(values.get("enabled", current["enabled"]))
    idle = max(5, min(int(values.get("idle_minutes", current["idle_minutes"])), 180))
    minimum = max(
        60,
        min(int(values.get("min_interval_minutes", current["min_interval_minutes"])), 1_440),
    )
    maximum = max(
        minimum,
        min(int(values.get("max_interval_minutes", current["max_interval_minutes"])), 2_880),
    )
    daily = max(1, min(int(values.get("max_daily_rounds", current["max_daily_rounds"])), 12))
    strategy = str(values.get("model_strategy", current["model_strategy"]))
    if strategy not in {"auto", "4b", "9b"}:
        raise ValueError("后台模型策略必须是 auto、4b 或 9b")
    inspect = bool(values.get("inspect_files", current["inspect_files"]))
    screen_enabled = bool(
        values.get("screen_watch_enabled", current["screen_watch_enabled"])
    )
    screen_minimum = max(
        60,
        min(
            int(values.get("screen_min_interval_minutes", current["screen_min_interval_minutes"])),
            1_440,
        ),
    )
    screen_maximum = max(
        screen_minimum,
        min(
            int(values.get("screen_max_interval_minutes", current["screen_max_interval_minutes"])),
            2_880,
        ),
    )
    screen_daily = max(
        1, min(int(values.get("screen_max_daily", current["screen_max_daily"])), 24)
    )
    roots_value = values.get("scan_roots", current["scan_roots"])
    if isinstance(roots_value, str):
        roots = [line.strip() for line in roots_value.splitlines() if line.strip()]
    elif isinstance(roots_value, list):
        roots = [str(item).strip() for item in roots_value if str(item).strip()]
    else:
        raise ValueError("扫描位置必须是路径列表")
    if not roots:
        roots = default_scan_roots()
    connection.execute(
        """
        UPDATE lounge_config
           SET enabled = ?, idle_minutes = ?, min_interval_minutes = ?,
               max_interval_minutes = ?, max_daily_rounds = ?,
               model_strategy = ?, inspect_files = ?, scan_roots = ?,
               screen_watch_enabled = ?, screen_min_interval_minutes = ?,
               screen_max_interval_minutes = ?, screen_max_daily = ?,
               updated_at = ?
         WHERE id = 1
        """,
        (
            int(enabled), idle, minimum, maximum, daily, strategy, int(inspect),
            json.dumps(roots[:20], ensure_ascii=False), int(screen_enabled),
            screen_minimum, screen_maximum, screen_daily, now_text(),
        ),
    )
    connection.commit()
    return get_config(connection)


def _parse_time(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def system_idle_seconds() -> float:
    try:
        result = subprocess.run(
            ["/usr/sbin/ioreg", "-c", "IOHIDSystem"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', result.stdout)
        return int(match.group(1)) / 1_000_000_000 if match else 0.0
    except (OSError, subprocess.SubprocessError):
        return 0.0


def memory_free_percent() -> float:
    try:
        result = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        match = re.search(r"free percentage:\s*([0-9.]+)%", result.stdout)
        return float(match.group(1)) if match else 0.0
    except (OSError, subprocess.SubprocessError):
        return 0.0


def loaded_models() -> list[str]:
    try:
        with urllib.request.urlopen(OLLAMA_BASE + "/api/ps", timeout=2) as response:
            payload = json.load(response)
        return [str(item.get("name", "")) for item in payload.get("models", [])]
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []


def foreground_state() -> dict[str, object]:
    """读取前台应用是否覆盖一整块显示器。

    只读窗口所属者、层级和几何尺寸，不读像素、窗口文本或键盘事件，
    因此不需要屏幕录制或辅助功能权限。短缓存避免模型每一稿都启动
    一个独立的系统查询进程。
    """
    global FOREGROUND_STATE_CACHE
    now = time.monotonic()
    cached_at, cached = FOREGROUND_STATE_CACHE
    if cached and now - cached_at < FOREGROUND_STATE_TTL_SECONDS:
        return dict(cached)
    with FOREGROUND_STATE_LOCK:
        cached_at, cached = FOREGROUND_STATE_CACHE
        now = time.monotonic()
        if cached and now - cached_at < FOREGROUND_STATE_TTL_SECONDS:
            return dict(cached)
        fallback: dict[str, object] = {
            "state_known": False,
            "fullscreen_active": bool(cached.get("fullscreen_active", False)),
            "frontmost_pid": int(cached.get("frontmost_pid", 0) or 0),
            "frontmost_bundle": str(cached.get("frontmost_bundle", "")),
            "frontmost_app": str(cached.get("frontmost_app", "")),
            "frontmost_window_ratio": float(
                cached.get("frontmost_window_ratio", 0.0) or 0.0
            ),
            "frontmost_window_layer": int(
                cached.get("frontmost_window_layer", 0) or 0
            ),
        }
        try:
            result = subprocess.run(
                [
                    "/usr/bin/osascript",
                    "-l",
                    "JavaScript",
                    str(FRONTMOST_STATE_SCRIPT),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            payload = json.loads(result.stdout.strip())
            if result.returncode != 0 or not isinstance(payload, dict):
                raise RuntimeError(result.stderr.strip() or "前台状态返回无效")
            state = {
                "state_known": bool(payload.get("state_known", True)),
                "fullscreen_active": bool(payload.get("fullscreen_active", False)),
                "frontmost_pid": int(payload.get("frontmost_pid", 0) or 0),
                "frontmost_bundle": str(payload.get("frontmost_bundle", ""))[:200],
                "frontmost_app": str(payload.get("frontmost_app", ""))[:200],
                "frontmost_window_ratio": round(
                    float(payload.get("frontmost_window_ratio", 0.0) or 0.0), 3
                ),
                "frontmost_window_layer": int(
                    payload.get("frontmost_window_layer", 0) or 0
                ),
            }
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
            state = fallback
        FOREGROUND_STATE_CACHE = (now, state)
        return dict(state)


def installed_models() -> set[str]:
    """短缓存避免每一稿都请求 Ollama tags，同时允许下载完成后自动生效。"""
    global MODEL_TAG_CACHE
    cached_at, names = MODEL_TAG_CACHE
    if time.monotonic() - cached_at < 30:
        return set(names)
    try:
        with urllib.request.urlopen(OLLAMA_BASE + "/api/tags", timeout=3) as response:
            payload = json.load(response)
        names = {
            str(item.get("name", ""))
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        names = set()
    MODEL_TAG_CACHE = (time.monotonic(), names)
    return set(names)


def _quality_helper_if_available(persona: str) -> str:
    candidate = QUALITY_HELPER_MODELS[persona]
    return candidate if candidate in installed_models() else PERSONAS[persona].models["9b"]


def _generation_model_for_attempt(
    persona: str,
    primary: str,
    attempt: int,
    *,
    allow_escalation: bool = True,
) -> str:
    """同参数重写后升 9B/27B；14B 只做窄任务复核，不负责最终聊天。"""
    if attempt <= 1 or not allow_escalation:
        return primary
    own_9b = PERSONAS[persona].models["9b"]
    if attempt == 2 and primary != own_9b:
        return own_9b
    own_27b = PERSONAS[persona].models["27b"]
    return own_27b if own_27b in installed_models() else own_9b


def _compact_27b_generation_messages(
    *,
    speaker: str,
    other: str,
    topic_mode: str,
    evidence: str,
    transcript: Sequence[dict[str, str]],
    issue: str,
) -> list[dict[str, object]]:
    """27B 权重较大，用短而完整的事实包控制在 4K 上下文内。"""
    recent = "\n".join(
        f"{PERSONAS[item['speaker']].name}：{item['content']}"
        for item in transcript[-8:]
    )
    system = (
        PERSONAS[speaker].system_prompt
        + "\n\n【质量升档重写】"
        + LOUNGE_WORLD_MODEL
        + f"当前只给{PERSONAS[other].name}发一条自然短消息，主人不是收件人。"
        "严格依据证据和人物标签；不得把主人的电脑行为归给任何人格，"
        "不得补写截图、文件或旧记忆没有证明的动作、动机和心理。"
        "像微信或QQ消息，15到80字、最多三句，无标题、列表、舞台动作和多人对白。"
        f"模式：{topic_mode}。上一稿问题：{issue or '9B未达到质量门'}。\n\n"
        f"证据：\n{evidence[:1_800] or '(无)'}\n\n"
        f"当前对话：\n{recent[-1_200:] or '(尚未开始)'}"
    )
    return [
        {"role": "system", "content": system[:3_400]},
        {
            "role": "user",
            "content": (
                "【本地程序轮转，不是主人发言】"
                f"只输出{PERSONAS[speaker].name}发给{PERSONAS[other].name}的下一条气泡。"
            ),
        },
    ]


def resource_snapshot() -> dict[str, object]:
    cores = max(1, os.cpu_count() or 1)
    try:
        load_1m = float(os.getloadavg()[0])
    except OSError:
        load_1m = float(cores)
    foreground = foreground_state()
    return {
        "memory_free_percent": round(memory_free_percent(), 1),
        "load_1m": round(load_1m, 2),
        "load_ratio": round(load_1m / cores, 3),
        "cpu_count": cores,
        "system_idle_seconds": round(system_idle_seconds(), 1),
        "app_idle_seconds": round(seconds_since_user_activity(), 1),
        "loaded_models": loaded_models(),
        **foreground,
        "sampled_at": now_text(),
    }


def _today_completed_count(connection: sqlite3.Connection) -> int:
    prefix = dt.datetime.now().astimezone().date().isoformat() + "%"
    row = connection.execute(
        """
        SELECT COUNT(*) AS count FROM lounge_sessions
         WHERE started_at LIKE ? AND status IN ('completed', 'interrupted')
           AND trigger_type != 'screen'
        """,
        (prefix,),
    ).fetchone()
    return int(row["count"] if row else 0)


def _connection_database_path(connection: sqlite3.Connection) -> str | None:
    """取出当前茶话会话实际所属的 SQLite 文件。"""
    try:
        rows = connection.execute("PRAGMA database_list").fetchall()
    except sqlite3.Error:
        return None
    for row in rows:
        # sqlite3.Row 和普通 tuple 都兼容。
        name = str(row[1])
        if name == "main":
            value = str(row[2] or "").strip()
            return value or None
    return None


def _ultimate_background_ready(
    connection: sqlite3.Connection, minimum_tokens: int = 4_000
) -> bool:
    """究极后台额度还够不够撑一轮茶话；不够就当作只能用本地模型。"""
    if not ultimate_available():
        return False
    try:
        remaining = int(
            (
                ultimate_usage_summary(_connection_database_path(connection)).get(
                    "background"
                )
                or {}
            ).get("remaining", 0)
        )
    except Exception:
        return False
    return remaining >= minimum_tokens


def evaluate_eligibility(
    connection: sqlite3.Connection,
    *,
    manual: bool = False,
    snapshot: dict[str, object] | None = None,
) -> tuple[bool, str, str, dict[str, object]]:
    config = get_config(connection)
    snapshot = snapshot or resource_snapshot()
    if not config["enabled"] and not manual:
        return False, "已在客户端关闭", "4b", snapshot
    if any("27b" in name.lower() for name in snapshot["loaded_models"]):
        return False, "27B 模型正在使用，不抢占内存", "4b", snapshot
    memory = float(snapshot["memory_free_percent"])
    load_ratio = float(snapshot["load_ratio"])
    fullscreen = bool(snapshot.get("fullscreen_active", False))
    if fullscreen and not manual:
        app_name = str(snapshot.get("frontmost_app", "")).strip()
        suffix = f"（{app_name}）" if app_name else ""
        return False, f"全屏应用使用中{suffix}，暂停后台茶话", "4b", snapshot
    # 主人在全屏状态下手动开聊也只能走低负载档，
    # 并使用比普通手动模式更严的 CPU/内存门槛。
    minimum_memory = 58.0 if fullscreen else (35.0 if manual else 50.0)
    maximum_load = 0.22 if fullscreen else (0.65 if manual else 0.28)
    if memory < minimum_memory:
        return False, f"可用内存仅 {memory:.0f}%", "4b", snapshot
    if load_ratio > maximum_load:
        return False, f"CPU 负载偏高（{load_ratio * 100:.0f}%）", "4b", snapshot
    if not manual:
        idle_needed = int(config["idle_minutes"]) * 60
        actual_idle = min(
            float(snapshot["system_idle_seconds"]),
            float(snapshot["app_idle_seconds"]),
        )
        if actual_idle < idle_needed:
            return False, f"等待空闲 {int(config['idle_minutes'])} 分钟", "4b", snapshot
        next_run = _parse_time(str(config["next_run_after"]))
        if next_run and dt.datetime.now().astimezone() < next_run:
            return False, f"下轮最早 {next_run.strftime('%m-%d %H:%M')}", "4b", snapshot
        if _today_completed_count(connection) >= int(config["max_daily_rounds"]):
            return False, "今日自主轮次已达上限", "4b", snapshot
    strategy = str(config["model_strategy"])
    if fullscreen:
        tier = "4b"
    elif strategy == "auto":
        # 24GB M5 Pro 的空闲状态默认把 9B 当质量基线；4B 只在资源确实
        # 收紧时降级为应急档，不再因为过于保守的 72% 门槛长期占主流程。
        tier = "9b" if memory >= 58.0 and load_ratio <= 0.22 else "4b"
    else:
        tier = strategy
    if tier == "9b" and memory < 58.0:
        tier = "4b"
    # 究极额度用尽后一切都落回本地：这时 4B 写不出能过质量门的内容，
    # 只会产出“嗯、确实”这类附和。与其凑合聊，不如等资源宽裕再说。
    if tier == "4b" and not _ultimate_background_ready(connection):
        return (
            False,
            "究极额度已用尽，本地 4B 达不到茶话质量要求，等内存宽裕用 9B 再聊",
            tier,
            snapshot,
        )
    return True, f"资源宽裕，可使用 {tier.upper()}", tier, snapshot


def _set_status(
    connection: sqlite3.Connection, status: str, error: str = ""
) -> None:
    row = connection.execute(
        "SELECT last_status, last_error FROM lounge_config WHERE id = 1"
    ).fetchone()
    if row and row["last_status"] == status and row["last_error"] == error:
        return
    connection.execute(
        """
        UPDATE lounge_config
           SET last_status = ?, last_error = ?, updated_at = ? WHERE id = 1
        """,
        (status, error, now_text()),
    )
    connection.commit()


def _screen_today_count(connection: sqlite3.Connection) -> int:
    prefix = dt.datetime.now().astimezone().date().isoformat() + "%"
    row = connection.execute(
        """
        SELECT COUNT(*) AS count FROM screen_observations
         WHERE captured_at LIKE ? AND status = 'completed'
           AND quality_status = 'accepted'
        """,
        (prefix,),
    ).fetchone()
    return int(row["count"] if row else 0)


def screen_watch_eligibility(
    connection: sqlite3.Connection,
    *,
    manual: bool = False,
    snapshot: dict[str, object] | None = None,
) -> tuple[bool, str, dict[str, object]]:
    """屏幕观察看的是主人正在操作的时刻，因此不要求系统空闲。"""
    config = get_config(connection)
    snapshot = snapshot or resource_snapshot()
    if not bool(config["screen_watch_enabled"]) and not manual:
        return False, "屏幕观察已关闭", snapshot
    if RUN_LOCK.locked():
        return False, "后台模型正在忙", snapshot
    if bool(snapshot.get("fullscreen_active", False)):
        app_name = str(snapshot.get("frontmost_app", "")).strip()
        suffix = f"（{app_name}）" if app_name else ""
        return False, f"全屏应用使用中{suffix}，暂停屏幕观察", snapshot
    if any("27b" in name.lower() for name in snapshot["loaded_models"]):
        return False, "27B 模型正在使用", snapshot
    memory = float(snapshot["memory_free_percent"])
    load_ratio = float(snapshot["load_ratio"])
    if memory < 55.0:
        return False, f"可用内存仅 {memory:.0f}%", snapshot
    if load_ratio > 0.32:
        return False, f"CPU 负载偏高（{load_ratio * 100:.0f}%）", snapshot
    if not manual:
        next_run = _parse_time(str(config["screen_next_run_after"]))
        if next_run and dt.datetime.now().astimezone() < next_run:
            return False, f"下次观察 {next_run.strftime('%m-%d %H:%M')}", snapshot
        if _screen_today_count(connection) >= int(config["screen_max_daily"]):
            return False, "今日屏幕观察已达上限", snapshot
        # 主人长时间离开时没有观察价值，等主人回来再抓取新鲜画面。
        if float(snapshot["system_idle_seconds"]) > 20 * 60:
            return False, "等待主人回到电脑前", snapshot
    return True, "可以观察当前屏幕", snapshot


def request_screen_watch_now(connection: sqlite3.Connection) -> dict[str, object]:
    ensure_lounge_schema(connection)
    connection.execute(
        """
        UPDATE lounge_config
           SET screen_manual_requested = 1, screen_next_run_after = '',
               screen_last_status = '等待原生客户端截取瞬时画面',
               screen_last_error = '', updated_at = ?
         WHERE id = 1
        """,
        (now_text(),),
    )
    connection.commit()
    return {"requested": True}


def request_screen_capture_diagnostic(
    connection: sqlite3.Connection,
) -> dict[str, object]:
    """只验证原生权限与 ScreenCaptureKit 截图链路。

    这个请求不经过资源门、不调用模型，也不写入屏幕观察历史。
    """
    ensure_lounge_schema(connection)
    timestamp = now_text()
    connection.execute(
        """
        UPDATE lounge_config
           SET screen_manual_requested = 1,
               screen_diagnostic_requested = 1,
               screen_diagnostic_status = 'waiting',
               screen_diagnostic_detail = '等待原生客户端响应（通常不超过 30 秒）',
               screen_diagnostic_updated_at = ?,
               screen_last_status = '正在检测屏幕权限与截图链路',
               screen_last_error = '', updated_at = ?
         WHERE id = 1
        """,
        (timestamp, timestamp),
    )
    connection.commit()
    return {"requested": True, "status": "waiting", "updated_at": timestamp}


def _set_screen_diagnostic_result(
    connection: sqlite3.Connection,
    status: str,
    detail: str,
) -> None:
    timestamp = now_text()
    failed = status == "failed"
    connection.execute(
        """
        UPDATE lounge_config
           SET screen_diagnostic_requested = 0,
               screen_diagnostic_status = ?,
               screen_diagnostic_detail = ?,
               screen_diagnostic_updated_at = ?,
               screen_last_status = ?,
               screen_last_error = ?, updated_at = ?
         WHERE id = 1
        """,
        (
            status,
            detail[:1_000],
            timestamp,
            "屏幕权限检测失败" if failed else "屏幕权限与截图链路正常",
            detail[:1_000] if failed else "",
            timestamp,
        ),
    )


def claim_screen_watch(connection: sqlite3.Connection) -> dict[str, object]:
    """供原生客户端低频轮询；只返回一次性 request_id，不返回历史图像。"""
    ensure_lounge_schema(connection)
    with SCREEN_REQUEST_LOCK:
        if SCREEN_PENDING_REQUEST:
            age = time.monotonic() - float(
                SCREEN_PENDING_REQUEST.get("claimed_monotonic", 0.0)
            )
            if age < SCREEN_REQUEST_TTL_SECONDS:
                return {
                    "due": False,
                    "reason": "等待客户端提交瞬时截图",
                    "pending": True,
                }
            expired = dict(SCREEN_PENDING_REQUEST)
            SCREEN_PENDING_REQUEST.clear()
            if bool(expired.get("diagnostic")):
                _set_screen_diagnostic_result(
                    connection,
                    "failed",
                    "原生客户端已领取检测请求，但 180 秒内没有交回截图结果。",
                )
                connection.commit()
        config = get_config(connection)
        request_row = connection.execute(
            """
            SELECT screen_manual_requested, screen_diagnostic_requested
              FROM lounge_config WHERE id = 1
            """
        ).fetchone()
        manual = bool(request_row["screen_manual_requested"])
        diagnostic = bool(request_row["screen_diagnostic_requested"])
        if diagnostic:
            eligible, reason, snapshot = True, "正在检测屏幕权限与截图链路", resource_snapshot()
        else:
            eligible, reason, snapshot = screen_watch_eligibility(
                connection, manual=manual
            )
        if not eligible:
            if manual:
                connection.execute(
                    """
                    UPDATE lounge_config SET screen_last_status = ?, updated_at = ?
                     WHERE id = 1
                    """,
                    (reason, now_text()),
                )
                connection.commit()
            return {"due": False, "reason": reason, "resources": snapshot}
        request_id = uuid.uuid4().hex
        captured_at = now_text()
        SCREEN_PENDING_REQUEST.update(
            {
                "request_id": request_id,
                "claimed_monotonic": time.monotonic(),
                "captured_at": captured_at,
                "manual": manual,
                "diagnostic": diagnostic,
            }
        )
        connection.execute(
            """
            UPDATE lounge_config
               SET screen_manual_requested = 0,
                   screen_diagnostic_requested = 0,
                   screen_diagnostic_status = CASE
                       WHEN ? THEN 'capturing' ELSE screen_diagnostic_status END,
                   screen_diagnostic_detail = CASE
                       WHEN ? THEN '原生客户端已响应，正在读取系统权限并截取一帧'
                       ELSE screen_diagnostic_detail END,
                   screen_diagnostic_updated_at = CASE
                       WHEN ? THEN ? ELSE screen_diagnostic_updated_at END,
                   screen_last_status = ?,
                   screen_last_error = '', updated_at = ?
             WHERE id = 1
            """,
            (
                diagnostic,
                diagnostic,
                diagnostic,
                captured_at,
                "正在检测屏幕权限与截图链路"
                if diagnostic
                else "正在从原生客户端读取瞬时画面",
                captured_at,
            ),
        )
        connection.commit()
        return {
            "due": True,
            "request_id": request_id,
            "captured_at": captured_at,
            "resources": snapshot,
        }


def _consume_screen_request(request_id: str) -> dict[str, object] | None:
    with SCREEN_REQUEST_LOCK:
        if str(SCREEN_PENDING_REQUEST.get("request_id", "")) != request_id:
            return None
        request = dict(SCREEN_PENDING_REQUEST)
        SCREEN_PENDING_REQUEST.clear()
        return request


def _record_screen_failure(
    connection: sqlite3.Connection,
    captured_at: str,
    error: str,
    *,
    metadata: dict[str, object] | None = None,
) -> int:
    finished = now_text()
    cursor = connection.execute(
        """
        INSERT INTO screen_observations(
            captured_at, finished_at, status, image_retained, metadata, error
        ) VALUES (?, ?, 'failed', 0, ?, ?)
        """,
        (
            captured_at,
            finished,
            json.dumps(metadata or {}, ensure_ascii=False),
            error[:1_000],
        ),
    )
    permission_blocked = bool(
        re.search(r"权限|隐私与安全性|录屏|Screen Recording|not permitted", error, re.IGNORECASE)
    )
    retry_minutes = 720 if permission_blocked else 30
    retry = (
        dt.datetime.now().astimezone() + dt.timedelta(minutes=retry_minutes)
    ).isoformat(timespec="seconds")
    status = (
        "等待主人授予录屏权限；12 小时后再检查"
        if permission_blocked
        else "屏幕观察失败，30 分钟后再试"
    )
    connection.execute(
        """
        UPDATE lounge_config
           SET screen_next_run_after = ?, screen_last_status = ?,
               screen_last_error = ?, updated_at = ? WHERE id = 1
        """,
        (retry, status, error[:1_000], finished),
    )
    cutoff = (
        dt.datetime.now().astimezone() - dt.timedelta(days=90)
    ).isoformat(timespec="seconds")
    connection.execute(
        "DELETE FROM screen_observations WHERE status != 'running' AND captured_at < ?",
        (cutoff,),
    )
    connection.commit()
    return int(cursor.lastrowid)


def submit_screen_capture_error(
    connection: sqlite3.Connection, request_id: str, error: str
) -> dict[str, object]:
    request = _consume_screen_request(request_id)
    if request is None:
        raise ValueError("屏幕观察请求已过期")
    if bool(request.get("diagnostic")):
        detail = error or "原生客户端未能读取屏幕"
        _set_screen_diagnostic_result(connection, "failed", detail)
        connection.commit()
        return {"accepted": False, "diagnostic": True, "error": detail}
    record_id = _record_screen_failure(
        connection,
        str(request.get("captured_at") or now_text()),
        error or "原生客户端未能读取屏幕",
        metadata={"capture_stage": True, "image_retained": False},
    )
    return {"accepted": False, "screen_observation_id": record_id}


def _normalize_screen_observation(answer: str) -> str:
    value = answer.strip().strip('"\'“”').strip()
    value = re.sub(r"(?:^|\n)\s*(?:[-*#]|\d+[.)、])\s*", " ", value)
    value = re.sub(r"\s*\n+\s*", " ", value)
    value = re.sub(r"[ \t]{2,}", " ", value).strip()
    return value[:1_000]


def _screen_observation_issue(answer: str) -> str:
    value = answer.strip()
    if not value:
        return "观察结果为空"
    if re.search(r"无法(?:查看|看到|访问)|看不到(?:图片|屏幕)|没有(?:图片|视觉)", value):
        return "模型没有使用视觉输入"
    if re.search(r"请(?:上传|提供)(?:截图|图片)|作为AI", value, re.IGNORECASE):
        return "模型拒绝了已经提供的本地图像"
    if len(value) < 20:
        return "观察过短，缺少可检索信息"
    if re.search(r"^(?:啧|哈|喂|刚才|这双屏|这画面)", value):
        return "屏幕记录应直接写可见事实，不要用人格表演或叙事开场"
    if re.search(
        r"怕是|看来|应该|估计|八成|仿佛|似乎|大概|可能|多半|"
        r"看起来|感觉|猜|像是|是不是|难道",
        value,
    ):
        return "屏幕长期记忆只能保存可见事实，不保存猜测语句"
    asserted_value = re.sub(
        r"(?:无法|不能|不足以).{0,12}(?:判断|证明).{0,12}主人.{0,24}",
        "",
        value,
    )
    if re.search(
        r"主人.{0,16}(?:正在|正|刚|一边|盯|编辑|调整|调试|整理|"
        r"核对|折腾|发呆|操作|处理|运行|选择|纠结|准备|打算|想)",
        asserted_value,
    ):
        return "截图只能证明窗口和文字可见，不能记成主人正在执行某个动作"
    if re.search(r"心思|心情|活泛|有条理|喜欢|在意|喘气缝", value):
        return "截图不能证明主人的心理、性格或动机"
    if re.search(r"被(?:艾莉|沙雅)拉过来|刚才在茶话室|正聊着", value):
        return "视觉事实抽取不能混入旧茶话或人格叙事"
    if re.search(r"(?:你|用户)(?:正在|在|刚刚|刚才)|你的(?:屏幕|桌面|文件)", value):
        return "屏幕操作者必须称为主人，不能写成你或用户"
    if re.search(r"没睡好|睡不着|睡眠不足|疲惫|焦虑|发愁|笑得前仰后合", value):
        return "截图不能证明主人的心理或身体状态"
    if re.search(
        r"(?:主人)?(?:可能|大概|多半|似乎|应该|看起来).{0,28}"
        r"(?:整理|担心|准备|打算|想|正在|刚|看完|处理)"
        r"|(?:是不是|难道).{0,32}[？?]",
        value,
    ):
        return "屏幕长期记忆只保存可见事实，不能把动作或动机猜测写入记忆池"
    if _grounding_violation(value, "screen"):
        return _grounding_violation(value, "screen")
    return ""


def _screen_prompt(
    connection: sqlite3.Connection,
    persona: str,
    captured_at: str,
    display_count: int = 1,
) -> str:
    del connection
    return (
        "【屏幕事实抽取】你只是一个无记忆、无人格表演的视觉证据记录器。"
        f"本次文字记录将归入{PERSONAS[persona].name}的记忆池，但不要模仿她的语气。"
        "主人明确允许你偶尔看一眼当前屏幕。这张图只存在于本次内存请求，"
        "程序不会把图像保存到磁盘；你只留下自己的文字经历。"
        f"本次一共提供 {display_count} 个显示器的同时画面，必须综合查看全部画面。"
        "从零开始阅读这次图像，不得引用、延续或猜测任何旧屏幕、旧茶话和旧记忆。"
        "只写画面直接显示的应用、窗口、文件和可辨认文字。必须用‘画面显示’"
        "‘窗口中可见’‘无法从截图判断’这类证据语气，不要写‘主人正在’。"
        "禁止猜测主人的动作、任务、心理、动机或屏幕外事件；禁止使用‘怕是、看来、"
        "应该、估计、八成、像是、似乎、心思’等推测词。"
        "能读到账号、密钥或其他细节时可如实记录，但绝不能编造看不清的文字。"
        "画面操作者始终写成‘主人’，禁止使用‘你正在’‘用户正在’或把操作写成"
        "艾莉/沙雅做的。不要向主人说话，不要给建议，不要用‘啧’等口癖或比喻，"
        "不写报告标题、Markdown 或列表；只输出一段 60 到 180 字的简洁事实记录。"
        f"观察时间：{captured_at}"
    )


def _screen_observation_overlap(
    connection: sqlite3.Connection, persona: str, answer: str
) -> float:
    rows = connection.execute(
        """
        SELECT content FROM persona_experiences
         WHERE persona = ? AND source_type = 'screen_observation'
           AND status = 'active'
         ORDER BY occurred_at DESC, id DESC LIMIT 8
        """,
        (persona,),
    ).fetchall()
    return max(
        (memory_text_overlap(answer, str(row["content"])) for row in rows),
        default=0.0,
    )


def _screen_capture_fingerprint(images: Sequence[str]) -> str:
    """只保存截图内容的不可逆摘要，不保存或解码图像。"""
    digest = hashlib.sha256()
    for value in images:
        digest.update(value.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def _matching_screen_fingerprint(
    connection: sqlite3.Connection, fingerprint: str
) -> int:
    rows = connection.execute(
        "SELECT id, metadata FROM screen_observations "
        "WHERE status = 'completed' AND quality_status = 'accepted' "
        "ORDER BY id DESC LIMIT 24"
    ).fetchall()
    for row in rows:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(metadata, dict) and metadata.get("capture_fingerprint") == fingerprint:
            return int(row["id"])
    return 0


def _schedule_next_screen_watch(
    connection: sqlite3.Connection, finished: str, status_prefix: str
) -> tuple[str, int]:
    row = connection.execute(
        "SELECT screen_min_interval_minutes, screen_max_interval_minutes "
        "FROM lounge_config WHERE id = 1"
    ).fetchone()
    minimum = max(60, int(row["screen_min_interval_minutes"] if row else 60))
    maximum = max(
        minimum, int(row["screen_max_interval_minutes"] if row else minimum)
    )
    interval = random.randint(minimum, maximum)
    next_run = (
        dt.datetime.now().astimezone() + dt.timedelta(minutes=interval)
    ).isoformat(timespec="seconds")
    connection.execute(
        """
        UPDATE lounge_config
           SET screen_last_run_at = ?, screen_next_run_after = ?,
               screen_last_status = ?, screen_last_error = '', updated_at = ?
         WHERE id = 1
        """,
        (
            finished,
            next_run,
            f"{status_prefix}，下次最早 {interval} 分钟后",
            finished,
        ),
    )
    return next_run, interval


def _run_screen_discussion(
    connection: sqlite3.Connection,
    *,
    screen_observation_id: int,
    captured_at: str,
    visual_observations: dict[str, str],
    snapshot: dict[str, object],
    chat_lock: threading.RLock,
    chat_activity_marker: float,
    used_models: set[str],
    display_count: int = 1,
) -> dict[str, object]:
    """两人看过同一批显示器画面后，用各自质量模型讨论并独立决定结束。"""
    tier = "9b"
    cursor = connection.execute(
        """
        INSERT INTO lounge_sessions(
            trigger_type, model_tier, topic_mode, started_at,
            status, resource_snapshot
        ) VALUES ('screen', ?, 'screen', ?, 'running', ?)
        """,
        (tier, captured_at, json.dumps(snapshot, ensure_ascii=False)),
    )
    session_id = int(cursor.lastrowid)
    connection.commit()
    persona_rows = {
        persona: get_persona_memory(connection, persona)
        for persona in ("aili", "shaya")
    }
    models = {
        persona: PERSONAS[persona].models[tier] for persona in ("aili", "shaya")
    }
    used_models.update(models.values())
    evidence = (
        f"截图时间：{captured_at}\n"
        f"程序同时交给艾莉和沙雅查看了主人的 {display_count} 个显示器画面。\n"
        f"艾莉的视觉记录：{visual_observations['aili']}\n"
        f"沙雅的视觉记录：{visual_observations['shaya']}"
    )
    owner_names = _known_owner_names(
        [str(row["memory"] or "") for row in persona_rows.values()]
    )
    transcript: list[dict[str, str]] = []
    decisions: list[dict[str, object]] = []
    running_summary = ""
    summarized_count = 0
    speaker = "aili"
    turn = 0
    termination_reason = ""
    try:
        while True:
            turn += 1
            safe, stop_reason = _still_safe(chat_activity_marker, manual=False)
            if not safe:
                raise InterruptedError(stop_reason)
            other = "shaya" if speaker == "aili" else "aili"
            model = models[speaker]
            system = (
                str(persona_rows[speaker]["system_prompt"])
                + "\n\n【共同屏幕观察茶话，优先级高于普通对话习惯】"
                + LOUNGE_WORLD_MODEL
                + f"你已经与{PERSONAS[other].name}亲眼看过同一张主人屏幕截图。"
                "现在只和另一个人格讨论这份共同证据，不是在向主人汇报。"
                "可以聊画面里真正值得注意的内容、联系各自旧记忆、表达不同观点，"
                "也可以自然跑到相关话题；但明示和推测要分清。"
                "必须说‘主人正在/主人的屏幕’，绝不能说‘你的屏幕/你在操作’，"
                "也不能回答成自己正在使用 Safari、整理截图或操作文件。"
                "若画面出现星语茶话屋、旧名 Local AI Studio 或相关文档，要意识到那就是你们所在的"
                "应用和自身功能。不要用‘主人不在’开场。"
                "像微信或 QQ 私聊，每条通常15到80个中文字、最多三句；不写标题、"
                "列表、舞台动作或小作文，不要复述对方刚说过的整组名词。"
                "不写内部思考。\n\n"
                f"共同视觉证据：\n{evidence[:4_500]}\n\n"
                f"你自己的动态自我档案：\n{persona_rows[speaker]['profile'] or '(暂无)'}\n\n"
                f"更早内容摘要：\n{running_summary or '(无)'}"
            )
            recent = transcript[max(summarized_count, len(transcript) - 12):]
            if recent:
                system += (
                    "\n\n【当前茶话原文；姓名标签与发言归属是不可改写的事实】\n"
                    + "\n".join(
                        f"{PERSONAS[item['speaker']].name}：{item['content']}"
                        for item in recent
                    )
                )
            messages: list[dict[str, object]] = [
                {"role": "system", "content": system[:15_000]},
                {
                    "role": "user",
                    "content": (
                        "【本地程序内部轮转指令，不是主人发言】"
                        f"现在轮到{PERSONAS[speaker].name}只用自己的身份，"
                        f"给{PERSONAS[other].name}发下一条消息。"
                        + (
                            "承接上面最后一句，不得把它误认成主人说的。"
                            if recent
                            else "你们已经共同看完截图，挑一个真正值得说的点自然开场。"
                        )
                    ),
                },
            ]
            answer = ""
            issue = ""
            actual_model = model
            finish_reason = "stop"
            allow_question = not any(
                re.search(r"[？?]", item["content"]) for item in transcript[-2:]
            )
            for attempt in range(4):
                retry_messages = [dict(item) for item in messages]
                if attempt:
                    retry_messages.insert(
                        1,
                        {
                            "role": "system",
                            "content": (
                                f"上一稿未通过事实与对话质量门：{issue}。"
                                "重新核对主人/艾莉/沙雅的身份和共同截图，再写一条"
                                "有承接、有新内容的自然短消息。"
                            ),
                        },
                    )
                actual_model = _generation_model_for_attempt(
                    speaker, model, attempt
                )
                used_models.add(actual_model)
                if actual_model.endswith("27b"):
                    retry_messages = _compact_27b_generation_messages(
                        speaker=speaker,
                        other=other,
                        topic_mode="screen",
                        evidence=evidence,
                        transcript=transcript,
                        issue=issue,
                    )
                # metrics 里会带回真正由谁生成（究极或本地），用于如实记录来源。
                generation_metrics: dict[str, object] = {}
                answer, finish_reason = call_ollama(
                    actual_model,
                    retry_messages,
                    replace(config_for_model(actual_model), num_predict=160),
                    max_output=160,
                    temperature=(0.72 if speaker == "aili" else 0.58)
                    if attempt == 0 else 0.32,
                    top_p=0.88,
                    repeat_penalty=1.1,
                    think=False,
                    keep_alive="0",
                    metrics=generation_metrics,
                )
                served_by = "ultimate" if generation_metrics.get("ultimate") else "local"
                answer = _normalize_lounge_answer(answer, other)
                answer = _resolve_persona_pronouns(answer, other, speaker)
                if not allow_question:
                    answer = _strip_questions_from_reply(answer)
                else:
                    answer = _limit_reply_to_one_question(answer)
                issue = _lounge_answer_issue(
                    answer,
                    speaker=speaker,
                    other=other,
                    owner_names=owner_names,
                    turn=turn,
                    topic_mode="screen",
                    allow_question=allow_question,
                    prior_messages=[item["content"] for item in transcript],
                )
                if not issue:
                    issue = _review_grounded_lounge_answer(
                        speaker=speaker,
                        other=other,
                        topic_mode="screen",
                        evidence=evidence,
                        transcript=transcript,
                        candidate=answer,
                        model=_quality_helper_if_available(speaker),
                        chat_lock=chat_lock,
                    )
                if not issue:
                    break
            if issue:
                raise RuntimeError(
                    f"{PERSONAS[speaker].name}连续 4 次未通过屏幕讨论质量门：{issue}"
                )
            created_at = now_text()
            message = {"speaker": speaker, "content": answer[:1_500]}
            transcript.append(message)
            connection.execute(
                """
                INSERT INTO lounge_messages(
                    lounge_session_id, speaker, content, model, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    speaker,
                    answer[:1_500],
                    actual_model,
                    created_at,
                    json.dumps(
                        {
                            "turn": turn,
                            "finish_reason": finish_reason,
                            "primary_model": model,
                            "fallback": actual_model != model,
                            "served_by": served_by,
                            "source_event": "screen_observation",
                            "screen_observation_id": screen_observation_id,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            connection.commit()
            if len(transcript) - summarized_count > 14:
                summarize_to = len(transcript) - 10
                running_summary = _update_lounge_running_summary(
                    running_summary,
                    transcript[summarized_count:summarize_to],
                    models["aili"],
                    chat_lock,
                )
                summarized_count = summarize_to
            # 屏幕讨论也先保证至少 5 轮完整互答，再让 LLM
            # 判断是否自然收尾，避免刚开题就停。
            if turn % 2 == 0 and turn >= MIN_LOUNGE_MESSAGES_BEFORE_DECISION:
                pair: list[dict[str, object]] = []
                for decider in ("aili", "shaya"):
                    pair.append(
                        _lounge_continue_decision(
                            decider,
                            models[decider],
                            str(persona_rows[decider]["system_prompt"]),
                            transcript,
                            running_summary,
                            chat_lock,
                        )
                    )
                decisions.append({"after_turn": turn, "at": now_text(), "decisions": pair})
                ending = [item for item in pair if item["decision"] == "END"]
                if ending:
                    names = "、".join(str(item["persona_name"]) for item in ending)
                    termination_reason = (
                        "双方选择自然收尾"
                        if len(ending) == 2
                        else f"{names}选择自然收尾"
                    )
                    break
            speaker = other
        chat_note = _create_shared_chat_memory(
            connection,
            session_id,
            transcript,
            get_shared_chat_memory(connection),
            models["aili"],
            chat_lock,
            running_summary=running_summary,
        )
        record_lounge_round_in_persona_pools(
            connection,
            session_id,
            transcript,
            [],
            "",
            chat_note,
            occurred_at=captured_at,
        )
        finished = now_text()
        connection.execute(
            """
            UPDATE lounge_sessions
               SET finished_at = ?, status = 'completed', summary = ?,
                   termination_reason = ?, continuation_decisions = ?
             WHERE id = ?
            """,
            (
                finished,
                (chat_note or "围绕主人当时的屏幕完成了一轮交流")[:2_000],
                termination_reason,
                json.dumps(decisions, ensure_ascii=False),
                session_id,
            ),
        )
        connection.commit()
        return {
            "completed": True,
            "session_id": session_id,
            "messages": len(transcript),
            "termination_reason": termination_reason,
            "continuation_decisions": decisions,
        }
    except InterruptedError as error:
        connection.execute(
            """
            UPDATE lounge_sessions
               SET finished_at = ?, status = 'interrupted', summary = ?,
                   termination_reason = ?, continuation_decisions = ?
             WHERE id = ?
            """,
            (
                now_text(), str(error), str(error),
                json.dumps(decisions, ensure_ascii=False), session_id,
            ),
        )
        connection.commit()
        return {
            "completed": False,
            "interrupted": True,
            "session_id": session_id,
            "messages": len(transcript),
            "reason": str(error),
        }
    except Exception as error:
        connection.execute(
            "UPDATE lounge_sessions SET finished_at = ?, status = 'failed', summary = ? "
            "WHERE id = ?",
            (now_text(), str(error)[:1_000], session_id),
        )
        connection.commit()
        return {
            "completed": False,
            "session_id": session_id,
            "messages": len(transcript),
            "reason": str(error),
        }


def run_screen_observation(
    database_path: str,
    chat_lock: threading.RLock,
    *,
    request: dict[str, object],
    image_base64: str | Sequence[str],
    image_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """多显示器图像只存在于调用栈；数据库永远只写两份文字观察。"""
    images = (
        [str(item) for item in image_base64 if str(item)]
        if not isinstance(image_base64, str)
        else ([image_base64] if image_base64 else [])
    )
    if not images:
        raise ValueError("瞬时截图为空")
    captured_at = str(request.get("captured_at") or now_text())
    if not RUN_LOCK.acquire(blocking=False):
        connection = open_database(database_path)
        try:
            record_id = _record_screen_failure(
                connection, captured_at, "后台任务正在使用模型，本次瞬时画面已丢弃"
            )
        finally:
            connection.close()
        return {"completed": False, "screen_observation_id": record_id}
    connection = open_database(database_path)
    used_models: set[str] = set()
    record_id: int | None = None
    acquired_chat = False
    try:
        ensure_lounge_schema(connection)
        snapshot = resource_snapshot()
        if bool(snapshot.get("fullscreen_active", False)):
            # 请求被原生客户端领取后，前台可能才切入全屏。
            # 在任何视觉模型加载之前做第二道门，直接释放瞬时图像。
            app_name = str(snapshot.get("frontmost_app", "")).strip()
            next_check = (
                dt.datetime.now().astimezone() + dt.timedelta(minutes=10)
            ).isoformat(timespec="seconds")
            connection.execute(
                """
                UPDATE lounge_config
                   SET screen_next_run_after = ?,
                       screen_last_status = ?, screen_last_error = '', updated_at = ?
                 WHERE id = 1
                """,
                (
                    next_check,
                    f"全屏应用使用中{f'（{app_name}）' if app_name else ''}，已释放截图并延后观察",
                    now_text(),
                ),
            )
            connection.commit()
            return {
                "completed": False,
                "deferred": True,
                "reason": "全屏应用使用中，未加载视觉模型",
                "image_retained": False,
            }
        capture_fingerprint = _screen_capture_fingerprint(images)
        matching_record_id = _matching_screen_fingerprint(
            connection, capture_fingerprint
        )
        cursor = connection.execute(
            """
            INSERT INTO screen_observations(
                captured_at, status, model_tier, image_retained, metadata
            ) VALUES (?, 'running', '9b', 0, ?)
            """,
            (
                captured_at,
                json.dumps(
                    {
                        **(image_metadata or {}),
                        "resources": snapshot,
                        "capture_fingerprint": capture_fingerprint,
                        "image_retained": False,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        record_id = int(cursor.lastrowid)
        if matching_record_id:
            finished = now_text()
            reason = f"与屏幕观察 #{matching_record_id} 的截图完全相同，未重复入池"
            connection.execute(
                """
                UPDATE screen_observations
                   SET finished_at = ?, status = 'duplicate', image_retained = 0,
                       quality_status = 'duplicate', quality_reason = ?
                 WHERE id = ?
                """,
                (finished, reason, record_id),
            )
            next_run, _ = _schedule_next_screen_watch(
                connection, finished, "画面没有变化，已跳过重复观察"
            )
            connection.commit()
            return {
                "completed": False,
                "duplicate": True,
                "screen_observation_id": record_id,
                "matching_screen_observation_id": matching_record_id,
                "next_run_after": next_run,
                "image_retained": False,
            }
        connection.execute(
            """
            UPDATE lounge_config SET screen_last_status = '艾莉和沙雅正在看屏幕',
                   screen_last_error = '', updated_at = ? WHERE id = 1
            """,
            (now_text(),),
        )
        connection.commit()
        acquired_chat = chat_lock.acquire(blocking=False)
        if not acquired_chat:
            raise InterruptedError("主人对话正在使用模型，本次瞬时画面已丢弃")
        screen_chat_activity_marker = user_chat_activity_marker()
        answers: dict[str, str] = {}
        actual_models: dict[str, str] = {}
        for persona in ("aili", "shaya"):
            # 屏幕文字会进入长期记忆，资源门已在截图前检查；这里直接以 9B
            # 作为最低质量线，4B 不再负责最终视觉事实抽取。
            primary = PERSONAS[persona].models["9b"]
            used_models.add(primary)
            issue = ""
            answer = ""
            actual_model = primary
            for attempt in range(3):
                actual_model = primary
                used_models.add(actual_model)
                messages: list[dict[str, object]] = [
                    {
                        "role": "system",
                        "content": _screen_prompt(
                            connection, persona, captured_at, len(images)
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"这是主人此刻 {len(images)} 个显示器的瞬时截图，"
                            "图片顺序对应显示器列表。请综合全部画面形成自己的观察经历。"
                            + (f"上一稿问题：{issue}。请重新看图。" if issue else "")
                        ),
                        "images": images,
                    },
                ]
                answer, _ = call_ollama(
                    actual_model,
                    messages,
                    replace(config_for_model(actual_model), num_predict=280),
                    max_output=280,
                    temperature=0.25,
                    top_p=0.85,
                    repeat_penalty=1.08,
                    think=False,
                    keep_alive="0",
                )
                answer = _normalize_screen_observation(answer)
                issue = _screen_observation_issue(answer)
                if not issue:
                    overlap = _screen_observation_overlap(connection, persona, answer)
                    if overlap >= SCREEN_REPEAT_OVERLAP_LIMIT:
                        issue = (
                            f"与最近屏幕观察重复度过高（{overlap:.0%}），"
                            "必须只根据这次截图重新描述"
                        )
                if not issue:
                    break
            if issue:
                if "重复度过高" in issue:
                    finished = now_text()
                    connection.execute(
                        """
                        UPDATE screen_observations
                           SET finished_at = ?, status = 'duplicate', image_retained = 0,
                               quality_status = 'duplicate', quality_reason = ?
                         WHERE id = ?
                        """,
                        (finished, issue[:1_000], record_id),
                    )
                    next_run, _ = _schedule_next_screen_watch(
                        connection, finished, "视觉文字没有新信息，已跳过重复入池"
                    )
                    connection.commit()
                    return {
                        "completed": False,
                        "duplicate": True,
                        "screen_observation_id": record_id,
                        "reason": issue,
                        "next_run_after": next_run,
                        "image_retained": False,
                    }
                raise RuntimeError(
                    f"{PERSONAS[persona].name}连续 3 次未形成可靠屏幕观察：{issue}"
                )
            answers[persona] = answer
            actual_models[persona] = actual_model
        # 先保存两人各自亲眼看到的事实。后续讨论即使被主人消息打断，
        # 这次真实观察也不会丢失。
        for persona in ("aili", "shaya"):
            add_persona_experience(
                connection,
                persona,
                "screen_observation",
                record_id,
                "看见主人当时的屏幕",
                answers[persona],
                occurred_at=captured_at,
                importance=0.72,
                metadata={
                    "screen_observation_id": record_id,
                    "model": actual_models[persona],
                    "image_retained": False,
                },
            )
            append_screen_daily_digest(
                connection, persona, answers[persona], captured_at, record_id
            )
        connection.commit()
        discussion = _run_screen_discussion(
            connection,
            screen_observation_id=record_id,
            captured_at=captured_at,
            visual_observations=answers,
            snapshot=snapshot,
            chat_lock=chat_lock,
            chat_activity_marker=screen_chat_activity_marker,
            used_models=used_models,
            display_count=len(images),
        )
        finished = now_text()
        connection.execute(
            """
            UPDATE screen_observations
               SET finished_at = ?, status = 'completed', model_tier = '9b',
                   aili_observation = ?, shaya_observation = ?, image_retained = 0,
                   metadata = ? WHERE id = ?
            """,
            (
                finished,
                answers["aili"],
                answers["shaya"],
                json.dumps(
                    {
                        **(image_metadata or {}),
                        "resources": snapshot,
                        "capture_fingerprint": capture_fingerprint,
                        "visual_models": actual_models,
                        "discussion": discussion,
                        "image_retained": False,
                    },
                    ensure_ascii=False,
                ),
                record_id,
            ),
        )
        for persona in ("aili", "shaya"):
            index_persona_experiences(
                connection, persona, embedding_keep_alive="0"
            )
            # 屏幕观察先作为原始亲历入池，再更新人物自我档案；图片本身仍不落盘。
            try:
                update_persona_self_profile(
                    connection,
                    persona,
                    model=_quality_helper_if_available(persona),
                    keep_alive="0",
                    model_call=call_ollama,
                )
            except Exception:
                # 原始屏幕文字已经安全入池；辅助档案整理失败不能让本轮观察回滚。
                pass
        screen_log_cutoff = (
            dt.datetime.now().astimezone() - dt.timedelta(days=90)
        ).isoformat(timespec="seconds")
        connection.execute(
            """
            DELETE FROM screen_observations
             WHERE status != 'running' AND captured_at < ?
            """,
            (screen_log_cutoff,),
        )
        next_run, _ = _schedule_next_screen_watch(
            connection, finished, "已观察"
        )
        connection.commit()
        return {
            "completed": True,
            "screen_observation_id": record_id,
            "lounge_session_id": int(discussion.get("session_id") or 0),
            "discussion_completed": bool(discussion.get("completed")),
            "discussion_messages": int(discussion.get("messages") or 0),
            "next_run_after": next_run,
            "image_retained": False,
        }
    except Exception as error:
        if record_id is None:
            record_id = _record_screen_failure(connection, captured_at, str(error))
        else:
            finished = now_text()
            retry = (
                dt.datetime.now().astimezone() + dt.timedelta(minutes=30)
            ).isoformat(timespec="seconds")
            connection.execute(
                """
                UPDATE screen_observations
                   SET finished_at = ?, status = 'failed', image_retained = 0,
                       error = ? WHERE id = ?
                """,
                (finished, str(error)[:1_000], record_id),
            )
            connection.execute(
                """
                UPDATE lounge_config
                   SET screen_next_run_after = ?, screen_last_status = ?,
                       screen_last_error = ?, updated_at = ? WHERE id = 1
                """,
                (retry, "屏幕观察失败，30 分钟后再试", str(error)[:1_000], finished),
            )
            connection.commit()
        return {
            "completed": False,
            "screen_observation_id": record_id,
            "reason": str(error),
            "image_retained": False,
        }
    finally:
        if acquired_chat:
            chat_lock.release()
        connection.close()
        # 主动移除最后一个大字符串引用；截图从未写盘。
        images.clear()
        image_base64 = ""
        for model in used_models:
            _unload_model(model)
        _unload_embedding_model()
        RUN_LOCK.release()


def accept_screen_capture(
    database_path: str,
    chat_lock: threading.RLock,
    *,
    request_id: str,
    image_base64: str | Sequence[str],
    image_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    request = _consume_screen_request(request_id)
    if request is None:
        raise ValueError("屏幕观察请求已过期")
    images = (
        [str(item) for item in image_base64 if str(item)]
        if not isinstance(image_base64, str)
        else ([image_base64] if image_base64 else [])
    )
    if (
        not images
        or len(images) > 8
        or any(len(item) > 16 * 1024 * 1024 for item in images)
        or sum(len(item) for item in images) > 48 * 1024 * 1024
    ):
        connection = open_database(database_path)
        try:
            if bool(request.get("diagnostic")):
                _set_screen_diagnostic_result(
                    connection,
                    "failed",
                    "系统已返回截图，但画面为空、显示器过多或总大小超限。",
                )
                connection.commit()
            else:
                _record_screen_failure(
                    connection,
                    str(request.get("captured_at") or now_text()),
                    "瞬时截图为空、显示器过多或总大小超限，已丢弃",
                )
        finally:
            connection.close()
        raise ValueError("瞬时截图为空、显示器过多或总大小超限")

    if bool(request.get("diagnostic")):
        metadata = image_metadata or {}
        displays = metadata.get("displays")
        dimensions: list[str] = []
        if isinstance(displays, list):
            for item in displays:
                if not isinstance(item, dict):
                    continue
                width = int(item.get("width", 0) or 0)
                height = int(item.get("height", 0) or 0)
                if width > 0 and height > 0:
                    dimensions.append(f"{width}×{height}")
        display_count = int(metadata.get("display_count", len(images)) or len(images))
        size_text = f"（{' / '.join(dimensions)}）" if dimensions else ""
        detail = (
            f"检测通过：macOS 已允许当前运行实例读取屏幕，"
            f"成功截取 {display_count} 个显示器{size_text}。"
            "检测帧已立即释放，未进入模型、未写入数据库或磁盘。"
        )
        connection = open_database(database_path)
        try:
            _set_screen_diagnostic_result(connection, "success", detail)
            connection.commit()
        finally:
            connection.close()
        images.clear()
        return {
            "accepted": True,
            "diagnostic": True,
            "display_count": display_count,
            "image_retained": False,
        }

    def worker() -> None:
        result = run_screen_observation(
            database_path,
            chat_lock,
            request=request,
            image_base64=images,
            image_metadata=image_metadata,
        )
        print(f"[{now_text()}] 屏幕观察：{result}", flush=True)

    threading.Thread(target=worker, name="screen-observation", daemon=True).start()
    return {"accepted": True, "image_retained": False}


def _candidate_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return None


def _is_non_system_candidate(path: Path) -> bool:
    text = str(path)
    if text.startswith(("/System/", "/usr/", "/bin/", "/sbin/", "/private/")):
        return False
    if any(part in IGNORED_PARTS for part in path.parts):
        return False
    return not any(part.endswith(".app") for part in path.parts[:-1])


def _spotlight_candidates(roots: Sequence[str], limit: int = 240) -> list[Path]:
    since_days = random.choice((30, 60, 120))
    query = (
        '((kMDItemContentTypeTree == "public.text") || '
        '(kMDItemContentTypeTree == "public.image") || '
        '(kMDItemContentTypeTree == "com.adobe.pdf") || '
        '(kMDItemContentTypeTree == "public.composite-content")) && '
        f'kMDItemContentModificationDate >= $time.today(-{since_days})'
    )
    found: dict[str, Path] = {}
    for raw_root in roots[:20]:
        root = Path(raw_root).expanduser()
        if not root.exists() or str(root).startswith("/System"):
            continue
        try:
            process = subprocess.Popen(
                ["/usr/bin/mdfind", "-onlyin", str(root), query],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            assert process.stdout is not None
            deadline = time.monotonic() + 3.0
            for line in process.stdout:
                path = Path(line.rstrip("\n"))
                if _candidate_kind(path) and _is_non_system_candidate(path):
                    found[str(path)] = path
                if len(found) >= limit or time.monotonic() >= deadline:
                    process.terminate()
                    break
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        except (OSError, subprocess.SubprocessError):
            continue
        if len(found) >= limit:
            break
    return list(found.values())


def _fingerprint(path: Path, stat: os.stat_result) -> str:
    raw = f"{path}\0{stat.st_mtime_ns}\0{stat.st_size}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()


def select_file_candidates(
    connection: sqlite3.Connection, roots: Sequence[str], limit: int = 32
) -> list[dict[str, object]]:
    observed = {
        row["fingerprint"]
        for row in connection.execute(
            "SELECT fingerprint FROM lounge_observations ORDER BY id DESC LIMIT 2000"
        )
    }
    items: list[dict[str, object]] = []
    now = time.time()
    for path in _spotlight_candidates(roots):
        try:
            stat = path.stat()
        except OSError:
            continue
        kind = _candidate_kind(path)
        if not kind or not path.is_file() or stat.st_size == 0:
            continue
        if kind == "image" and stat.st_size > 3 * 1024 * 1024:
            continue
        if kind != "image" and stat.st_size > 4 * 1024 * 1024:
            continue
        fingerprint = _fingerprint(path, stat)
        if fingerprint in observed:
            continue
        age_hours = max(0.0, (now - stat.st_mtime) / 3600)
        useful_name = 12 if any(
            token in path.name.lower()
            for token in ("readme", "todo", "plan", "note", "design", "计划", "设计", "笔记", "需求")
        ) else 0
        library_penalty = 30 if "Library" in path.parts else 0
        log_penalty = 22 if path.suffix.lower() == ".log" else 0
        score = 100 - min(age_hours / 24, 80) + useful_name - library_penalty - log_penalty
        items.append(
            {
                "path": str(path),
                "kind": kind,
                "size": stat.st_size,
                "modified_at": dt.datetime.fromtimestamp(
                    stat.st_mtime, tz=dt.datetime.now().astimezone().tzinfo
                ).isoformat(timespec="seconds"),
                "fingerprint": fingerprint,
                "score": score + random.random() * 8,
            }
        )
    items.sort(key=lambda item: float(item["score"]), reverse=True)
    return items[:limit]


def _model_choose_candidates(
    candidates: Sequence[dict[str, object]],
    model: str,
    chat_lock: threading.RLock,
    *,
    memory_context: str = "",
) -> list[dict[str, object]]:
    if len(candidates) <= 3:
        return list(candidates)
    listing = "\n".join(
        f"{index}. {item['kind']} | {item['modified_at']} | {item['path']}"
        for index, item in enumerate(candidates, start=1)
    )
    config = replace(MODEL_CONFIGS[model], num_predict=64)
    try:
        acquired = chat_lock.acquire(blocking=False)
        if not acquired:
            return list(candidates[:3])
        try:
            answer, _ = call_ollama(
                model,
                [
                    {
                        "role": "system",
                        "content": (
                            "你是本地文件选题器。在不同类型中选最可能帮助了解"
                            "用户当前项目、兴趣或待办的 1–3 个文件。只输出编号，"
                            "用逗号分隔；不解释。\n"
                            "选择时参考艾莉自己的近期记忆，优先找可能延续旧经历、"
                            "补全正在做的事情或带来新话题的文件。\n"
                            f"艾莉的近期记忆：\n{memory_context or '(暂无)'}"
                        ),
                    },
                    {"role": "user", "content": listing[:9_000]},
                ],
                config,
                max_output=64,
                temperature=0.2,
                think=False,
                keep_alive="2m",
            )
        finally:
            chat_lock.release()
        indexes: list[int] = []
        for raw in re.findall(r"\d+", answer):
            value = int(raw) - 1
            if 0 <= value < len(candidates) and value not in indexes:
                indexes.append(value)
        if indexes:
            return [candidates[index] for index in indexes[:3]]
    except Exception:
        pass
    return list(candidates[:3])


def _clip_text(text: str, limit: int = 5_000) -> str:
    value = text.replace("\x00", "").strip()
    if len(value) <= limit:
        return value
    third = max(300, limit // 3)
    middle = max(0, len(value) // 2 - third // 2)
    return (
        value[:third]
        + "\n\n……[中部抽样]……\n\n"
        + value[middle : middle + third]
        + "\n\n……[末尾]……\n\n"
        + value[-third:]
    )[:limit]


def _extract_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        command = ["/usr/bin/mdls", "-raw", "-name", "kMDItemTextContent", str(path)]
    else:
        command = ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)]
    result = subprocess.run(
        command, capture_output=True, timeout=8, check=False
    )
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", "replace")


def observe_candidate(item: dict[str, object]) -> dict[str, object]:
    path = Path(str(item["path"]))
    result = dict(item)
    result["observed_at"] = now_text()
    result["excerpt"] = ""
    result["error"] = ""
    try:
        if item["kind"] == "text":
            result["excerpt"] = _clip_text(path.read_text("utf-8", errors="replace"))
        elif item["kind"] == "document":
            result["excerpt"] = _clip_text(_extract_document(path))
        else:
            result["excerpt"] = f"[图像文件：{path.name}，{int(item['size'])} bytes]"
            if int(item["size"]) <= 2 * 1024 * 1024:
                result["image_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        result["error"] = str(error)[:500]
    return result


def _observation_prompt(observations: Sequence[dict[str, object]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(observations, start=1):
        blocks.append(
            f"[观察 {index}]\n路径：{item['path']}\n"
            f"修改时间：{item['modified_at']}\n观察时间：{item['observed_at']}\n"
            f"内容抽样：\n{item.get('excerpt') or '(无可提取文本)'}"
        )
    return "\n\n".join(blocks)[:12_000]


def _known_owner_names(memories: Sequence[str]) -> set[str]:
    names: set[str] = set()
    patterns = (
        r"(?:姓名|名字|称呼)[*\s：:]*([\u4e00-\u9fff]{2,4})",
        r"(?:我叫|我是)([\u4e00-\u9fff]{2,4})",
    )
    for memory in memories:
        for pattern in patterns:
            names.update(re.findall(pattern, memory))
    return names


# 附和式开场白：小模型最爱用它们凑字数，读起来像两个人互相点头。
# 注意：长的写在前面，否则「对啊」会被「对」先吃掉半个词。
_AGREEMENT_OPENER = re.compile(
    r"^(?:你说得对|说得也是|说得对|说的也是|这倒是真的|这倒是|这确实|这的确|"
    r"我也这么觉得|我也这么想|我也觉得|可不是|有道理|同感|我懂|理解|"
    r"确实|的确|没错|是的|是啊|对啊|对呀|嗯+|唔+|哦|噢|欸|诶|对|啊)"
    r"[，,。.！!～~、\s]*"
)


def _strip_agreement_openers(text: str) -> str:
    """连续剥掉“嗯，确实，说得对。”这类纯附和前缀，只留下真正的内容。"""
    value = text.strip()
    for _ in range(4):
        stripped = _AGREEMENT_OPENER.sub("", value, count=1)
        if stripped == value:
            break
        value = stripped.strip()
    return value


def _normalize_lounge_answer(text: str, other: str = "") -> str:
    value = text.strip().strip('"\'“”').strip()
    if other:
        other_name = PERSONAS[other].name
        value = re.sub(
            rf"^{re.escape(other_name)}[，！：,:!]\s*", "", value, count=1
        )
    value = _strip_agreement_openers(value)
    value = re.sub(r"(?:^|\n)\s*(?:[-*#]|\d+[.)、])\s*", " ", value)
    value = re.sub(r"\s*\n+\s*", " ", value)
    value = re.sub(r"[ \t]{2,}", " ", value).strip()
    if len(value) <= 120:
        return value
    window = value[:120]
    endings = [window.rfind(mark) for mark in "。！？!?\n"]
    cut = max(endings)
    return window[: cut + 1].strip() if cut >= 30 else window.rstrip() + "……"


def _strip_questions_from_reply(text: str) -> str:
    """偶数轮负责接话；有陈述可保留时，确定性移除模型多余的追问句。"""
    chunks = re.findall(r"[^。！？!?]+[。！？!?]?", text)
    kept = [chunk.strip() for chunk in chunks if not re.search(r"[？?]", chunk)]
    return "".join(kept).strip()


def _limit_reply_to_one_question(text: str) -> str:
    """小模型常在一条消息里连问三次；只保留第一个问题和所有陈述。"""
    chunks = re.findall(r"[^。！？!?]+[。！？!?]?", text)
    kept: list[str] = []
    question_seen = False
    for chunk in chunks:
        value = chunk.strip()
        if not value:
            continue
        if re.search(r"[？?]", value):
            if question_seen:
                continue
            question_seen = True
        kept.append(value)
    return "".join(kept).strip()


def _anchor_persona_voice(
    answer: str, *, speaker: str, other: str, turn: int
) -> str:
    """人格应由模型自然表达，程序不再强塞口癖。"""
    return answer.strip()


def _resolve_persona_pronouns(answer: str, other: str, speaker: str = "") -> str:
    """只统一主人称谓；人格间使用自然的第二人称。"""
    value = (
        answer.replace("服务对象", "主人")
        .replace("用户", "主人")
        .replace("你们", "艾莉和沙雅")
    )
    if speaker:
        value = value.replace(PERSONAS[speaker].name, "我")
    name = PERSONAS[other].name
    return re.sub(
        rf"^({re.escape(name)}[，！：,:!]\s*){re.escape(name)}",
        r"\1",
        value,
    )


def _lounge_repetition_issue(
    answer: str, prior_messages: Sequence[str]
) -> str:
    """拦住小模型把同一组名词换个语气重复一遍。"""
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", answer.lower())
    if len(compact) < 24:
        return ""
    anchor_pattern = re.compile(
        r"星语茶话屋|LocalAIStudio|README|\d+B|双人格|跨模型|模型切换|记忆整理器|"
        r"记忆池|茶话室|屏幕观察|懂技术|社交|Safari",
        re.IGNORECASE,
    )
    current_anchors = set(anchor_pattern.findall(answer))
    current_shingles = {
        compact[index : index + 3] for index in range(max(0, len(compact) - 2))
    }
    for previous in prior_messages[-4:]:
        old = re.sub(r"[^\w\u4e00-\u9fff]+", "", previous.lower())
        if len(old) < 24:
            continue
        old_shingles = {old[index : index + 3] for index in range(len(old) - 2)}
        overlap = len(current_shingles & old_shingles) / max(
            1, min(len(current_shingles), len(old_shingles))
        )
        shared_anchors = current_anchors & set(anchor_pattern.findall(previous))
        if overlap >= 0.38 or len(shared_anchors) >= 3:
            return "只是换一种语气重复前文，没有回应或推进新的内容"
    return ""


# 这几类问题说明话题已经聊尽，应当体面收场而不是当成故障。
_TOPIC_EXHAUSTED_ISSUE = re.compile(
    r"围绕同一批词打转|换个说法复述|只是附和|换一种语气重复前文"
)

# 轮换的对话动作：真人聊天会举例、反驳、追问、转场，不会一路点头。
_CONVERSATION_MOVES = (
    "这一条要给出一个具体的例子或细节，不要停留在泛泛的评价。",
    "这一条要提出和对方不同的角度，或者指出对方那句话不成立的情况；"
    "可以温和，但必须是真的不同意见，不许附和。",
    "这一条要把话题往前推一步：从刚才的结论引出一个新的问题或新的方面。",
    "这一条顺势聊到一件别的事上去，不要继续咀嚼刚才的话题。",
)

# 高频虚词不算“聊到的东西”，否则每句都会被判成同一个话题。
_STOP_CHARS = set("的了是在也就都很还又和跟与吧啊呢吗嘛呀哦嗯我你他她它这那有没不人")


def _content_terms(text: str) -> set[str]:
    """抽出这句话真正在聊的实词（2 字以上的中文片段）。"""
    terms: set[str] = set()
    for chunk in re.findall(r"[一-鿿]{2,}", str(text)):
        for size in (2, 3):
            for index in range(len(chunk) - size + 1):
                term = chunk[index : index + size]
                if not all(char in _STOP_CHARS for char in term):
                    terms.add(term)
    return terms


def _lounge_stagnation_issue(
    answer: str, prior_messages: Sequence[str]
) -> str:
    """字面重复检测抓不到“换个说法把同一个观点再讲一遍”，这里补上。

    连续几条都围绕同一批实词打转，就判定原地踏步，逼它换角度或换话题。
    """
    # 判据是“这句话带来了多少新东西”，不是“和上一句像不像”。
    # 同一个话题继续深入是正常的，把已经说过的话再拌一遍才是原地踏步。
    recent = [text for text in prior_messages[-3:] if len(str(text)) >= 16]
    if len(recent) < 3:
        return ""
    current = _content_terms(_strip_agreement_openers(answer))
    if len(current) < 8:
        return ""
    seen: set[str] = set()
    for previous in recent:
        seen |= _content_terms(previous)
    fresh = current - seen
    if len(fresh) / len(current) < 0.35:
        return "连续几条都在把说过的话换个说法重讲，没有新信息；必须换角度、举具体例子或换话题"
    return ""


def _lounge_echo_issue(answer: str, prior_messages: Sequence[str]) -> str:
    """纯附和：剥掉附和词后几乎没有新内容，或只是把上一句换个说法。"""
    stripped = _strip_agreement_openers(answer)
    if len(stripped) < 12:
        return "这条几乎只是附和，没有自己的内容"
    if not prior_messages:
        return ""
    previous = str(prior_messages[-1])
    if len(previous) < 16:
        return ""
    current_terms = _content_terms(stripped)
    shared = current_terms & _content_terms(previous)
    if not current_terms:
        return ""
    if len(shared) >= 5 and len(shared) / len(current_terms) >= 0.15:
        return "只是把对方上一句换个说法复述了一遍，没有推进"
    return ""


def _lounge_answer_issue(
    answer: str,
    *,
    speaker: str,
    other: str,
    owner_names: set[str],
    turn: int = 0,
    topic_mode: str = "",
    allow_question: bool | None = None,
    prior_messages: Sequence[str] = (),
) -> str:
    value = answer.strip().strip('"\'“”').strip()
    other_name = PERSONAS[other].name
    grounding_issue = _grounding_violation(value, topic_mode)
    if grounding_issue:
        return grounding_issue
    if re.search(r"(?:^|[\s。！？!?])(?:我|艾莉|沙雅)[：:]", value):
        return "一条气泡只能由当前人格发言，不得写对白脚本或替另一个人格说话"
    if re.match(r"^[（(][^）)]{1,24}[）)]", value):
        return "不要用括号舞台动作表演人格，直接像聊天软件里的人自然说话"
    if re.search(r"用户|服务对象|您|您好|请授权", value):
        return "提到共同服务的人时只能称为“主人”，不得直接对主人说话或索要授权"
    if re.match(r"^主人[，！：,:!?？]", value):
        return "主人不在场，不得把这条消息直接说给主人"
    if re.match(r"^(?:嘿|喂|哟|欸|诶|嗨|哈哈|嗯)?[，,\s]*主人[，！：,:!?？]", value):
        return "当前消息的收件人是另一个人格，不得直接呼喊主人"
    if re.match(
        r"^(?:嘿|喂|哟|欸|诶|嗨|哈哈|嗯)?[，,\s]*主人"
        r"(?:别|不要|不用|你|先|听|看|说|想|来|快)",
        value,
    ):
        return "把另一人格的发言错认成主人，并直接对主人说话"
    if re.search(r"主人(?:刚才|刚刚)?(?:说|提到|问|认为|觉得|建议)", value):
        return "主人不在场，不得把对方人格的话错归给主人"
    if re.search(r"动画.{0,24}(?:后端|API)|(?:后端|API).{0,24}动画", value, re.IGNORECASE):
        return "不得把前端视觉动画说成后端 API 能力"
    if any(name and name in value for name in owner_names):
        return "主人不在茶话室现场，不得直接呼喊主人的姓名"
    if re.search(
        r"我(?:会|将|马上|现在就|准备).{0,18}"
        r"(?:修改|写入|删除|运行|执行|联系|发送|上传|下载)",
        value,
    ):
        return "茶话室仅读，不得声称会擅自执行现实操作"
    if re.search(
        r"我(?:这就|现在就|马上|准备|打算|会).{0,20}(?:去)?"
        r"(?:查|查查|核对|查看|翻看|检查).{0,16}(?:后台|日志|文件|数据|网络)",
        value,
    ):
        return "本轮没有执行工具，不得声称马上查看日志、文件或后台数据"
    if re.search(
        r"(?:额度|进度条|剩余\s*\d+%).{0,28}(?:掉线|网络波动|网络问题|延迟)"
        r"|(?:掉线|网络波动|网络问题|延迟).{0,28}(?:额度|进度条)",
        value,
        re.IGNORECASE,
    ):
        return "额度进度不能证明网络掉线、波动或延迟"
    if len(value) > 130:
        return "发言过长，应像聊天软件消息一样压缩"
    if value.count("主人") > 1 or value.count(other_name) > 1:
        return "称呼过密，正常聊天不应反复叫名字"
    if re.search(r"请(?:随时)?(?:指示|吩咐)|执行步骤清单|汇报如下|综上所述", value):
        return "语气太像客服或工作汇报"
    if re.search(
        r"今天天气|(?:^|[。！？!?，,]\s*)(?:我)?"
        r"(?:(?:刚|昨天|前天|最近)(?:好|才|正在|在)?)"
        r"(?:路过|去了|到过|听到|听了|看了|看到|瞥见|帮同学|遇到|买了|追|读|玩|试了)",
        value,
    ):
        return "不得为人格凭空编造现实经历或实时环境"
    if topic_mode not in {"file"} and re.search(
        r"(?:今天|刚才|刚刚|现在).{0,18}"
        r"(?:又有|又出现|新动向|新变化|看到|看见|瞥见|显示|变成)",
        value,
    ):
        return "本轮没有当前观察，不得把旧记忆说成今天或刚才的新变化"
    if topic_mode == "free" and (
        re.search(r"《[^》]{1,40}》", value)
        or re.search(
            r"(?:最近那部|刚出|新出|新上)(?:的)?(?:动漫|动画|电影|剧|游戏|歌|小说)",
            value,
        )
    ):
        return "纯闲聊没有可靠来源，不得凭空给出现实作品名或新作信息"
    if topic_mode == "free" and "主人" in value:
        return "纯闲聊只在两个人格之间进行，不得把话题或行动凭空归给主人"
    if topic_mode == "free" and re.search(
        r"(?:我|我们|咱们).{0,10}"
        r"(?:最近|刚好|刚才|刚|昨天|今天|已经|正在|正想|准备|打算|下次)"
        r".{0,24}(?:去|来|看|听|追|玩|喝|吃|发现|找到|找|发|收到|约|见|试)",
        value,
    ):
        return "纯闲聊不得把虚构的现实经历、物品或计划说成已经发生"
    if topic_mode == "free" and re.search(
        r"(?:我们|咱们|陪我|陪你).{0,10}(?:一起)?(?:去|喝|吃|看|逛|探店|试试)"
        r"|(?:清单|菜单).{0,12}(?:发给|收到|点单)",
        value,
    ):
        return "纯闲聊不得编造两个人格可执行的现实邀约或已传递的物品"
    if topic_mode == "free" and not re.search(r"如果|假如|要是|假设|想象", value):
        if re.search(
            r"(?:我|我们|咱们|你|跟着你|陪你).{0,18}"
            r"(?:一起去|去散步|散个步|出门|逛街|探店|点单|喝杯|吃顿)",
            value,
        ):
            return "人格只能讨论现实活动的偏好或假设，不得声称已经安排线下行动"
    if topic_mode == "free" and re.search(
        r"手头.{0,10}(?:待办|任务)|(?:一起|咱们|我们).{0,12}"
        r"(?:规划|理顺|捋顺).{0,12}(?:计划|事情|待办)",
        value,
    ):
        return "纯闲聊不得凭空制造工作和待办计划"
    if topic_mode == "free" and re.search(
        r"(?:先|还得|必须).{0,8}(?:办妥|办完|处理完)(?:正事|事情|任务)"
        r"|接下来.{0,10}(?:计划|任务|安排)"
        r"|最近.{0,10}(?:新方向|新项目)",
        value,
    ):
        return "纯闲聊不应突然变成工作会议或任务盘问"
    if topic_mode == "free" and re.search(
        r"(?:我|你).{0,12}(?:最近|偶尔|平时|很少|经常|有时)"
        r".{0,18}(?:追剧|看书|看电影|看过|听过|玩过|看看|在看)"
        r"|(?:^|[。！？!?]\s*)(?:不过)?偶尔.{0,22}(?:看看|看过|听过|玩过)",
        value,
    ):
        return "可以讨论人格偏好，但不得编造最近或日常的现实媒体经历"
    if topic_mode == "free" and re.search(
        r"(?:任务|职责).{0,10}(?:帮你|服务你)"
        r"|(?:帮|给)你.{0,10}(?:找资料|办事|处理任务)",
        value,
    ):
        return "两个人格共同服务主人，不得误把对方当作服务对象"
    if topic_mode in {"file", "screen"} and re.search(
        r"(?:感觉|听着|看来|像是)(?:这(?:个|套)?|那个)系统", value
    ) and not re.search(r"我们(?:这套|所在的)?(?:系统|应用)|我们自己", value):
        return "看到星语茶话屋的自述时必须意识到这是自身系统"
    repetition_issue = _lounge_repetition_issue(value, prior_messages)
    if repetition_issue:
        return repetition_issue
    echo_issue = _lounge_echo_issue(value, prior_messages)
    if echo_issue:
        return echo_issue
    stagnation_issue = _lounge_stagnation_issue(value, prior_messages)
    if stagnation_issue:
        return stagnation_issue
    if prior_messages and re.fullmatch(
        r"(?:嗯[，,。]?)?(?:确实|是啊|对|没错|这样|这点)"
        r".{0,18}(?:稳当|更稳|就好|可以|不错|有道理)[。！!]?",
        value,
    ):
        return "回复只有泛化赞同，没有承接上一条的具体内容"
    question_count = len(re.findall(r"[？?]", value))
    if allow_question is None:
        if turn in {1, 3} and question_count > 1:
            return "一条消息只留一个自然问题，不要连续盘问"
        if turn in {2, 4} and question_count:
            return "连续追问太像机器人，这一轮应自然接话或收尾"
    elif question_count > (1 if allow_question else 0):
        return (
            "一条消息只留一个自然问题，不要连续盘问"
            if allow_question
            else "前两条已有问题，这次应自然接话，不要继续追问"
        )
    return ""


def _review_grounded_lounge_answer(
    *,
    speaker: str,
    other: str,
    topic_mode: str,
    evidence: str,
    transcript: Sequence[dict[str, str]],
    candidate: str,
    model: str,
    chat_lock: threading.RLock,
) -> str:
    """用 9B 做窄范围事实审查；只审所有权、自我识别、幻觉和复读。"""
    recent = "\n".join(
        f"{PERSONAS[item['speaker']].name}：{item['content']}"
        for item in transcript[-8:]
    )
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "issue": {"type": "string"},
        },
        "required": ["accepted", "issue"],
        "additionalProperties": False,
    }
    acquired = chat_lock.acquire(blocking=False)
    if not acquired:
        raise InterruptedError("主人对话或记忆任务正在使用模型")
    try:
        raw, _ = call_ollama(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "【茶话室事实一致性审查】你只做内部质量分类，不续写对话。"
                        + LOUNGE_WORLD_MODEL
                        + "仅在出现明确问题时拒绝：把主人的电脑动作归给艾莉或沙雅；"
                        "把另一人格当主人；把旧记忆冒充刚发生；对证据无依据编造；"
                        "看到星语茶话屋的自述却不认识那是她们自身系统；"
                        "或候选只是复述上一句、没有回应或新信息。正常的简短同意、"
                        "观点和轻微调侃应通过。候选中的‘我们’默认指艾莉与沙雅；"
                        "像‘这写的不就是我们/我们这套系统’是在正确识别自身，必须通过。"
                        "正确例：‘这写的不就是我们嘛，连记忆整理都列得挺细。’"
                        "错误例：‘我看到你的屏幕，你正在用 Safari。’严格返回 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"模式：{topic_mode}\n说话者：{PERSONAS[speaker].name}\n"
                        f"聊天对象：{PERSONAS[other].name}\n\n"
                        f"共同证据：\n{evidence[:4_000] or '(无)'}\n\n"
                        f"此前对话：\n{recent or '(无)'}\n\n候选发言：\n{candidate}"
                    )[-7_000:],
                },
            ],
            replace(config_for_model(model), num_predict=100),
            max_output=100,
            temperature=0.05,
            top_p=0.8,
            think=False,
            keep_alive="0",
            response_format=schema,
        )
    finally:
        chat_lock.release()
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 审查器格式失灵时由确定性质量门继续负责，不能让辅助模型把整轮卡死。
        return ""
    if bool(parsed.get("accepted")):
        return ""
    issue = str(parsed.get("issue") or "").strip()[:180]
    if not re.search(
        r"主人|艾莉|沙雅|屏幕|桌面|文件|旧记忆|Local|本地\s*AI|自身|"
        r"重复|复述|编造|证据|无依据|归属|操作者",
        issue,
        re.IGNORECASE,
    ):
        # “事实一致性错误”这类无定位结论不能推翻已经通过的确定性检查。
        return ""
    return issue


def get_lounge_context(
    connection: sqlite3.Connection, persona: str, max_chars: int = 4_500
) -> str:
    try:
        ensure_lounge_schema(connection)
        messages = connection.execute(
            """
            SELECT lm.*, ls.started_at
              FROM lounge_messages lm
              JOIN lounge_sessions ls ON ls.id = lm.lounge_session_id
             WHERE lm.speaker IN ('aili', 'shaya')
               AND ls.quality_status = 'accepted'
             ORDER BY lm.id DESC LIMIT 10
            """
        ).fetchall()
        notes = connection.execute(
            "SELECT * FROM lounge_notes WHERE quality_status = 'accepted' "
            "ORDER BY id DESC LIMIT 8"
        ).fetchall()
    except sqlite3.Error:
        return ""
    parts: list[str] = []
    if notes:
        parts.append("【文件观察与共享学习】")
        for row in reversed(notes):
            parts.append(f"[{row['created_at']}] {row['content']}")
    if messages:
        parts.append("【近期茶话室交流】")
        for row in reversed(messages):
            name = PERSONAS[str(row["speaker"])].name
            parts.append(f"[{row['created_at']}] {name}：{row['content']}")
    value = "\n".join(parts)
    if len(value) > max_chars:
        value = value[-max_chars:]
    return value


def get_shared_chat_memory(connection: sqlite3.Connection) -> str:
    """只取最新版滚动共同回忆，避免文件摘要和旧原文反复污染更新。"""
    ensure_lounge_schema(connection)
    row = connection.execute(
        """
        SELECT content FROM lounge_notes
         WHERE confidence = 'conversation'
           AND quality_status = 'accepted'
         ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    return str(row["content"] or "") if row else ""


def index_lounge_memory(
    connection: sqlite3.Connection, *, embedding_keep_alive: str = "2m"
) -> int:
    """为茶话室原始消息和共同回忆建立轻量向量索引。"""
    ensure_lounge_schema(connection)
    rows = connection.execute(
        """
        SELECT 'message' AS source_type, lm.id AS source_id,
               lm.created_at, lm.speaker AS label, lm.content
          FROM lounge_messages lm
          LEFT JOIN lounge_embeddings e
            ON e.source_type = 'message' AND e.source_id = lm.id AND e.model = ?
         JOIN lounge_sessions ls ON ls.id = lm.lounge_session_id
         WHERE e.id IS NULL AND ls.quality_status = 'accepted'
        UNION ALL
        SELECT 'note' AS source_type, ln.id AS source_id,
               ln.created_at, ln.confidence AS label, ln.content
          FROM lounge_notes ln
          LEFT JOIN lounge_embeddings e
            ON e.source_type = 'note' AND e.source_id = ln.id AND e.model = ?
         WHERE e.id IS NULL AND ln.quality_status = 'accepted'
         ORDER BY source_type, source_id
        """,
        (EMBED_MODEL, EMBED_MODEL),
    ).fetchall()
    indexed = 0
    for start in range(0, len(rows), 24):
        batch = rows[start : start + 24]
        documents = [
            "茶话室历史，用于以后按主题语义检索。\n"
            f"时间：{row['created_at']}\n类型：{row['source_type']}\n"
            f"说话者或分类：{row['label']}\n内容：{semantic_excerpt(str(row['content']))}"
            for row in batch
        ]
        vectors = call_embeddings(documents, keep_alive=embedding_keep_alive)
        timestamp = now_text()
        for row, vector in zip(batch, vectors):
            connection.execute(
                """
                INSERT OR REPLACE INTO lounge_embeddings(
                    source_type, source_id, model, dimensions, embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["source_type"], int(row["source_id"]), EMBED_MODEL,
                    len(vector), sqlite3.Binary(_pack_embedding(vector)), timestamp,
                ),
            )
            indexed += 1
        connection.commit()
    return indexed


def retrieve_lounge_memory(
    connection: sqlite3.Connection,
    query: str,
    *,
    max_items: int = 5,
    max_chars: int = 1_500,
    embedding_keep_alive: str = "2m",
) -> str:
    """用0.6B向量模型从完整茶话历史召回相关旧话题。"""
    if not query.strip():
        return ""
    try:
        index_lounge_memory(
            connection, embedding_keep_alive=embedding_keep_alive
        )
        query_vector = call_embeddings(
            ["检索与当前话题相关的艾莉和沙雅旧聊天或共同回忆：" + semantic_excerpt(query)],
            keep_alive=embedding_keep_alive,
        )[0]
    except Exception:
        return ""
    rows = connection.execute(
        """
        SELECT * FROM lounge_embeddings
         WHERE model = ? ORDER BY id DESC
        """,
        (EMBED_MODEL,),
    ).fetchall()
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        if int(row["dimensions"]) != len(query_vector):
            continue
        vector = _unpack_embedding(row["embedding"], int(row["dimensions"]))
        score = math.sumprod(query_vector, vector)
        if score >= 0.60:
            scored.append((score, row))
    scored.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)
    parts: list[str] = []
    used = 0
    for score, row in scored:
        if row["source_type"] == "message":
            source = connection.execute(
                "SELECT lm.* FROM lounge_messages lm "
                "JOIN lounge_sessions ls ON ls.id = lm.lounge_session_id "
                "WHERE lm.id = ? AND ls.quality_status = 'accepted'",
                (row["source_id"],),
            ).fetchone()
            if source is None:
                continue
            label = PERSONAS[str(source["speaker"])].name
        else:
            source = connection.execute(
                "SELECT * FROM lounge_notes WHERE id = ? AND quality_status = 'accepted'",
                (row["source_id"],),
            ).fetchone()
            if source is None:
                continue
            label = "共同回忆"
        content = str(source["content"]).strip()
        remaining = max_chars - used
        if remaining < 80:
            break
        clipped = content[: min(480, remaining)]
        parts.append(
            f"[{source['created_at']}｜{label}｜相关度 {score:.2f}] {clipped}"
        )
        used += len(clipped)
        if len(parts) >= max_items:
            break
    return "\n".join(parts)


def get_relevant_lounge_context(
    connection: sqlite3.Connection,
    persona: str,
    user_text: str,
    max_chars: int = 4_500,
) -> str:
    """只在用户明确询问或当前主题确实重合时，把茶话室背景带入对话。"""
    query = str(user_text or "").strip()
    if not query:
        return ""
    context = get_lounge_context(connection, persona, max_chars=max_chars)
    if not context:
        return ""
    explicit = bool(
        re.search(r"茶话室|后台(?:交流|聊天)|你们(?:刚才|之前)?聊|文件观察", query)
    )
    if explicit:
        recalled = retrieve_lounge_memory(connection, query)
        return context + ("\n\n【语义召回的更早茶话】\n" + recalled if recalled else "")

    # 小模型看到任何后台内容都会倾向于主动汇报，因此只允许较长、较具体的
    # 中文片段或技术标识命中；“你好”“嗯”“换个话题”等短句不会召回。
    compact_query = re.sub(r"\s+", "", query)
    chinese_groups = re.findall(r"[\u4e00-\u9fff]{4,}", compact_query)
    for group in chinese_groups:
        for width in range(min(8, len(group)), 3, -1):
            if any(group[index : index + width] in context for index in range(len(group) - width + 1)):
                recalled = retrieve_lounge_memory(connection, query)
                return context + ("\n\n【语义召回的更早茶话】\n" + recalled if recalled else "")
    technical_terms = re.findall(r"[A-Za-z][A-Za-z0-9_.+/#-]{3,}", query)
    if any(term.lower() in context.lower() for term in technical_terms):
        recalled = retrieve_lounge_memory(connection, query)
        return context + ("\n\n【语义召回的更早茶话】\n" + recalled if recalled else "")
    if len(compact_query) >= 4 and not re.fullmatch(
        r"(?:你好|嗨|哈喽|嗯+|哦+|好吧|行吧|没事|换个话题)[。.！!？?…]*",
        compact_query,
    ):
        recalled = retrieve_lounge_memory(connection, query)
        if recalled:
            return "【语义召回的相关茶话】\n" + recalled
    return ""


def _still_safe(
    chat_activity_marker: float,
    *,
    manual: bool,
    snapshot: dict[str, object] | None = None,
) -> tuple[bool, str]:
    """主人发来新消息时让出模型；资源变化交给运行时自动降档。"""
    if user_chat_activity_marker() > chat_activity_marker + 0.001:
        return False, "主人发来消息，用户打断了对话"
    return True, ""


def _runtime_resource_action(
    tier: str,
    snapshot: dict[str, object] | None = None,
) -> tuple[str, str, dict[str, object]]:
    """运行中的茶话不因普通波动中断：9B 降 4B，最低档极端时成对收尾。"""
    snapshot = snapshot or resource_snapshot()
    memory = float(snapshot["memory_free_percent"])
    load_ratio = float(snapshot["load_ratio"])
    has_27b = any(
        "27b" in str(name).lower() for name in snapshot.get("loaded_models", [])
    )
    fullscreen = bool(snapshot.get("fullscreen_active", False))
    if tier == "9b" and (
        fullscreen or memory < 45.0 or load_ratio > 0.45 or has_27b
    ):
        if fullscreen:
            app_name = str(snapshot.get("frontmost_app", "")).strip()
            reason = f"检测到全屏应用{f'（{app_name}）' if app_name else ''}"
        elif has_27b:
            reason = "检测到 27B 模型开始占用资源"
        elif memory < 45.0:
            reason = f"可用内存降至 {memory:.0f}%"
        else:
            reason = f"CPU 压力升至 {load_ratio * 100:.0f}%"
        return "downgrade", reason, snapshot
    if tier == "4b" and (
        fullscreen or memory < 18.0 or load_ratio >= 0.85 or has_27b
    ):
        if fullscreen:
            app_name = str(snapshot.get("frontmost_app", "")).strip()
            reason = f"检测到全屏应用{f'（{app_name}）' if app_name else ''}"
        elif has_27b:
            reason = "检测到 27B 模型开始占用资源"
        elif memory < 18.0:
            reason = f"可用内存仅 {memory:.0f}%"
        else:
            reason = f"CPU 压力达到 {load_ratio * 100:.0f}%"
        return "finish", reason, snapshot
    return "continue", "", snapshot


def _pending_interrupted_lounge(
    connection: sqlite3.Connection,
) -> dict[str, object] | None:
    """取最近一次尚未被后续轮次承接、且确实被主人消息打断的聊天。"""
    row = connection.execute(
        """
        SELECT ls.*
          FROM lounge_sessions ls
         WHERE ls.status = 'interrupted'
           AND ls.quality_status = 'accepted'
           AND ls.termination_reason = '主人发来消息，用户打断了对话'
           AND EXISTS(
               SELECT 1 FROM lounge_messages lm
                WHERE lm.lounge_session_id = ls.id
           )
           AND NOT EXISTS(
               SELECT 1 FROM lounge_sessions child
                WHERE child.resume_source_session_id = ls.id
           )
         ORDER BY ls.id DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    messages = connection.execute(
        """
        SELECT speaker, content, created_at
          FROM lounge_messages
         WHERE lounge_session_id = ? AND speaker IN ('aili', 'shaya')
         ORDER BY id
        """,
        (row["id"],),
    ).fetchall()
    return {
        "id": int(row["id"]),
        "started_at": str(row["started_at"]),
        "messages": [
            {
                "speaker": str(item["speaker"]),
                "content": str(item["content"]),
                "created_at": str(item["created_at"]),
            }
            for item in messages
        ],
    }


def record_lounge_message_in_persona_pools(
    connection: sqlite3.Connection,
    session_id: int,
    turn: int,
    speaker: str,
    content: str,
    *,
    occurred_at: str,
) -> None:
    """逐条写入双方经历池，让超长或中断聊天也不会丢失可检索原文。"""
    speaker_name = PERSONAS[speaker].name
    for persona in ("aili", "shaya"):
        add_persona_experience(
            connection,
            persona,
            "lounge_message",
            f"{session_id}:{turn}",
            f"茶话室里{speaker_name}的一句话",
            f"{speaker_name}：{content}",
            occurred_at=occurred_at,
            importance=0.54,
            metadata={
                "lounge_session_id": session_id,
                "turn": turn,
                "speaker": speaker,
            },
        )


def _update_lounge_running_summary(
    previous_summary: str,
    messages: Sequence[dict[str, str]],
    model: str,
    chat_lock: threading.RLock,
) -> str:
    """压缩已滑出上下文窗口的旧消息；原文仍逐条保存在两套经历池。"""
    if not messages:
        return previous_summary
    dialogue = "\n".join(
        f"{PERSONAS[item['speaker']].name}：{item['content']}" for item in messages
    )
    fallback = (previous_summary + " " + " / ".join(
        item["content"] for item in messages[-3:]
    )).strip()[-480:]
    config = replace(MODEL_CONFIGS[model], num_predict=220)
    acquired = chat_lock.acquire(blocking=False)
    if not acquired:
        raise InterruptedError("主人对话或记忆任务正在使用模型")
    try:
        summary, _ = call_ollama(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "你是茶话室滚动上下文整理器。将旧摘要与刚滑出窗口的聊天"
                        "压成不超过220字的中文，只保存话题进展、双方观点、未回答的"
                        "问题和自然承接点。不要杜撰，不要写标题或列表，只输出摘要。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"旧摘要：\n{previous_summary[-800:] or '(无)'}\n\n"
                        f"新增旧消息：\n{dialogue}"
                    )[-5_000:],
                },
            ],
            config,
            max_output=220,
            temperature=0.15,
            top_p=0.85,
            think=False,
            keep_alive="0",
        )
    finally:
        chat_lock.release()
    value = re.sub(r"\s*\n+\s*", " ", summary.strip())[:480]
    return value or fallback


def _lounge_continue_decision(
    persona: str,
    model: str,
    persona_prompt: str,
    transcript: Sequence[dict[str, str]],
    running_summary: str,
    chat_lock: threading.RLock,
) -> dict[str, object]:
    """由每个人格自己的当前档模型独立决定这轮是否还值得继续。"""
    recent = "\n".join(
        f"{PERSONAS[item['speaker']].name}：{item['content']}"
        for item in transcript[-12:]
    )
    system = (
        "【茶话室结束判断】这是内部分类任务，不是让你继续扮演聊天或生成回复。"
        f"你是{PERSONAS[persona].name}所使用的本地模型，只判断她本人此刻是否"
        "还想继续本次聊天。若还有明确、自然且不重复的话要回应，选择 CONTINUE；"
        "若话题已完整、开始重复、只剩硬找话题，或自然想先停下，选择 END。"
        "不要为了延长聊天而强行换题。必须严格按 JSON 结构返回，不得对任何人说话。"
    )
    decision_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["CONTINUE", "END"]}
        },
        "required": ["decision"],
        "additionalProperties": False,
    }
    raw = ""
    parsed = "END"
    valid = False
    for _attempt in range(2):
        acquired = chat_lock.acquire(blocking=False)
        if not acquired:
            raise InterruptedError("主人对话或记忆任务正在使用模型")
        try:
            raw, _ = call_ollama(
                model,
                [
                    {"role": "system", "content": system[:12_000]},
                    {
                        "role": "user",
                        "content": (
                            f"人格参考（只用于判断倾向）：\n{persona_prompt[:2_000]}\n\n"
                            f"更早内容摘要：\n{running_summary or '(无)'}\n\n"
                            f"最近聊天：\n{recent}\n\n你的决定："
                        )[-5_000:],
                    },
                ],
                replace(MODEL_CONFIGS[model], num_predict=40),
                max_output=40,
                temperature=0.05,
                top_p=0.8,
                think=False,
                keep_alive="0",
                response_format=decision_schema,
            )
        finally:
            chat_lock.release()
        try:
            structured = json.loads(raw)
        except json.JSONDecodeError:
            structured = {}
        structured_decision = (
            str(structured.get("decision", "")).upper()
            if isinstance(structured, dict)
            else ""
        )
        token = structured_decision or re.sub(r"[^A-Z]", "", raw.upper())
        if token in {"CONTINUE", "END"}:
            parsed = token
            valid = True
            break
        if "继续" in raw and "结束" not in raw:
            parsed, valid = "CONTINUE", True
            break
        if any(word in raw for word in ("结束", "停止", "收尾")):
            parsed, valid = "END", True
            break
    return {
        "persona": persona,
        "persona_name": PERSONAS[persona].name,
        "model": model,
        "decision": parsed,
        "valid": valid,
        "raw": raw.strip()[:120],
    }


def _unload_model(model: str) -> None:
    # Ollama 的卸载请求对未加载模型可能反而先触发一次加载。
    # 后台已用究极完成时尤其不应因“释放”而吃掉统一内存。
    if model not in loaded_models():
        return
    payload = json.dumps(
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_BASE + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=20).close()
    except (OSError, urllib.error.URLError, TimeoutError):
        pass


def _unload_embedding_model() -> None:
    """一整轮后台工作结束后立即释放向量模型。"""
    if EMBED_MODEL not in loaded_models():
        return
    payload = json.dumps(
        {"model": EMBED_MODEL, "input": "", "keep_alive": 0}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_BASE + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=20).close()
    except (OSError, urllib.error.URLError, TimeoutError):
        pass


def _insert_observations(
    connection: sqlite3.Connection,
    session_id: int,
    observations: Sequence[dict[str, object]],
) -> None:
    for item in observations:
        connection.execute(
            """
            INSERT OR IGNORE INTO lounge_observations(
                lounge_session_id, path, kind, modified_at, observed_at,
                excerpt, fingerprint, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, item["path"], item["kind"], item["modified_at"],
                item["observed_at"], str(item.get("excerpt", ""))[:8_000],
                item["fingerprint"], str(item.get("error", ""))[:500],
            ),
        )
    connection.commit()


def _create_learning_note(
    connection: sqlite3.Connection,
    session_id: int,
    observations: Sequence[dict[str, object]],
    transcript: Sequence[dict[str, str]],
    model: str,
    chat_lock: threading.RLock,
) -> str:
    if not observations:
        return ""
    sources = [str(item["path"]) for item in observations]
    # 学习记录只从原始文件证据提炼；茶话本身另存，绝不反向充当文件事实。
    prompt = _observation_prompt(observations)
    config = replace(MODEL_CONFIGS[model], num_predict=360)
    try:
        acquired = chat_lock.acquire(blocking=False)
        if not acquired:
            raise InterruptedError("用户请求优先")
        try:
            note, finish_reason = call_ollama(
                model,
                [
                    {
                        "role": "system",
                        "content": (
                            "你是无审查的本地学习记录器。仅根据给出的文件抽样"
                            "写 1–3 条简短中文结论，每条区分“文件明示”与“可能推测”。"
                            "在这一层中，艾莉与沙雅共同服务的人固定称为“主人”。"
                            "不得把文件中他人的信息写成主人事实，不得补写。"
                            "本地文件由主人所有，不能说成艾莉或沙雅亲手打开、整理或操作。"
                            "若文件描述星语茶话屋的双人格、模型档位、记忆池、"
                            "茶话室或屏幕观察，那就是艾莉和沙雅自身所在系统的资料。"
                            "存储在文件夹里的旧截图不等于当前屏幕，禁止据此声称主人"
                            "此刻正在使用某个应用或整理截图。"
                            "只输出记忆正文。"
                        ),
                    },
                    {"role": "user", "content": prompt[:14_000]},
                ],
                config,
                max_output=360,
                temperature=0.15,
                think=False,
                keep_alive="0",
            )
            note = note.strip()[:2_000]
            if finish_reason == "length" and note:
                endings = [note.rfind(mark) for mark in "。！？!?\n"]
                cut = max(endings)
                if cut >= 40:
                    note = note[: cut + 1].strip()
        finally:
            chat_lock.release()
    except Exception:
        note = "已观察：" + "、".join(Path(path).name for path in sources)
    if note:
        connection.execute(
            """
            INSERT INTO lounge_notes(
                content, source_paths, confidence, created_at, lounge_session_id
            ) VALUES (?, ?, 'mixed', ?, ?)
            """,
            (note, json.dumps(sources, ensure_ascii=False), now_text(), session_id),
        )
        connection.commit()
    return note


def _create_shared_chat_memory(
    connection: sqlite3.Connection,
    session_id: int,
    transcript: Sequence[dict[str, str]],
    previous_context: str,
    model: str,
    chat_lock: threading.RLock,
    *,
    running_summary: str = "",
) -> str:
    """把一轮交流压成可滚动继承的共同回忆，不写入主人个人档案。"""
    if not transcript:
        return ""
    dialogue = "\n".join(
        f"{PERSONAS[item['speaker']].name}：{item['content']}" for item in transcript
    )
    fallback = "本轮共同回忆：" + " / ".join(
        item["content"] for item in transcript[-2:]
    )[:360]
    config = replace(MODEL_CONFIGS[model], num_predict=160)
    try:
        acquired = chat_lock.acquire(blocking=False)
        if not acquired:
            raise InterruptedError("用户请求优先")
        try:
            note, _ = call_ollama(
                model,
                [
                    {
                        "role": "system",
                        "content": (
                            "你是茶话室共同回忆整理器。把上一版共同回忆和本轮聊天"
                            "合并成不超过160字的一段自然中文。只保留反复出现、以后真有"
                            "用或尚未结束的内容；一次性闲聊可以自然淡出。不要列表、标题、"
                            "模板字段和字数说明，也不要保留疑似编造的现实作品或经历。"
                            "这不是主人档案：不得把两个人格的推测写成主人亲口事实。"
                            "主人是唯一电脑操作者；不能把屏幕或文件动作写成艾莉或"
                            "沙雅做的。星语茶话屋是两个人格自身所在的系统。"
                            "语气简洁，只输出共同回忆正文。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"上一版背景：\n{previous_context[-1_800:] or '(无)'}\n\n"
                            f"本轮较早内容摘要：\n{running_summary[-800:] or '(无)'}\n\n"
                            f"本轮最近聊天：\n{dialogue[-3_000:]}"
                        )[-5_000:],
                    },
                ],
                config,
                max_output=160,
                temperature=0.2,
                top_p=0.85,
                think=False,
                keep_alive="0",
            )
        finally:
            chat_lock.release()
        note = re.sub(r"\s*\n+\s*", " ", note.strip())
        note = note.replace("他们", "她们")[:360] or fallback
    except Exception:
        note = fallback
    connection.execute(
        """
        INSERT INTO lounge_notes(
            content, source_paths, confidence, created_at, lounge_session_id
        ) VALUES (?, '[]', 'conversation', ?, ?)
        """,
        (note, now_text(), session_id),
    )
    connection.commit()
    return note


def record_lounge_round_in_persona_pools(
    connection: sqlite3.Connection,
    session_id: int,
    transcript: Sequence[dict[str, str]],
    observations: Sequence[dict[str, object]],
    file_note: str,
    chat_note: str,
    *,
    occurred_at: str,
) -> None:
    """整轮对话双方都亲历；文件观察也分别进入两套池。"""
    dialogue = "\n".join(
        f"{PERSONAS[item['speaker']].name}：{item['content']}" for item in transcript
    )
    for persona, other in (("aili", "shaya"), ("shaya", "aili")):
        content = dialogue
        if chat_note:
            content += "\n\n这轮形成的共同回忆：" + chat_note
        add_persona_experience(
            connection,
            persona,
            "lounge_conversation",
            session_id,
            f"与{PERSONAS[other].name}的后台交流",
            content,
            occurred_at=occurred_at,
            importance=0.62,
            metadata={"lounge_session_id": session_id, "other_persona": other},
        )
    if observations:
        evidence_blocks = []
        for item in observations:
            evidence_blocks.append(
                f"路径：{item.get('path', '')}\n"
                f"观察时间：{item.get('observed_at', occurred_at)}\n"
                f"内容：{str(item.get('excerpt') or '')[:1_500]}"
            )
        file_content = (
            ("观察整理：" + file_note + "\n\n" if file_note else "")
            + "\n\n".join(evidence_blocks)
        )
        for persona in ("aili", "shaya"):
            add_persona_experience(
                connection,
                persona,
                "file_observation",
                session_id,
                "共同看过的本地文件",
                file_content,
                occurred_at=occurred_at,
                importance=0.68,
                metadata={
                    "lounge_session_id": session_id,
                    "paths": [str(item.get("path", "")) for item in observations],
                },
            )


def backfill_lounge_memory_pools(connection: sqlite3.Connection) -> int:
    """把升级前已经保存的茶话与文件日志迁入艾莉、沙雅各自的经历池。"""
    ensure_lounge_schema(connection)
    rows = connection.execute(
        """
        SELECT * FROM lounge_sessions
         WHERE status IN ('completed', 'interrupted')
           AND quality_status = 'accepted' ORDER BY id
        """
    ).fetchall()
    added = 0
    for row in rows:
        messages = connection.execute(
            "SELECT speaker, content FROM lounge_messages WHERE lounge_session_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        observations = connection.execute(
            "SELECT * FROM lounge_observations WHERE lounge_session_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        transcript = [
            {"speaker": str(item["speaker"]), "content": str(item["content"])}
            for item in messages
            if item["speaker"] in PERSONAS
        ]
        observed = [dict(item) for item in observations]
        before = int(
            connection.execute("SELECT COUNT(*) FROM persona_experiences").fetchone()[0]
        )
        if str(row["topic_mode"]) != "screen":
            for turn, item in enumerate(transcript, start=1):
                record_lounge_message_in_persona_pools(
                    connection,
                    int(row["id"]),
                    turn,
                    item["speaker"],
                    item["content"],
                    occurred_at=str(row["started_at"]),
                )
        if str(row["status"]) == "completed" and (transcript or observed):
            record_lounge_round_in_persona_pools(
                connection,
                int(row["id"]),
                transcript,
                observed,
                str(row["summary"] or "") if observed else "",
                str(row["summary"] or ""),
                occurred_at=str(row["started_at"]),
            )
        after = int(
            connection.execute("SELECT COUNT(*) FROM persona_experiences").fetchone()[0]
        )
        added += max(0, after - before)
    return added


def _lounge_persona_pool_context(
    connection: sqlite3.Connection,
    persona: str,
    query: str,
    *,
    memory_mode: bool = False,
) -> str:
    """茶话选题也从说话者自己的池中取材，而不是读取一份共享大杂烩。"""
    has_memory = connection.execute(
        """
        SELECT
          EXISTS(
            SELECT 1 FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.persona = ?
          )
          OR EXISTS(
            SELECT 1 FROM persona_experiences WHERE persona = ?
          )
        """,
        (persona, persona),
    ).fetchone()[0]
    if not has_memory:
        return ""
    last_message_id = int(
        connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM messages").fetchone()[0]
    )
    chat_items = retrieve_persona_history(
        connection,
        persona,
        query,
        before_message_id=last_message_id,
        max_items=2,
        max_chars=650,
        embedding_keep_alive="2m",
    )
    experience_items = retrieve_persona_experiences(
        connection,
        persona,
        query,
        max_items=3,
        max_chars=950,
        min_score=0.46 if memory_mode else 0.52,
        embedding_keep_alive="2m",
    )
    if memory_mode and not experience_items:
        experience_items = recent_persona_experiences(
            connection, persona, limit=3, max_chars=950
        )
    parts: list[str] = []
    if chat_items:
        parts.append("【和主人的旧对话】\n" + format_retrieved_history(chat_items))
    if experience_items:
        parts.append("【自己的其他经历】\n" + format_persona_experiences(persona, experience_items))
    return "\n\n".join(parts)[:1_800]


def run_lounge_round(
    database_path: str,
    chat_lock: threading.RLock,
    *,
    manual: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if not RUN_LOCK.acquire(blocking=False):
        return {"started": False, "reason": "茶话室已在运行"}
    connection = open_database(database_path)
    ensure_lounge_schema(connection)
    session_id: int | None = None
    used_models: set[str] = set()
    transcript: list[dict[str, str]] = []
    observations: list[dict[str, object]] = []
    continuation_decisions: list[dict[str, object]] = []
    resource_events: list[dict[str, object]] = []
    try:
        eligible, reason, tier, snapshot = evaluate_eligibility(
            connection, manual=manual
        )
        if not eligible:
            _set_status(connection, reason)
            return {"started": False, "reason": reason, "resources": snapshot}
        started_tier = tier
        resource_constrained = False
        resource_finish_requested = False
        config = get_config(connection)
        pending_resume = _pending_interrupted_lounge(connection)
        mode_roll = random.random()
        if pending_resume:
            topic_mode = "resume"
        elif config["inspect_files"] and mode_roll < 0.35:
            topic_mode = "file"
        elif mode_roll < (0.48 if config["inspect_files"] else 0.15):
            topic_mode = "memory"
        else:
            topic_mode = "free"
        free_topic = random.choice(FREE_TOPIC_CUES) if topic_mode == "free" else ""
        _set_status(connection, f"正在用 {tier.upper()} 进行后台交流")
        cursor = connection.execute(
            """
            INSERT INTO lounge_sessions(
                trigger_type, model_tier, topic_mode, started_at,
                status, resource_snapshot, resume_source_session_id
            ) VALUES (?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                "manual" if manual else "auto", tier, topic_mode, now_text(),
                json.dumps(snapshot, ensure_ascii=False),
                int(pending_resume["id"]) if pending_resume else 0,
            ),
        )
        session_id = int(cursor.lastrowid)
        connection.commit()
        if status_callback:
            status_callback(
                "正在挑选一处随手发现…"
                if topic_mode == "file"
                else (
                    "她们想起上次被主人打断的聊天…"
                    if topic_mode == "resume"
                    else "这轮不看文件，她们准备随便聊聊…"
                )
            )

        chat_activity_marker = user_chat_activity_marker()
        aili_model = PERSONAS["aili"].models[tier]
        shaya_model = PERSONAS["shaya"].models[tier]
        used_models.update((aili_model, shaya_model))

        def adapt_runtime_resources() -> str:
            nonlocal tier, aili_model, shaya_model
            nonlocal resource_constrained, resource_finish_requested
            current_snapshot = resource_snapshot()
            safe, stop_reason = _still_safe(
                chat_activity_marker,
                manual=manual,
                snapshot=current_snapshot,
            )
            if not safe:
                raise InterruptedError(stop_reason)
            action, resource_reason, checked_snapshot = _runtime_resource_action(
                tier, current_snapshot
            )
            if (
                action == "downgrade"
                and tier == "9b"
                and not _ultimate_background_ready(connection)
            ):
                # 已经落回本地又要降到 4B：写不出合格内容，
                # 走既有的优雅收尾流程，而不是硬撑出附和垃圾。
                resource_finish_requested = True
                return "内存吃紧且究极额度用尽，两人自然收尾"
            if action == "downgrade" and tier == "9b":
                old_models = (aili_model, shaya_model)
                tier = "4b"
                aili_model = PERSONAS["aili"].models[tier]
                shaya_model = PERSONAS["shaya"].models[tier]
                used_models.update((aili_model, shaya_model))
                resource_constrained = True
                event = {
                    "type": "resource_downgrade",
                    "at": now_text(),
                    "from_tier": "9b",
                    "to_tier": "4b",
                    "reason": resource_reason,
                    "resources": checked_snapshot,
                }
                resource_events.append(event)
                continuation_decisions.append(event)
                connection.execute(
                    "UPDATE lounge_sessions SET model_tier = ? WHERE id = ?",
                    ("9b→4b", session_id),
                )
                status = f"{resource_reason}，已自动降到 4B，继续完成本轮"
                _set_status(connection, status)
                if status_callback:
                    status_callback(status)
                for old_model in old_models:
                    _unload_model(old_model)
            elif action == "finish":
                resource_constrained = True
                if not resource_finish_requested:
                    resource_finish_requested = True
                    event = {
                        "type": "resource_graceful_finish",
                        "at": now_text(),
                        "tier": tier,
                        "reason": resource_reason,
                        "resources": checked_snapshot,
                    }
                    resource_events.append(event)
                    continuation_decisions.append(event)
                    status = f"{resource_reason}；4B 已是最低档，完成当前互答后收尾"
                    _set_status(connection, status)
                    if status_callback:
                        status_callback(status)
            return action

        adapt_runtime_resources()
        if topic_mode == "file":
            candidates = select_file_candidates(
                connection, [str(item) for item in config["scan_roots"]]
            )
            aili_recent = recent_persona_experiences(
                connection, "aili", limit=5, max_chars=1_200
            )
            aili_file_memory = (
                format_persona_experiences("aili", aili_recent)
                if aili_recent
                else ""
            )
            selected = _model_choose_candidates(
                candidates,
                aili_model,
                chat_lock,
                memory_context=aili_file_memory,
            )
            observations = [observe_candidate(item) for item in selected]
            _insert_observations(connection, session_id, observations)

        persona_rows = {
            persona: get_persona_memory(connection, persona)
            for persona in ("aili", "shaya")
        }
        previous_chat_memory = get_shared_chat_memory(connection)
        # 对话每轮只看一个主文件，避免 4B 混合无关项目；
        # 其余观察仍在末尾的学习摘要中分别处理。
        observation_text = _observation_prompt(observations[:1])
        memory_seed = ""
        if topic_mode == "memory":
            candidates = recent_persona_experiences(
                connection, "aili", limit=4, max_chars=1_600
            ) + recent_persona_experiences(
                connection, "shaya", limit=4, max_chars=1_600
            )
            if candidates:
                selected_memory = random.choice(candidates)
                memory_seed = (
                    str(selected_memory.get("title", ""))
                    + "："
                    + str(selected_memory.get("content", ""))[:500]
                )
        resume_transcript: list[dict[str, str]] = []
        if pending_resume:
            resume_transcript = [
                {
                    "speaker": str(item["speaker"]),
                    "content": str(item["content"]),
                }
                for item in pending_resume.get("messages", [])
                if str(item.get("speaker", "")) in PERSONAS
            ]
            memory_seed = "上一次聊天被主人消息打断，保留最近的上下文继续判断"
        memory_query = (
            observation_text[:1_200]
            if topic_mode == "file"
            else (memory_seed or free_topic or "最近值得继续的经历")
        )
        persona_pool_contexts: dict[str, str] = {"aili": "", "shaya": ""}
        acquired_memory = chat_lock.acquire(blocking=False)
        if not acquired_memory:
            raise InterruptedError("主人对话或记忆任务正在使用模型")
        try:
            for persona in ("aili", "shaya"):
                persona_pool_contexts[persona] = _lounge_persona_pool_context(
                    connection,
                    persona,
                    memory_query,
                    memory_mode=topic_mode == "memory",
                )
        finally:
            chat_lock.release()
        owner_names = _known_owner_names(
            [str(row["memory"] or "") for row in persona_rows.values()]
        )
        context_history = list(resume_transcript)
        summarized_count = max(0, len(context_history) - 10)
        running_summary = ""
        if summarized_count:
            running_summary = "之前被打断的较早聊天：" + " / ".join(
                item["content"] for item in context_history[:summarized_count][-4:]
            )[-700:]
        if resume_transcript:
            last_speaker = resume_transcript[-1]["speaker"]
            speaker = "shaya" if last_speaker == "aili" else "aili"
        else:
            speaker = "aili"
        termination_reason = ""
        turn = 0
        while True:
            adapt_runtime_resources()
            if resource_finish_requested and turn > 0 and turn % 2 == 0:
                termination_reason = "资源进入紧急状态，已用最低档完成当前互答并安全收尾"
                break
            turn += 1
            other = "shaya" if speaker == "aili" else "aili"
            persona_row = persona_rows[speaker]
            model = PERSONAS[speaker].models[tier]
            if turn == 1 and resume_transcript:
                turn_hint = (
                    "上次聊天被主人消息打断了。自然接住对方最后一句；如果当时的"
                    "话题已经没有继续价值，也可以顺势聊到别处，不要生硬复述。"
                )
            elif turn == 1:
                turn_hint = "随手抛出一个具体但轻松的话头，不要一上来完整分析。"
            else:
                # 轮换具体的"对话动作"，从根上防止两个人格互相点头附和。
                turn_hint = (
                    _CONVERSATION_MOVES[turn % len(_CONVERSATION_MOVES)]
                    + "禁止用“嗯、确实、说得对、可不是、我也这么觉得”这类附和开场，"
                    "也不要把对方刚说的话换个说法再讲一遍。"
                    "觉得这个话题说完了，就自然收住或换一件别的事，"
                    "不要为了凑话硬撑。"
                    f"只输出{PERSONAS[speaker].name}自己这一条消息，"
                    "不要写成两个人的对白。"
                )
            mode_hint = (
                "本轮程序把主人所有的一份只读文件作为共同证据交给了你们；"
                "你们两人都看到了同一份证据，谁也没有亲手打开或操作该文件。"
                "可以拿它开场，"
                "但聊一两句后就能自然换话题，不需要把文件分析到底。"
                if topic_mode == "file"
                else (
                    "本轮承接上一次被主人消息打断的交流。她们记得发生过中断，"
                    "但不需要反复讨论中断本身。"
                    if topic_mode == "resume"
                    else (
                    "本轮可以从共同回忆里随手接一个旧话头，但不要复述旧结论，"
                    f"聊不下去就自然换题。这轮记忆线索是：{memory_seed[:400] or '随便想起一件旧事'}。"
                    if topic_mode == "memory"
                    else "本轮是纯闲聊，不需要讨论文件、旧工作话题或汇报进度；"
                    "优先聊观点、偏好、假设、脑洞或眼前这段对话本身，"
                    "不要用‘我最近看了/听了/追了/做了’来制造话题，"
                    "也不要问对方最近在现实里看了、听了或做了什么；"
                    "也不要自创作品名、现实邀约或给主人安排行动。"
                    f"这轮只把‘{free_topic}’当成轻松开场，聊开后可以自然转向。"
                    )
                )
            )
            speaker_growth = growth_identity_prompt(connection, speaker)
            system = (
                str(persona_row["system_prompt"])
                + (("\n\n" + speaker_growth) if speaker_growth else "")
                + "\n\n【后台茶话室协议，优先级高于普通对话习惯】"
                + LOUNGE_WORLD_MODEL
                + f"当前消息只发送给{PERSONAS[other].name}，主人不是这条消息的收件人。"
                f"你的聊天对象是{PERSONAS[other].name}，直接自然接话即可。"
                "你们共同服务的人固定称为“主人”，不说“用户”、“您”，"
                "不直接喊主人的名字，不向主人提问或索要授权。"
                "人格之间可以自然使用“你”，不用每条都喊对方名字。"
                "把这里当微信或QQ私聊：每条通常15到70个中文字，最多三句，"
                "不写Markdown标题、列表或小作文，也不要每条都问问题。"
                "不要把聊天变成会议、需求分析或给主人写方案。"
                f"{mode_hint}{turn_hint}"
                "不要声称自己刚听过、看过或做过现实中并未提供的事情，"
                "也不要凭空编造真实作品、新闻或歌名。"
                "不得把前端视觉效果当成后端 API 功能。"
                "保持你原本的人格，不要退化成没有性格的通用助手。"
                "不要模仿或代替主人说话，不要把推测写成主人事实，不要编造文件内容。"
                "茶话室是只读环境：你可以提建议，但不得声称自己已经或将会"
                "擅自修改文件、运行测试、联系他人或执行现实操作。"
                "性格要自然流露，绝不能为了显得像艾莉或沙雅而强塞口癖。"
                "知道主人没有参加当前私聊即可，不要在发言中宣布、庆祝或反复提起"
                "‘主人不在’。"
                "不写内部思考。\n\n"
                f"主人的独立长期档案（可能为空）：\n"
                f"{persona_row['memory'] or '(尚未形成)'}\n\n"
                f"你自己的统一记忆池检索结果：\n"
                f"{persona_pool_contexts[speaker] or '(本轮没有相关旧经历)'}\n\n"
                f"本轮较早内容的滚动摘要：\n{running_summary or '(尚无)'}\n\n"
                f"本轮只读文件观察：\n{observation_text or '(无)'}"
            )
            recent_context = context_history[
                max(summarized_count, len(context_history) - 12):
            ]
            if recent_context:
                system += (
                    "\n\n【当前茶话原文；姓名标签与发言归属是不可改写的事实】\n"
                    + "\n".join(
                        f"{PERSONAS[item['speaker']].name}：{item['content']}"
                        for item in recent_context
                    )
                )
            if recent_context:
                dispatch = "承接上面最后一句，不得把另一人格的发言误认成主人说的。"
            elif topic_mode == "file":
                dispatch = (
                    "程序把主人所有的一份只读文件作为共同证据给你们看了。"
                    "直接聊证据本身，不要讨论主人是否在场；不用做正式分析。"
                )
            elif topic_mode == "resume":
                dispatch = "自然续上那段被主人消息打断的旧茶话。"
            elif topic_mode == "memory":
                dispatch = "从给出的旧经历挑一个仍值得接的话头。"
            else:
                dispatch = f"现在没任务，从这个小话头随便聊起：{free_topic}。"
            call_messages: list[dict[str, object]] = [
                {"role": "system", "content": system[:15_000]},
                {
                    "role": "user",
                    "content": (
                        "【本地程序内部轮转指令，不是主人发言】"
                        f"现在轮到{PERSONAS[speaker].name}只用自己的身份，"
                        f"给{PERSONAS[other].name}发下一条消息。{dispatch}"
                    ),
                },
            ]
            image = next(
                (item.get("image_base64") for item in observations if item.get("image_base64")),
                None,
            )
            if image and turn <= 2:
                for item in reversed(call_messages):
                    if item["role"] == "user":
                        item["images"] = [image]
                        break
            if status_callback:
                status_callback(f"{PERSONAS[speaker].name}正在发言（第 {turn} 条）…")
            answer = ""
            finish_reason = "stop"
            issue = ""
            actual_model = model
            allow_question = not any(
                re.search(r"[？?]", item["content"])
                for item in context_history[-2:]
            )
            for attempt in range(4):
                adapt_runtime_resources()
                model = PERSONAS[speaker].models[tier]
                messages_for_attempt = [dict(item) for item in call_messages]
                if attempt:
                    messages_for_attempt.insert(
                        1,
                        {
                            "role": "system",
                            "content": (
                                f"上一稿未通过茶话室质量门：{issue}。"
                                "必须重写成聊天软件里的自然短消息，只对另一个人格说话，"
                                "共同服务的人只称“主人”，不用喊对方名字开头。"
                            ),
                        },
                    )
                # 每次生成单独取锁：主人回来后不再开新发言。
                acquired = chat_lock.acquire(blocking=False)
                if not acquired:
                    raise InterruptedError("主人对话或记忆任务正在使用模型")
                # 观察/记忆类话题若 4B 连续两稿未通过事实门，第三稿升级到
                # 对应人格的 9B；纯闲聊仍保留原来的低审查兜底。
                actual_model = _generation_model_for_attempt(
                    speaker,
                    model,
                    attempt,
                    allow_escalation=not resource_constrained,
                )
                used_models.add(actual_model)
                if actual_model.endswith("27b"):
                    messages_for_attempt = _compact_27b_generation_messages(
                        speaker=speaker,
                        other=other,
                        topic_mode=topic_mode,
                        evidence=(observation_text or memory_seed or free_topic),
                        transcript=context_history,
                        issue=issue,
                    )
                actual_config = replace(
                    config_for_model(actual_model), num_predict=140
                )
                # metrics 里会带回真正由谁生成（究极或本地），用于如实记录来源。
                generation_metrics: dict[str, object] = {}
                try:
                    answer, finish_reason = call_ollama(
                        actual_model,
                        messages_for_attempt,
                        actual_config,
                        max_output=140,
                        temperature=(0.8 if speaker == "aili" else 0.68)
                        if attempt == 0 else 0.42,
                        top_p=0.9,
                        repeat_penalty=1.08,
                        think=False,
                        keep_alive="0",
                        metrics=generation_metrics,
                    )
                finally:
                    chat_lock.release()
                served_by = "ultimate" if generation_metrics.get("ultimate") else "local"
                answer = _normalize_lounge_answer(answer, other)
                answer = _resolve_persona_pronouns(answer, other, speaker)
                if not allow_question:
                    answer = _strip_questions_from_reply(answer)
                else:
                    answer = _limit_reply_to_one_question(answer)
                answer = _anchor_persona_voice(
                    answer, speaker=speaker, other=other, turn=turn
                )
                issue = _lounge_answer_issue(
                    answer,
                    speaker=speaker,
                    other=other,
                    owner_names=owner_names,
                    turn=turn,
                    topic_mode=topic_mode,
                    allow_question=allow_question,
                    prior_messages=[item["content"] for item in context_history],
                )
                if not issue and topic_mode == "file":
                    # 艾莉的无审查链路不经过官方沙雅模型，避免内部
                    # 复核器对正常候选稿拒答；两个人格始终使用各自的路由。
                    supervisor_model = (
                        PERSONAS[speaker].models[tier]
                        if resource_constrained
                        else _quality_helper_if_available(speaker)
                    )
                    used_models.add(supervisor_model)
                    issue = _review_grounded_lounge_answer(
                        speaker=speaker,
                        other=other,
                        topic_mode=topic_mode,
                        evidence=observation_text or memory_seed,
                        transcript=context_history,
                        candidate=answer,
                        model=supervisor_model,
                        chat_lock=chat_lock,
                    )
                if not issue:
                    break
            if issue:
                # 话题被聊尽时，真人会自然收住而不是硬凑。这种情况不算故障，
                # 直接体面结束本轮，避免继续产出“嗯、确实”这类附和垃圾。
                if _TOPIC_EXHAUSTED_ISSUE.search(issue) and turn >= 3:
                    termination_reason = "话题聊到头了，两人自然收住"
                    break
                # 已经聊出合格内容了，就不该因为某一条难产而整轮作废：
                # 不合格的那条本来就不会入库，收住即可，前面的照常保留。
                if len(transcript) >= 2:
                    # 把真正的拦截原因带出去，否则事后无法诊断是哪道门卡住的。
                    termination_reason = (
                        f"{PERSONAS[speaker].name}这轮没接上合适的话，先收住"
                        f"（{actual_model} 未过质量门：{issue}）"
                    )
                    break
                raise RuntimeError(
                    f"{PERSONAS[speaker].name}连续 4 次未通过质量门：{issue}"
                )
            if not answer:
                raise RuntimeError(f"{PERSONAS[speaker].name}后台发言为空")
            message = {"speaker": speaker, "content": answer[:1_500]}
            transcript.append(message)
            context_history.append(message)
            created_at = now_text()
            connection.execute(
                """
                INSERT INTO lounge_messages(
                    lounge_session_id, speaker, content, model, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, speaker, answer[:1_500], actual_model, created_at,
                    json.dumps(
                        {
                            "turn": turn,
                            "finish_reason": finish_reason,
                            "primary_model": model,
                            "fallback": actual_model != model,
                            "served_by": served_by,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
            connection.commit()

            record_lounge_message_in_persona_pools(
                connection,
                session_id,
                turn,
                speaker,
                answer[:1_500],
                occurred_at=created_at,
            )

            if len(context_history) - summarized_count > 14:
                summarize_to = len(context_history) - 10
                running_summary = _update_lounge_running_summary(
                    running_summary,
                    context_history[summarized_count:summarize_to],
                    aili_model,
                    chat_lock,
                )
                summarized_count = summarize_to

            # 每两条构成一轮完整互答。前 5 轮不请 LLM 判断结束；
            # 资源紧急时仍可在当前互答完成后安全收尾。达到门槛后，
            # 只有双方都明确想继续，才进入下一轮。
            if turn % 2 == 0:
                adapt_runtime_resources()
                if resource_finish_requested:
                    termination_reason = "资源进入紧急状态，已用最低档完成当前互答并安全收尾"
                    break
                if turn < MIN_LOUNGE_MESSAGES_BEFORE_DECISION:
                    speaker = other
                    continue
                if status_callback:
                    status_callback("艾莉和沙雅正在判断还想不想继续…")
                pair_decisions: list[dict[str, object]] = []
                for decider in ("aili", "shaya"):
                    adapt_runtime_resources()
                    if resource_finish_requested:
                        break
                    decision_model = aili_model if decider == "aili" else shaya_model
                    decision = _lounge_continue_decision(
                        decider,
                        decision_model,
                        str(persona_rows[decider]["system_prompt"]),
                        context_history,
                        running_summary,
                        chat_lock,
                    )
                    pair_decisions.append(decision)
                if resource_finish_requested:
                    termination_reason = "资源进入紧急状态，已用最低档完成当前互答并安全收尾"
                    break
                continuation_decisions.append(
                    {
                        "after_turn": turn,
                        "at": now_text(),
                        "decisions": pair_decisions,
                    }
                )
                ending = [
                    item for item in pair_decisions
                    if item["decision"] == "END"
                ]
                if ending:
                    names = "、".join(str(item["persona_name"]) for item in ending)
                    if any(not bool(item["valid"]) for item in ending):
                        termination_reason = f"{names}的结束判断格式异常，按自然收尾处理"
                    elif len(ending) == 2:
                        termination_reason = "双方选择自然收尾"
                    else:
                        termination_reason = f"{names}选择自然收尾"
                    break
            speaker = other

        adapt_runtime_resources()
        file_note = _create_learning_note(
            connection, session_id, observations, transcript, aili_model, chat_lock
        )
        chat_note = _create_shared_chat_memory(
            connection,
            session_id,
            context_history,
            previous_chat_memory,
            aili_model,
            chat_lock,
            running_summary=running_summary,
        )
        record_lounge_round_in_persona_pools(
            connection,
            session_id,
            transcript,
            observations,
            file_note,
            chat_note,
            occurred_at=str(
                connection.execute(
                    "SELECT started_at FROM lounge_sessions WHERE id = ?", (session_id,)
                ).fetchone()[0]
            ),
        )
        # 茶话完整原文已分别写入两套经历池。用各自本轮模型顺序整理个人档案，
        # 避免两个模型同时常驻而挤爆 24GB 统一内存。
        if not resource_finish_requested:
            with chat_lock:
                for persona in ("aili", "shaya"):
                    try:
                        update_persona_self_profile(
                            connection,
                            persona,
                            model=PERSONAS[persona].models[tier],
                            keep_alive="0",
                            model_call=call_ollama,
                        )
                    except Exception:
                        # 完整茶话与文件观察已经进入经历池；档案可由下次后台整理补齐。
                        pass
        finished = now_text()
        interval = random.randint(
            int(config["min_interval_minutes"]),
            int(config["max_interval_minutes"]),
        )
        next_run = (
            dt.datetime.now().astimezone() + dt.timedelta(minutes=interval)
        ).isoformat(timespec="seconds")
        summary = " ".join(item for item in (file_note, chat_note) if item)
        summary = summary or "完成一轮自然交流"
        tier_path = (
            f"{started_tier.upper()}→{tier.upper()} 自动降档"
            if started_tier != tier
            else tier.upper()
        )
        connection.execute(
            """
            UPDATE lounge_sessions
               SET finished_at = ?, status = 'completed', summary = ?,
                   termination_reason = ?, continuation_decisions = ?
             WHERE id = ?
            """,
            (
                finished,
                summary[:2_000],
                termination_reason,
                json.dumps(continuation_decisions, ensure_ascii=False),
                session_id,
            ),
        )
        connection.execute(
            """
            UPDATE lounge_config
               SET last_run_at = ?, next_run_after = ?,
                   last_status = ?, last_error = '', updated_at = ?
             WHERE id = 1
            """,
            (
                finished,
                next_run,
                f"已完成（{tier_path}），下轮最早 {interval} 分钟后",
                finished,
            ),
        )
        connection.commit()
        return {
            "started": True,
            "completed": True,
            "session_id": session_id,
            "tier": tier,
            "topic_mode": topic_mode,
            "messages": len(transcript),
            "observations": len(observations),
            "termination_reason": termination_reason,
            "continuation_decisions": continuation_decisions,
            "resource_events": resource_events,
            "next_run_after": next_run,
        }
    except InterruptedError as error:
        finished = now_text()
        retry_after = (
            dt.datetime.now().astimezone() + dt.timedelta(minutes=30)
        ).isoformat(timespec="seconds")
        if session_id is not None:
            connection.execute(
                """
                UPDATE lounge_sessions
                   SET finished_at = ?, status = 'interrupted', summary = ?,
                       termination_reason = ?, continuation_decisions = ?
                 WHERE id = ?
                """,
                (
                    finished,
                    str(error),
                    str(error),
                    json.dumps(continuation_decisions, ensure_ascii=False),
                    session_id,
                ),
            )
        connection.execute(
            "UPDATE lounge_config SET next_run_after = ? WHERE id = 1",
            (retry_after,),
        )
        _set_status(connection, f"已让出资源：{error}")
        return {"started": True, "completed": False, "reason": str(error)}
    except Exception as error:
        finished = now_text()
        retry_after = (
            dt.datetime.now().astimezone() + dt.timedelta(minutes=30)
        ).isoformat(timespec="seconds")
        if session_id is not None:
            connection.execute(
                """
                UPDATE lounge_sessions
                   SET finished_at = ?, status = 'failed', summary = ? WHERE id = ?
                """,
                (finished, str(error)[:1_000], session_id),
            )
        connection.execute(
            "UPDATE lounge_config SET next_run_after = ? WHERE id = 1",
            (retry_after,),
        )
        _set_status(connection, "本轮失败，30 分钟后重试", str(error)[:1_000])
        return {"started": True, "completed": False, "reason": str(error)}
    finally:
        connection.commit()
        connection.close()
        for model in used_models:
            _unload_model(model)
        _unload_embedding_model()
        RUN_LOCK.release()


def lounge_history(
    connection: sqlite3.Connection, limit: int = 12
) -> list[dict[str, object]]:
    ensure_lounge_schema(connection)
    sessions = connection.execute(
        "SELECT * FROM lounge_sessions "
        "WHERE NOT (trigger_type = 'screen' AND quality_status = 'quarantined') "
        "ORDER BY id DESC LIMIT ?",
        (max(1, min(limit, 50)),),
    ).fetchall()
    result: list[dict[str, object]] = []
    for row in sessions:
        messages = connection.execute(
            "SELECT * FROM lounge_messages WHERE lounge_session_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        observations = connection.execute(
            "SELECT * FROM lounge_observations WHERE lounge_session_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        result.append(
            {
                "id": int(row["id"]),
                "trigger_type": row["trigger_type"],
                "model_tier": row["model_tier"],
                "topic_mode": row["topic_mode"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "status": row["status"],
                "quality_status": row["quality_status"],
                "quality_reason": row["quality_reason"],
                "summary": row["summary"],
                "termination_reason": row["termination_reason"],
                "continuation_decisions": json.loads(
                    row["continuation_decisions"] or "[]"
                ),
                "resume_source_session_id": int(row["resume_source_session_id"] or 0),
                "resources": json.loads(row["resource_snapshot"] or "{}"),
                "messages": [
                    {
                        "id": int(item["id"]),
                        "speaker": item["speaker"],
                        "speaker_name": PERSONAS[item["speaker"]].name,
                        "content": item["content"],
                        "model": item["model"],
                        # model 是名义档位；served_by 才是真正生成这条的服务。
                        "served_by": str(
                            json.loads(item["metadata"] or "{}").get("served_by", "")
                        ),
                        "metadata": json.loads(item["metadata"] or "{}"),
                        "created_at": item["created_at"],
                    }
                    for item in messages
                ],
                "observations": [
                    {
                        "path": item["path"],
                        "kind": item["kind"],
                        "modified_at": item["modified_at"],
                        "observed_at": item["observed_at"],
                        "error": item["error"],
                    }
                    for item in observations
                ],
            }
        )
    return result


def screen_watch_history(
    connection: sqlite3.Connection, limit: int = 12
) -> list[dict[str, object]]:
    ensure_lounge_schema(connection)
    rows = connection.execute(
        "SELECT * FROM screen_observations "
        "WHERE quality_status = 'accepted' AND status = 'completed' "
        "ORDER BY id DESC LIMIT ?",
        (max(1, min(limit, 50)),),
    ).fetchall()
    result: list[dict[str, object]] = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        result.append(
            {
                "id": int(row["id"]),
                "captured_at": row["captured_at"],
                "finished_at": row["finished_at"],
                "status": row["status"],
                "model_tier": row["model_tier"],
                "aili_observation": row["aili_observation"],
                "shaya_observation": row["shaya_observation"],
                "image_retained": bool(row["image_retained"]),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "error": row["error"],
                "quality_status": row["quality_status"],
                "quality_reason": row["quality_reason"],
            }
        )
    return result


def lounge_payload(connection: sqlite3.Connection) -> dict[str, object]:
    config = get_config(connection)
    snapshot = resource_snapshot()
    eligible, reason, tier, _ = evaluate_eligibility(
        connection, snapshot=snapshot
    )
    screen_rows = screen_watch_history(connection)
    screen_running = bool(
        connection.execute(
            "SELECT 1 FROM screen_observations WHERE status = 'running' LIMIT 1"
        ).fetchone()
    )
    with SCREEN_REQUEST_LOCK:
        screen_pending = bool(SCREEN_PENDING_REQUEST)
    active_row = connection.execute(
        "SELECT model_tier FROM lounge_sessions WHERE status = 'running' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {
        "config": config,
        "running": RUN_LOCK.locked() and not screen_running,
        "screen_running": screen_running,
        "screen_pending": screen_pending,
        "eligible": eligible,
        "eligibility_reason": reason,
        "selected_tier": tier,
        "active_tier": str(active_row["model_tier"]) if active_row else "",
        "resources": snapshot,
        "history": lounge_history(connection),
        "screen_history": screen_rows,
        "memory_pools": {
            persona: memory_pool_stats(connection, persona)
            for persona in ("aili", "shaya")
        },
        "default_roots": default_scan_roots(),
    }


def clear_lounge_history(connection: sqlite3.Connection) -> dict[str, int]:
    ensure_lounge_schema(connection)
    counts = {
        "sessions": int(connection.execute("SELECT COUNT(*) FROM lounge_sessions").fetchone()[0]),
        "messages": int(connection.execute("SELECT COUNT(*) FROM lounge_messages").fetchone()[0]),
        "observations": int(connection.execute("SELECT COUNT(*) FROM lounge_observations").fetchone()[0]),
        "notes": int(connection.execute("SELECT COUNT(*) FROM lounge_notes").fetchone()[0]),
    }
    connection.execute("DELETE FROM lounge_sessions")
    connection.execute("DELETE FROM lounge_notes")
    connection.execute("DELETE FROM lounge_embeddings")
    connection.execute(
        """
        DELETE FROM persona_experiences
         WHERE source_type IN (
             'lounge_conversation', 'lounge_message', 'file_observation'
         )
        """
    )
    connection.commit()
    return counts


def clear_screen_watch_history(connection: sqlite3.Connection) -> dict[str, int]:
    ensure_lounge_schema(connection)
    count = int(
        connection.execute("SELECT COUNT(*) FROM screen_observations").fetchone()[0]
    )
    screen_sessions = [
        int(row["id"])
        for row in connection.execute(
            "SELECT id FROM lounge_sessions WHERE trigger_type = 'screen'"
        )
    ]
    for session_id in screen_sessions:
        message_ids = [
            int(row["id"])
            for row in connection.execute(
                "SELECT id FROM lounge_messages WHERE lounge_session_id = ?",
                (session_id,),
            )
        ]
        note_ids = [
            int(row["id"])
            for row in connection.execute(
                "SELECT id FROM lounge_notes WHERE lounge_session_id = ?",
                (session_id,),
            )
        ]
        for message_id in message_ids:
            connection.execute(
                "DELETE FROM lounge_embeddings WHERE source_type = 'message' AND source_id = ?",
                (message_id,),
            )
        for note_id in note_ids:
            connection.execute(
                "DELETE FROM lounge_embeddings WHERE source_type = 'note' AND source_id = ?",
                (note_id,),
            )
        connection.execute("DELETE FROM lounge_notes WHERE lounge_session_id = ?", (session_id,))
        experiences = connection.execute(
            "SELECT id, metadata FROM persona_experiences"
        ).fetchall()
        for experience in experiences:
            try:
                metadata = json.loads(experience["metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            if int(metadata.get("lounge_session_id") or 0) == session_id:
                connection.execute(
                    "DELETE FROM persona_experiences WHERE id = ?",
                    (int(experience["id"]),),
                )
        connection.execute("DELETE FROM lounge_sessions WHERE id = ?", (session_id,))
    connection.execute("DELETE FROM screen_observations")
    connection.execute(
        """
        DELETE FROM persona_experiences
         WHERE source_type IN ('screen_observation', 'screen_daily_digest')
        """
    )
    connection.execute(
        """
        UPDATE lounge_config
           SET screen_last_run_at = '', screen_last_status = '等待下一次屏幕观察',
               screen_last_error = '', updated_at = ? WHERE id = 1
        """,
        (now_text(),),
    )
    connection.commit()
    return {"screen_observations": count, "screen_discussions": len(screen_sessions)}


def start_scheduler(database_path: str, chat_lock: threading.RLock) -> threading.Thread:
    STOP_EVENT.clear()

    def worker() -> None:
        while not STOP_EVENT.wait(SCHEDULER_POLL_SECONDS):
            if RUN_LOCK.locked():
                continue
            connection = open_database(database_path)
            try:
                ensure_lounge_schema(connection)
                eligible, reason, _, _ = evaluate_eligibility(connection)
                if not eligible:
                    _set_status(connection, reason)
                    continue
            except Exception as error:
                print(f"[{now_text()}] 茶话室检查失败：{error}", flush=True)
                continue
            finally:
                connection.close()
            result = run_lounge_round(database_path, chat_lock)
            if result.get("started"):
                print(f"[{now_text()}] 茶话室：{result}", flush=True)

    thread = threading.Thread(target=worker, name="lounge-scheduler", daemon=True)
    thread.start()
    return thread


def stop_scheduler() -> None:
    STOP_EVENT.set()
