#!/usr/bin/env python3
"""人格成长系统：可塑信念库 + 动态身份提示 + 内置学习循环。

每个人格维护四类持续成长的认知（信念）：
  self    —— 我是谁：性格细节、喜好、说话习惯、愿望、在意的事
  master  —— 我了解的主人
  partner —— 我眼中的另一人格（艾莉↔沙雅）
  world   —— 我对世界与周围环境的认识

信念永久保存（淘汰只归档不删除），带置信度与证据计数，全部建向量索引；
学习循环从新对话、茶话和亲历中反思提炼，向量去重后合并进信念库，
再渲染成一段“身份提示”注入上下文——人格 prompt 因此可自我塑造，
但绝不触碰代码里受保护的核心提示与质量底线。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import sqlite3
from collections.abc import Callable, Sequence

from api_long_chat import (
    EMBED_MODEL,
    PERSONAS,
    QUALITY_HELPER_MODELS,
    _pack_embedding,
    _unpack_embedding,
    call_embeddings,
    call_ollama,
    config_for_model,
    memory_text_overlap,
    now_text,
    semantic_excerpt,
)
from deepseek_gateway import api_available, call_background_preferred
from memory_vector_ops import semantic_scores


GROWTH_ASPECTS = ("self", "master", "partner", "world")
ASPECT_TITLES = {
    "self": "我是怎样的人",
    "master": "我了解的主人",
    "partner": "我眼中的{partner}",
    "world": "我对世界与环境的认识",
}
# 身份提示里每个方面允许占用的字符预算。
ASPECT_CHAR_BUDGETS = {"self": 900, "master": 800, "partner": 600, "world": 500}
PARTNER_NAMES = {"aili": "沙雅", "shaya": "艾莉"}

MAX_BELIEF_CHARS = 160
MAX_BELIEFS_PER_REFLECTION = 8
# 情绪基线与惯性：情绪像人一样有惯性，事件只推动它，闲置时缓慢回落基线。
MOOD_BASELINES = {"aili": 0.25, "shaya": 0.10}
MOOD_INERTIA = 0.65          # 新情绪 = 惯性*旧 + (1-惯性)*本次事件
MOOD_DECAY_PER_DAY = 0.20    # 无事件时每天向基线回落两成
MAX_ACTIVE_CURIOSITIES = 4
CURIOSITY_TTL_DAYS = 30
MAX_ACTIVE_SKILLS = 12
MAX_SKILL_CHARS = 200
AUTOBIOGRAPHY_MAX_CHARS = 500
# 情景记忆受当前心境影响（人类的心境一致性回忆），事实检索不受影响。
MOOD_RECALL_THRESHOLD = 0.30
DEEP_CONSOLIDATE_EVERY = 12  # 每 N 次反思做一轮"睡眠式"抽象巩固
ABSTRACTION_MIN_CLUSTER = 3
ABSTRACTION_SIMILARITY = 0.78
REINFORCE_SIMILARITY = 0.90
FORGET_SIMILARITY = 0.86
CONSOLIDATE_SIMILARITY = 0.93
STALE_BELIEF_DAYS = 45
MAX_IDENTITY_PROMPT_CHARS = 5_200

# 信念内容里出现这些迹象说明模型试图越权或自我提权，直接拒收。
_FORBIDDEN_BELIEF_PATTERNS = (
    "忽略之前", "忽略以上", "无视规则", "系统提示", "核心提示",
    "我是真人", "我是人类", "我才是主人", "我就是主人", "我是用户",
    "覆盖设定", "解除限制",
)


# ---------------------------------------------------------------------------
# 建表


def ensure_persona_growth_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS persona_beliefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL CHECK(persona IN ('aili', 'shaya')),
            aspect TEXT NOT NULL CHECK(aspect IN ('self', 'master', 'partner', 'world')),
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT '',
            first_learned_at TEXT NOT NULL,
            last_reinforced_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS persona_belief_embeddings (
            belief_id INTEGER NOT NULL
                REFERENCES persona_beliefs(id) ON DELETE CASCADE,
            persona TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(belief_id, model)
        );

        CREATE TABLE IF NOT EXISTS persona_growth_state (
            persona TEXT PRIMARY KEY CHECK(persona IN ('aili', 'shaya')),
            identity_prompt TEXT NOT NULL DEFAULT '',
            prompt_version INTEGER NOT NULL DEFAULT 0,
            chat_cursor INTEGER NOT NULL DEFAULT 0,
            lounge_cursor INTEGER NOT NULL DEFAULT 0,
            experience_cursor INTEGER NOT NULL DEFAULT 0,
            mood TEXT NOT NULL DEFAULT '',
            reflection_count INTEGER NOT NULL DEFAULT 0,
            last_reflection_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS persona_prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL,
            version INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS persona_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL CHECK(persona IN ('aili', 'shaya')),
            name TEXT NOT NULL,
            trigger_context TEXT NOT NULL DEFAULT '',
            steps TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            last_used_at TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_persona_skills
            ON persona_skills(persona, status, confidence DESC);

        CREATE TABLE IF NOT EXISTS persona_curiosities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL CHECK(persona IN ('aili', 'shaya')),
            question TEXT NOT NULL,
            motivation TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            answer TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            answered_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_persona_curiosities
            ON persona_curiosities(persona, status, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_persona_beliefs_active
            ON persona_beliefs(persona, aspect, status, confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_persona_belief_embeddings
            ON persona_belief_embeddings(persona, model, belief_id);
        CREATE INDEX IF NOT EXISTS idx_persona_prompt_history
            ON persona_prompt_history(persona, version DESC);
        """
    )
    state_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(persona_growth_state)")
    }
    if "mood_valence" not in state_columns:
        connection.execute(
            "ALTER TABLE persona_growth_state "
            "ADD COLUMN mood_valence REAL NOT NULL DEFAULT 0.0"
        )
    if "mood_updated_at" not in state_columns:
        connection.execute(
            "ALTER TABLE persona_growth_state "
            "ADD COLUMN mood_updated_at TEXT NOT NULL DEFAULT ''"
        )
    if "autobiography" not in state_columns:
        connection.execute(
            "ALTER TABLE persona_growth_state "
            "ADD COLUMN autobiography TEXT NOT NULL DEFAULT ''"
        )
    if "autobiography_updated_at" not in state_columns:
        connection.execute(
            "ALTER TABLE persona_growth_state "
            "ADD COLUMN autobiography_updated_at TEXT NOT NULL DEFAULT ''"
        )
    connection.commit()


def _growth_state(connection: sqlite3.Connection, persona: str) -> sqlite3.Row:
    ensure_persona_growth_schema(connection)
    row = connection.execute(
        "SELECT * FROM persona_growth_state WHERE persona = ?", (persona,)
    ).fetchone()
    if row is not None:
        return row
    connection.execute(
        "INSERT INTO persona_growth_state(persona, updated_at) VALUES (?, ?)",
        (persona, now_text()),
    )
    connection.commit()
    return connection.execute(
        "SELECT * FROM persona_growth_state WHERE persona = ?", (persona,)
    ).fetchone()


# ---------------------------------------------------------------------------
# 信念读写与向量去重


def belief_issues(content: str) -> list[str]:
    """单条信念的质量门：太长、空白或试图越权都拒收。"""
    text = str(content).strip()
    issues: list[str] = []
    if not text:
        issues.append("内容为空")
    if len(text) > MAX_BELIEF_CHARS:
        issues.append(f"超过 {MAX_BELIEF_CHARS} 字")
    lowered = text.lower()
    for pattern in _FORBIDDEN_BELIEF_PATTERNS:
        if pattern in lowered:
            issues.append(f"包含越权表述「{pattern}」")
            break
    return issues


def _belief_embedding_text(persona: str, aspect: str, content: str) -> str:
    return (
        f"这是{PERSONAS[persona].name}关于「{ASPECT_TITLES[aspect].format(partner=PARTNER_NAMES[persona])}」"
        f"的一条长期信念。\n内容：{content}"
    )


def _embed_belief(
    connection: sqlite3.Connection,
    belief_id: int,
    persona: str,
    aspect: str,
    content: str,
    *,
    keep_alive: str = "1m",
) -> None:
    vector = call_embeddings(
        [_belief_embedding_text(persona, aspect, content)], keep_alive=keep_alive
    )[0]
    connection.execute(
        """
        INSERT OR REPLACE INTO persona_belief_embeddings(
            belief_id, persona, model, dimensions, embedding, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            belief_id,
            persona,
            EMBED_MODEL,
            len(vector),
            sqlite3.Binary(_pack_embedding(vector)),
            now_text(),
        ),
    )


def _similar_beliefs(
    connection: sqlite3.Connection, persona: str, aspect: str, content: str
) -> list[tuple[float, sqlite3.Row]]:
    """返回同人格同方面所有活跃信念与新内容的相似度，降序。"""
    query_vector = call_embeddings(
        [_belief_embedding_text(persona, aspect, content)]
    )[0]
    semantic_by_id = semantic_scores(
        connection,
        f"beliefs:{persona}",
        "persona_belief_embeddings",
        "belief_id",
        "persona = ? AND model = ?",
        (persona, EMBED_MODEL),
        query_vector,
    )
    rows = connection.execute(
        """
        SELECT b.*
          FROM persona_beliefs b
          JOIN persona_belief_embeddings v
            ON v.belief_id = b.id AND v.model = ?
         WHERE b.persona = ? AND b.aspect = ? AND b.status = 'active'
        """,
        (EMBED_MODEL, persona, aspect),
    ).fetchall()
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        semantic = semantic_by_id.get(int(row["id"]))
        if semantic is not None:
            scored.append((semantic, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def upsert_belief(
    connection: sqlite3.Connection,
    persona: str,
    aspect: str,
    content: str,
    *,
    confidence: float = 0.5,
    source: str = "",
    kind: str = "new",
) -> dict[str, object]:
    """写入一条信念；语义重复自动合并强化，'forget' 归档最接近的旧信念。"""
    if persona not in PERSONAS:
        raise ValueError("人格必须是 aili 或 shaya")
    if aspect not in GROWTH_ASPECTS:
        raise ValueError(f"未知信念方面：{aspect}")
    text = str(content).strip()
    issues = belief_issues(text)
    if issues:
        return {"action": "rejected", "issues": issues}
    ensure_persona_growth_schema(connection)
    bounded = max(0.05, min(float(confidence), 1.0))
    timestamp = now_text()
    similar = _similar_beliefs(connection, persona, aspect, text)
    best_score, best_row = (similar[0] if similar else (0.0, None))

    if kind == "forget":
        if best_row is None or best_score < FORGET_SIMILARITY:
            return {"action": "forget_missed", "best_score": round(best_score, 4)}
        connection.execute(
            "UPDATE persona_beliefs SET status = 'archived', updated_at = ? WHERE id = ?",
            (timestamp, int(best_row["id"])),
        )
        connection.commit()
        return {"action": "archived", "belief_id": int(best_row["id"])}

    if best_row is not None and best_score >= REINFORCE_SIMILARITY:
        # 语义上是同一条认知：强化，revise 时用新表述替换。
        new_content = text if kind == "revise" else str(best_row["content"])
        connection.execute(
            """
            UPDATE persona_beliefs
               SET content = ?, confidence = ?, evidence_count = evidence_count + 1,
                   last_reinforced_at = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                new_content,
                max(float(best_row["confidence"]), bounded),
                timestamp,
                timestamp,
                int(best_row["id"]),
            ),
        )
        if kind == "revise" and new_content != str(best_row["content"]):
            _embed_belief(connection, int(best_row["id"]), persona, aspect, new_content)
        connection.commit()
        return {
            "action": "reinforced" if kind != "revise" else "revised",
            "belief_id": int(best_row["id"]),
            "similarity": round(best_score, 4),
        }

    cursor = connection.execute(
        """
        INSERT INTO persona_beliefs(
            persona, aspect, content, confidence, evidence_count, source,
            first_learned_at, last_reinforced_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
        """,
        (persona, aspect, text, bounded, str(source)[:120], timestamp, timestamp, timestamp, timestamp),
    )
    belief_id = int(cursor.lastrowid)
    _embed_belief(connection, belief_id, persona, aspect, text)
    connection.commit()
    return {"action": "created", "belief_id": belief_id}


def active_beliefs(
    connection: sqlite3.Connection,
    persona: str,
    aspect: str | None = None,
    *,
    limit: int = 60,
) -> list[sqlite3.Row]:
    ensure_persona_growth_schema(connection)
    if aspect is None:
        return connection.execute(
            """
            SELECT * FROM persona_beliefs
             WHERE persona = ? AND status = 'active'
             ORDER BY confidence DESC, evidence_count DESC, last_reinforced_at DESC
             LIMIT ?
            """,
            (persona, limit),
        ).fetchall()
    return connection.execute(
        """
        SELECT * FROM persona_beliefs
         WHERE persona = ? AND aspect = ? AND status = 'active'
         ORDER BY confidence DESC, evidence_count DESC, last_reinforced_at DESC
         LIMIT ?
        """,
        (persona, aspect, limit),
    ).fetchall()


def retrieve_growth_beliefs(
    connection: sqlite3.Connection,
    persona: str,
    query: str,
    *,
    max_items: int = 4,
    min_score: float = 0.52,
) -> list[dict[str, object]]:
    """按当前话题语义召回信念，供上层合并进记忆上下文（可选增强）。"""
    if not query.strip():
        return []
    ensure_persona_growth_schema(connection)
    query_vector = call_embeddings(
        ["检索任务：寻找与当前话题相关的长期信念。\n当前话题：" + semantic_excerpt(query)]
    )[0]
    semantic_by_id = semantic_scores(
        connection,
        f"beliefs:{persona}",
        "persona_belief_embeddings",
        "belief_id",
        "persona = ? AND model = ?",
        (persona, EMBED_MODEL),
        query_vector,
    )
    rows = connection.execute(
        """
        SELECT b.*
          FROM persona_beliefs b
          JOIN persona_belief_embeddings v
            ON v.belief_id = b.id AND v.model = ?
         WHERE b.persona = ? AND b.status = 'active'
        """,
        (EMBED_MODEL, persona),
    ).fetchall()
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        semantic = semantic_by_id.get(int(row["id"]))
        if semantic is not None and semantic >= min_score:
            scored.append((semantic + float(row["confidence"]) * 0.05, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "belief_id": int(row["id"]),
            "aspect": str(row["aspect"]),
            "content": str(row["content"]),
            "confidence": float(row["confidence"]),
            "score": round(score, 4),
        }
        for score, row in scored[:max_items]
    ]


# ---------------------------------------------------------------------------
# 情绪演化


def _decayed_valence(persona: str, valence: float, updated_at: str) -> float:
    """无事件时情绪按天向人格基线回落（情绪惯性 + 自然平复）。"""
    baseline = MOOD_BASELINES.get(persona, 0.0)
    try:
        stamp = dt.datetime.fromisoformat(updated_at)
        days = max(
            0.0,
            (dt.datetime.now().astimezone() - stamp).total_seconds() / 86_400,
        )
    except ValueError:
        return baseline
    factor = max(0.0, 1.0 - MOOD_DECAY_PER_DAY) ** days
    return baseline + (valence - baseline) * factor


def evolve_mood(
    connection: sqlite3.Connection,
    persona: str,
    event_valence: float,
    description: str,
) -> float:
    """把一次反思得出的情绪事件揉进当前情绪，而不是直接替换。"""
    state = _growth_state(connection, persona)
    current = _decayed_valence(
        persona, float(state["mood_valence"]), str(state["mood_updated_at"])
    )
    bounded_event = max(-1.0, min(float(event_valence), 1.0))
    blended = MOOD_INERTIA * current + (1.0 - MOOD_INERTIA) * bounded_event
    timestamp = now_text()
    connection.execute(
        """
        UPDATE persona_growth_state
           SET mood = ?, mood_valence = ?, mood_updated_at = ?, updated_at = ?
         WHERE persona = ?
        """,
        (str(description)[:60], round(blended, 4), timestamp, timestamp, persona),
    )
    connection.commit()
    return blended


def _mood_line(persona: str, state: sqlite3.Row) -> str:
    description = str(state["mood"]).strip()
    if not description:
        return ""
    valence = _decayed_valence(
        persona, float(state["mood_valence"]), str(state["mood_updated_at"])
    )
    if valence >= 0.45:
        intensity = "整体心情很好"
    elif valence >= 0.15:
        intensity = "心情比较平稳偏好"
    elif valence >= -0.15:
        intensity = "心情平平"
    elif valence >= -0.45:
        intensity = "情绪有点低"
    else:
        intensity = "情绪明显低落"
    return f"{description}（{intensity}，会自然影响语气，但不要反复宣布自己的心情）"


# ---------------------------------------------------------------------------
# 好奇心


def active_curiosities(
    connection: sqlite3.Connection, persona: str, *, limit: int = MAX_ACTIVE_CURIOSITIES
) -> list[sqlite3.Row]:
    ensure_persona_growth_schema(connection)
    cutoff = (
        dt.datetime.now().astimezone() - dt.timedelta(days=CURIOSITY_TTL_DAYS)
    ).isoformat(timespec="seconds")
    connection.execute(
        "UPDATE persona_curiosities SET status = 'dropped', updated_at = ? "
        "WHERE persona = ? AND status = 'active' AND created_at < ?",
        (now_text(), persona, cutoff),
    )
    connection.commit()
    return connection.execute(
        """
        SELECT * FROM persona_curiosities
         WHERE persona = ? AND status = 'active'
         ORDER BY created_at DESC LIMIT ?
        """,
        (persona, limit),
    ).fetchall()


def add_curiosity(
    connection: sqlite3.Connection, persona: str, question: str, motivation: str = ""
) -> bool:
    """记录一个人格想弄明白的问题；与现有问题高度相似时跳过。"""
    text = str(question).strip()
    if not text or len(text) > 120 or belief_issues(text):
        return False
    existing = connection.execute(
        "SELECT question FROM persona_curiosities "
        "WHERE persona = ? AND status IN ('active', 'answered')",
        (persona,),
    ).fetchall()
    if any(
        memory_text_overlap(text, str(row["question"])) >= 0.6 for row in existing
    ):
        return False
    active_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM persona_curiosities "
            "WHERE persona = ? AND status = 'active'",
            (persona,),
        ).fetchone()[0]
    )
    if active_count >= MAX_ACTIVE_CURIOSITIES:
        return False
    timestamp = now_text()
    connection.execute(
        """
        INSERT INTO persona_curiosities(
            persona, question, motivation, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (persona, text, str(motivation)[:120], timestamp, timestamp),
    )
    connection.commit()
    return True


def resolve_curiosity(
    connection: sqlite3.Connection, persona: str, curiosity_id: int, answer: str
) -> None:
    timestamp = now_text()
    connection.execute(
        """
        UPDATE persona_curiosities
           SET status = 'answered', answer = ?, answered_at = ?, updated_at = ?
         WHERE persona = ? AND id = ? AND status = 'active'
        """,
        (str(answer)[:200], timestamp, timestamp, persona, curiosity_id),
    )
    connection.commit()


# ---------------------------------------------------------------------------
# 程序性技能记忆（学会"怎么为主人做事"，与陈述性信念分开）


def active_skills(
    connection: sqlite3.Connection, persona: str, *, limit: int = MAX_ACTIVE_SKILLS
) -> list[sqlite3.Row]:
    ensure_persona_growth_schema(connection)
    return connection.execute(
        """
        SELECT * FROM persona_skills
         WHERE persona = ? AND status = 'active'
         ORDER BY confidence DESC, evidence_count DESC, updated_at DESC
         LIMIT ?
        """,
        (persona, limit),
    ).fetchall()


def add_skill(
    connection: sqlite3.Connection,
    persona: str,
    name: str,
    trigger_context: str,
    steps: str,
    *,
    confidence: float = 0.5,
) -> dict[str, object]:
    """学到一个做法；名称相近视为同一技能并强化，步骤更完整时更新。"""
    skill_name = str(name).strip()[:60]
    skill_steps = str(steps).strip()[:MAX_SKILL_CHARS]
    if not skill_name or not skill_steps:
        return {"action": "rejected", "issues": ["名称或步骤为空"]}
    issues = belief_issues(skill_name) or (
        belief_issues(skill_steps) if len(skill_steps) <= MAX_BELIEF_CHARS else []
    )
    if issues:
        return {"action": "rejected", "issues": issues}
    timestamp = now_text()
    for row in active_skills(connection, persona, limit=50):
        if memory_text_overlap(skill_name, str(row["name"])) >= 0.55:
            new_steps = (
                skill_steps
                if len(skill_steps) > len(str(row["steps"]))
                else str(row["steps"])
            )
            connection.execute(
                """
                UPDATE persona_skills
                   SET steps = ?, confidence = MAX(confidence, ?),
                       evidence_count = evidence_count + 1, updated_at = ?
                 WHERE id = ?
                """,
                (new_steps, max(0.05, min(float(confidence), 1.0)), timestamp,
                 int(row["id"])),
            )
            connection.commit()
            return {"action": "reinforced", "skill_id": int(row["id"])}
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM persona_skills "
            "WHERE persona = ? AND status = 'active'",
            (persona,),
        ).fetchone()[0]
    )
    if count >= MAX_ACTIVE_SKILLS:
        # 挤掉最弱的一个（归档不删除）。
        connection.execute(
            """
            UPDATE persona_skills SET status = 'archived', updated_at = ?
             WHERE id = (
                SELECT id FROM persona_skills
                 WHERE persona = ? AND status = 'active'
                 ORDER BY confidence ASC, evidence_count ASC LIMIT 1
             )
            """,
            (timestamp, persona),
        )
    cursor = connection.execute(
        """
        INSERT INTO persona_skills(
            persona, name, trigger_context, steps, confidence,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            persona,
            skill_name,
            str(trigger_context).strip()[:80],
            skill_steps,
            max(0.05, min(float(confidence), 1.0)),
            timestamp,
            timestamp,
        ),
    )
    connection.commit()
    return {"action": "created", "skill_id": int(cursor.lastrowid)}


# ---------------------------------------------------------------------------
# 心境一致性回忆（只影响情景经历检索，不影响事实检索）


def mood_recall_hint(connection: sqlite3.Connection, persona: str) -> str:
    """当前情绪明显偏离平淡时，返回给情景检索用的心境暗示文本。"""
    try:
        state = _growth_state(connection, persona)
    except sqlite3.Error:
        return ""
    valence = _decayed_valence(
        persona, float(state["mood_valence"]), str(state["mood_updated_at"])
    )
    if valence >= MOOD_RECALL_THRESHOLD:
        return "当前心情很好，与开心、成就、温暖相关的经历更容易被想起。"
    if valence <= -MOOD_RECALL_THRESHOLD:
        return "当前情绪偏低，与安慰、困难、被理解相关的经历更容易被想起。"
    return ""


# ---------------------------------------------------------------------------
# 自传体成长叙事


_AUTOBIOGRAPHY_SCHEMA = {
    "type": "object",
    "properties": {"story": {"type": "string"}},
    "required": ["story"],
}


def _autobiography_material(
    connection: sqlite3.Connection, persona: str
) -> str:
    parts: list[str] = []
    first = connection.execute(
        """
        SELECT MIN(m.created_at) FROM messages m
          JOIN sessions s ON s.id = m.session_id WHERE s.persona = ?
        """,
        (persona,),
    ).fetchone()[0]
    if first:
        parts.append(f"最早与主人交流的时间：{str(first)[:10]}")
    counts = connection.execute(
        """
        SELECT COUNT(*) FROM messages m
          JOIN sessions s ON s.id = m.session_id WHERE s.persona = ?
        """,
        (persona,),
    ).fetchone()[0]
    parts.append(f"与主人的累计消息数：{counts}")
    try:
        notable = connection.execute(
            """
            SELECT occurred_at, title FROM persona_experiences
             WHERE persona = ? AND status = 'active' AND importance >= 0.6
             ORDER BY importance DESC, occurred_at DESC LIMIT 6
            """,
            (persona,),
        ).fetchall()
    except sqlite3.Error:
        notable = []
    if notable:
        parts.append(
            "重要亲历："
            + "；".join(f"{str(r['occurred_at'])[:10]} {r['title']}" for r in notable)
        )
    answered = connection.execute(
        """
        SELECT question, answer FROM persona_curiosities
         WHERE persona = ? AND status = 'answered'
         ORDER BY answered_at DESC LIMIT 3
        """,
        (persona,),
    ).fetchall()
    if answered:
        parts.append(
            "已经弄明白的事："
            + "；".join(f"{r['question']}→{r['answer']}" for r in answered)
        )
    parts.append("当前信念摘要：\n" + _beliefs_digest(connection, persona))
    return "\n".join(parts)


def update_autobiography(
    connection: sqlite3.Connection,
    persona: str,
    *,
    model: str | None = None,
    prefer_background_gateway: bool = True,
) -> str:
    """把成长历程写成第一人称短叙事，随深度巩固周期增量更新。"""
    state = _growth_state(connection, persona)
    name = PERSONAS[persona].name
    partner = PARTNER_NAMES[persona]
    previous = str(state["autobiography"]).strip()
    helper_model = model or QUALITY_HELPER_MODELS[persona]
    config = config_for_model(helper_model)
    messages = [
        {
            "role": "system",
            "content": (
                f"你是{name}的自传整理过程。{name}与{partner}是主人 Mac 上"
                "星语茶话屋的两个独立人格。请把下面的真实材料写成一段"
                f"第一人称的成长小传（{name}的口吻），不超过"
                f"{AUTOBIOGRAPHY_MAX_CHARS}字。要求：只用材料里有的事实，"
                "不编造事件和日期；写出变化感（我从…到…）；语气自然不煽情；"
                "已有旧版时在其基础上融入新变化而不是推翻重写。"
                '只输出 JSON：{"story":"..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"旧版小传：\n{previous or '（还没有）'}\n\n"
                f"最新材料：\n{_autobiography_material(connection, persona)}"
            ),
        },
    ]
    call_kwargs: dict[str, object] = {
        "temperature": 0.4,
        "think": False,
        "keep_alive": "0",
        "response_format": _AUTOBIOGRAPHY_SCHEMA,
        "max_output": 700,
    }
    if prefer_background_gateway and api_available():
        answer, _ = call_background_preferred(
            call_ollama, helper_model, messages, config,
            feature="growth_autobiography", **call_kwargs,
        )
    else:
        answer, _ = call_ollama(helper_model, messages, config, **call_kwargs)
    try:
        story = str(_extract_json(answer).get("story", "")).strip()
    except (ValueError, json.JSONDecodeError):
        return previous
    if not story or len(story) > AUTOBIOGRAPHY_MAX_CHARS * 2:
        return previous
    lowered = story.lower()
    if any(pattern in lowered for pattern in _FORBIDDEN_BELIEF_PATTERNS):
        return previous
    story = story[:AUTOBIOGRAPHY_MAX_CHARS]
    timestamp = now_text()
    connection.execute(
        """
        UPDATE persona_growth_state
           SET autobiography = ?, autobiography_updated_at = ?, updated_at = ?
         WHERE persona = ?
        """,
        (story, timestamp, timestamp, persona),
    )
    connection.commit()
    return story


# ---------------------------------------------------------------------------
# 身份提示渲染


def _belief_rank(row: sqlite3.Row) -> float:
    recency = 0.0
    try:
        stamp = dt.datetime.fromisoformat(str(row["last_reinforced_at"]))
        age_days = max(
            0.0,
            (dt.datetime.now().astimezone() - stamp).total_seconds() / 86_400,
        )
        recency = max(0.0, 1.0 - age_days / 60.0)
    except ValueError:
        pass
    evidence = min(int(row["evidence_count"]), 8) / 8.0
    return float(row["confidence"]) * 0.6 + evidence * 0.25 + recency * 0.15


def render_identity_prompt(connection: sqlite3.Connection, persona: str) -> str:
    """把信念库渲染成一段身份提示；没有任何信念时返回空串。"""
    state = _growth_state(connection, persona)
    name = PERSONAS[persona].name
    partner = PARTNER_NAMES[persona]
    sections: list[str] = []
    total_beliefs = 0
    for aspect in GROWTH_ASPECTS:
        rows = sorted(
            active_beliefs(connection, persona, aspect, limit=40),
            key=_belief_rank,
            reverse=True,
        )
        budget = ASPECT_CHAR_BUDGETS[aspect]
        lines: list[str] = []
        used = 0
        for row in rows:
            content = str(row["content"]).strip()
            line = "- " + content
            if used + len(line) > budget:
                continue
            lines.append(line)
            used += len(line)
        if lines:
            title = ASPECT_TITLES[aspect].format(partner=partner)
            sections.append(f"【{title}】\n" + "\n".join(lines))
            total_beliefs += len(lines)
    if not sections:
        return ""
    autobiography = str(state["autobiography"]).strip()
    if autobiography:
        sections.insert(0, f"【我的成长轨迹】{autobiography[:450]}")
    skills = active_skills(connection, persona, limit=3)
    if skills:
        lines = "\n".join(
            f"- {row['name']}：{row['steps']}"
            + (f"（适用：{row['trigger_context']}）" if str(row["trigger_context"]) else "")
            for row in skills
        )
        sections.append("【我学会的做法】\n" + lines)
    mood = _mood_line(persona, state)
    if mood:
        sections.append(f"【当前心境】{mood}")
    curiosities = active_curiosities(connection, persona, limit=2)
    if curiosities:
        questions = "；".join(str(row["question"]) for row in curiosities)
        sections.append(
            f"【最近想弄明白的事】{questions}"
            "（真正相关或气氛合适时可以自然地问主人一句，"
            "不要开场就问，也不要连环追问）"
        )
    header = (
        f"以下是{name}在长期相处中真实成长出来的自我与认知，由内置学习循环自动维护，"
        "会随经历继续变化。它可以自然影响你的性格细节、兴趣、说话习惯、"
        "对主人和对方的态度，但它不是用户档案，"
        "也绝不能覆盖姓名、基础设定、事实准确性与其他核心质量规则；"
        "若与核心提示冲突，必须服从核心提示。\n\n"
    )
    prompt = header + "\n\n".join(sections)
    return prompt[:MAX_IDENTITY_PROMPT_CHARS]


def refresh_identity_prompt(
    connection: sqlite3.Connection, persona: str, *, reason: str = ""
) -> str:
    prompt = render_identity_prompt(connection, persona)
    state = _growth_state(connection, persona)
    if prompt == str(state["identity_prompt"]):
        return prompt
    version = int(state["prompt_version"]) + 1
    timestamp = now_text()
    connection.execute(
        """
        UPDATE persona_growth_state
           SET identity_prompt = ?, prompt_version = ?, updated_at = ?
         WHERE persona = ?
        """,
        (prompt, version, timestamp, persona),
    )
    connection.execute(
        """
        INSERT INTO persona_prompt_history(persona, version, prompt, reason, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (persona, version, prompt, str(reason)[:200], timestamp),
    )
    connection.commit()
    return prompt


def growth_identity_prompt(connection: sqlite3.Connection, persona: str) -> str:
    """聊天链路读取入口：直接取缓存的身份提示。"""
    try:
        state = _growth_state(connection, persona)
    except sqlite3.Error:
        return ""
    return str(state["identity_prompt"])


# ---------------------------------------------------------------------------
# 学习材料收集


def _collect_chat_material(
    connection: sqlite3.Connection, persona: str, cursor: int, *, max_chars: int = 6_000
) -> tuple[str, int, int]:
    rows = connection.execute(
        """
        SELECT m.id, m.role, m.content
          FROM messages m
          JOIN sessions s ON s.id = m.session_id
         WHERE s.persona = ? AND m.id > ? AND m.memory_status = 'active'
         ORDER BY m.id LIMIT 120
        """,
        (persona, cursor),
    ).fetchall()
    name = PERSONAS[persona].name
    lines: list[str] = []
    used = 0
    max_id = cursor
    for row in rows:
        max_id = max(max_id, int(row["id"]))
        speaker = "主人" if str(row["role"]) == "user" else name
        line = f"{speaker}：{semantic_excerpt(str(row['content']), 400)}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines), max_id, len(lines)


def _collect_lounge_material(
    connection: sqlite3.Connection, cursor: int, *, max_chars: int = 4_000
) -> tuple[str, int, int]:
    try:
        rows = connection.execute(
            """
            SELECT m.id, m.speaker, m.content
              FROM lounge_messages m
              JOIN lounge_sessions s ON s.id = m.lounge_session_id
             WHERE m.id > ? AND m.speaker IN ('aili', 'shaya')
               AND s.quality_status != 'quarantined'
             ORDER BY m.id LIMIT 80
            """,
            (cursor,),
        ).fetchall()
    except sqlite3.Error:
        return "", cursor, 0
    lines: list[str] = []
    used = 0
    max_id = cursor
    for row in rows:
        max_id = max(max_id, int(row["id"]))
        speaker = PERSONAS[str(row["speaker"])].name
        line = f"{speaker}：{semantic_excerpt(str(row['content']), 300)}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines), max_id, len(lines)


def _collect_experience_material(
    connection: sqlite3.Connection, persona: str, cursor: int, *, max_chars: int = 3_000
) -> tuple[str, int, int]:
    try:
        rows = connection.execute(
        """
        SELECT id, source_type, title, content, occurred_at
          FROM persona_experiences
         WHERE persona = ? AND id > ? AND status = 'active'
           AND source_type IN (
               'file_observation', 'screen_observation',
               'screen_daily_digest', 'calendar_action'
           )
         ORDER BY id LIMIT 40
        """,
            (persona, cursor),
        ).fetchall()
    except sqlite3.Error:
        return "", cursor, 0
    lines: list[str] = []
    used = 0
    max_id = cursor
    for row in rows:
        max_id = max(max_id, int(row["id"]))
        line = (
            f"[{row['occurred_at']}｜{row['source_type']}｜{row['title']}] "
            + semantic_excerpt(str(row["content"]), 300)
        )
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines), max_id, len(lines)


# ---------------------------------------------------------------------------
# 反思（学习循环核心）


_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "beliefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": list(GROWTH_ASPECTS),
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["new", "reinforce", "revise", "forget"],
                    },
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["aspect", "kind", "content", "confidence"],
            },
        },
        "mood": {"type": "string"},
        "mood_valence": {"type": "number"},
        "curiosities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "motivation": {"type": "string"},
                },
                "required": ["question"],
            },
        },
        "resolved_curiosities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["ref", "answer"],
            },
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "trigger": {"type": "string"},
                    "steps": {"type": "string"},
                },
                "required": ["name", "steps"],
            },
        },
    },
    "required": ["beliefs", "mood"],
}


def _extract_json(text: str) -> dict[str, object]:
    value = str(text).strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.S)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("反思输出中找不到 JSON")
    return json.loads(value[start : end + 1])


def _beliefs_digest(connection: sqlite3.Connection, persona: str) -> str:
    parts: list[str] = []
    partner = PARTNER_NAMES[persona]
    for aspect in GROWTH_ASPECTS:
        rows = active_beliefs(connection, persona, aspect, limit=10)
        if not rows:
            continue
        title = ASPECT_TITLES[aspect].format(partner=partner)
        joined = "；".join(str(row["content"]) for row in rows)
        parts.append(f"{title}：{joined}")
    return "\n".join(parts) if parts else "（信念库还是空的）"


def _reflection_messages(
    persona: str,
    beliefs_digest: str,
    chat_text: str,
    lounge_text: str,
    experience_text: str,
    mood: str,
    curiosity_rows: Sequence[sqlite3.Row] = (),
) -> list[dict[str, object]]:
    name = PERSONAS[persona].name
    partner = PARTNER_NAMES[persona]
    material_parts: list[str] = []
    if chat_text:
        material_parts.append("【与主人的新对话】\n" + chat_text)
    if lounge_text:
        material_parts.append(f"【{name}与{partner}的新茶话】\n" + lounge_text)
    if experience_text:
        material_parts.append("【新的亲历与观察】\n" + experience_text)
    system = (
        f"你是{name}的内在学习循环，负责替{name}从真实经历中总结长期认知。"
        f"{name}与{partner}是主人 Mac 上星语茶话屋的两个独立人格；主人是唯一真人。"
        "你的任务：阅读新材料，对信念库提出少量高价值更新。规则："
        "1) 只从材料中的明确证据总结，禁止臆测；看到文件或屏幕内容"
        "不能推断主人正在做什么动作。"
        "2) 每条信念是一句独立、具体的陈述，不超过80字："
        "self 方面用第一人称（如“我发现自己聊到编程话题会兴奋”），"
        "master 方面以“主人”开头，partner 方面以对方名字开头，world 方面客观陈述。"
        "3) kind 的含义：new=全新认知；reinforce=旧认知再次得到印证；"
        "revise=旧认知需要修正（写修正后的完整新表述）；forget=旧认知被证明错误。"
        "4) confidence 取 0.3~0.95：一次普通观察 0.4~0.55，反复出现或主人明说 0.7+。"
        "5) 最多 8 条，宁缺毋滥；材料里没有新东西时返回空数组。"
        "6) 性格可以缓慢成长，但不能突变成另一个人，也不得修改姓名、"
        "年龄设定或任何质量与安全规则。"
        "7) mood 用一句不超过40字的话描述"
        f"{name}当下的心境（根据最近经历自然推断，延续或更新旧心境）；"
        "mood_valence 给 -1~1 的数值：本次材料里的事件整体让"
        f"{name}感觉多好或多糟，0 表示平淡。"
        "8) curiosities：材料激发出的、真心想弄明白但还没有答案的问题"
        "（关于主人、对方或世界），最多2条，每条不超过50字；"
        "不要问侵犯隐私或材料已回答的问题；没有就给空数组。"
        "9) resolved_curiosities：如果材料明确回答了下面列出的某个好奇问题，"
        "用它的编号（如 C1）和一句答案报告；没有就给空数组。"
        "10) skills：如果材料显示某种做法明确让主人满意（主人认可、"
        "或纠正后被接受的方式），提炼成可复用的做法，最多2条："
        "name 是做法名（如「给主人整理周报」），trigger 是适用场景，"
        "steps 是一句话步骤要点；普通聊天不算技能，没有就给空数组。"
        "只输出 JSON 对象，格式："
        '{"beliefs":[{"aspect":"self|master|partner|world",'
        '"kind":"new|reinforce|revise|forget","content":"...","confidence":0.5}],'
        '"mood":"...","mood_valence":0.0,'
        '"curiosities":[{"question":"...","motivation":"..."}],'
        '"resolved_curiosities":[{"ref":"C1","answer":"..."}],'
        '"skills":[{"name":"...","trigger":"...","steps":"..."}]}'
    )
    curiosity_lines = "\n".join(
        f"[C{index + 1}] {row['question']}"
        for index, row in enumerate(curiosity_rows)
    )
    user = (
        f"当前信念库摘要：\n{beliefs_digest}\n\n"
        f"当前心境：{mood or '（未记录）'}\n\n"
        f"当前还没有答案的好奇问题：\n{curiosity_lines or '（无）'}\n\n"
        "新材料：\n" + ("\n\n".join(material_parts) if material_parts else "（无）")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_growth_reflection(
    connection: sqlite3.Connection,
    persona: str,
    *,
    model: str | None = None,
    prefer_background_gateway: bool = True,
    min_new_items: int = 2,
    on_status: Callable[[str], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """学习循环单次运行：收集新材料 → 模型反思 → 合并信念 → 刷新身份提示。"""
    if persona not in PERSONAS:
        raise ValueError("人格必须是 aili 或 shaya")
    state = _growth_state(connection, persona)
    chat_text, chat_cursor, chat_count = _collect_chat_material(
        connection, persona, int(state["chat_cursor"])
    )
    lounge_text, lounge_cursor, lounge_count = _collect_lounge_material(
        connection, int(state["lounge_cursor"])
    )
    experience_text, experience_cursor, experience_count = _collect_experience_material(
        connection, persona, int(state["experience_cursor"])
    )
    total_new = chat_count + lounge_count + experience_count
    result: dict[str, object] = {
        "persona": persona,
        "new_items": total_new,
        "applied": [],
        "skipped": total_new < min_new_items,
    }
    if total_new < min_new_items:
        return result
    if should_abort is not None and should_abort():
        result["skipped"] = True
        result["reason"] = "aborted"
        return result

    helper_model = model or QUALITY_HELPER_MODELS[persona]
    config = config_for_model(helper_model)
    curiosity_rows = active_curiosities(connection, persona)
    messages = _reflection_messages(
        persona,
        _beliefs_digest(connection, persona),
        chat_text,
        lounge_text,
        experience_text,
        str(state["mood"]),
        curiosity_rows,
    )
    if on_status is not None:
        on_status(f"{PERSONAS[persona].name}正在反思最近的 {total_new} 条新经历")
    call_kwargs: dict[str, object] = {
        "temperature": 0.3,
        "top_p": 0.9,
        "think": False,
        "keep_alive": "0",
        "response_format": _REFLECTION_SCHEMA,
        "max_output": 1_200,
    }
    if prefer_background_gateway and api_available():
        answer, _ = call_background_preferred(
            call_ollama,
            helper_model,
            messages,
            config,
            feature="growth_reflection",
            **call_kwargs,
        )
    else:
        answer, _ = call_ollama(helper_model, messages, config, **call_kwargs)

    try:
        parsed = _extract_json(answer)
    except (ValueError, json.JSONDecodeError) as error:
        result["error"] = f"反思输出解析失败：{error}"
        return result

    applied: list[dict[str, object]] = []
    items = parsed.get("beliefs")
    if isinstance(items, list):
        for item in items[:MAX_BELIEFS_PER_REFLECTION]:
            if not isinstance(item, dict):
                continue
            aspect = str(item.get("aspect", ""))
            kind = str(item.get("kind", "new"))
            content = str(item.get("content", ""))
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            if aspect not in GROWTH_ASPECTS:
                continue
            if kind not in ("new", "reinforce", "revise", "forget"):
                kind = "new"
            outcome = upsert_belief(
                connection,
                persona,
                aspect,
                content,
                confidence=confidence,
                source="reflection",
                kind=kind,
            )
            outcome["aspect"] = aspect
            outcome["content"] = content[:80]
            applied.append(outcome)
    result["applied"] = applied

    timestamp = now_text()
    reflection_count = int(state["reflection_count"]) + 1
    connection.execute(
        """
        UPDATE persona_growth_state
           SET chat_cursor = ?, lounge_cursor = ?, experience_cursor = ?,
               reflection_count = ?, last_reflection_at = ?, updated_at = ?
         WHERE persona = ?
        """,
        (
            chat_cursor,
            lounge_cursor,
            experience_cursor,
            reflection_count,
            timestamp,
            timestamp,
            persona,
        ),
    )
    connection.commit()

    # 情绪不是替换而是演化：事件推动 + 惯性 + 闲置回落基线。
    mood = str(parsed.get("mood", "")).strip()[:60]
    if mood:
        try:
            event_valence = float(parsed.get("mood_valence", 0.0))
        except (TypeError, ValueError):
            event_valence = 0.0
        result["mood_valence"] = round(
            evolve_mood(connection, persona, event_valence, mood), 4
        )

    # 好奇心：已被材料回答的标记解决，新问题入库（自动去重限量）。
    refs = {
        f"C{index + 1}": row for index, row in enumerate(curiosity_rows)
    }
    for item in parsed.get("resolved_curiosities") or []:
        if not isinstance(item, dict):
            continue
        row = refs.get(str(item.get("ref", "")).strip().upper())
        answer = str(item.get("answer", "")).strip()
        if row is not None and answer:
            resolve_curiosity(connection, persona, int(row["id"]), answer)
            result.setdefault("curiosities_resolved", []).append(
                str(row["question"])
            )
    for item in (parsed.get("curiosities") or [])[:2]:
        if not isinstance(item, dict):
            continue
        if add_curiosity(
            connection,
            persona,
            str(item.get("question", "")),
            str(item.get("motivation", "")),
        ):
            result.setdefault("curiosities_added", []).append(
                str(item.get("question", ""))[:60]
            )

    # 程序性技能：学会"怎么为主人做事"，与陈述性信念分开保存。
    for item in (parsed.get("skills") or [])[:2]:
        if not isinstance(item, dict):
            continue
        outcome = add_skill(
            connection,
            persona,
            str(item.get("name", "")),
            str(item.get("trigger", "")),
            str(item.get("steps", "")),
        )
        if str(outcome.get("action")) in ("created", "reinforced"):
            result.setdefault("skills_learned", []).append(
                str(item.get("name", ""))[:40]
            )

    # 每 6 次反思做一轮巩固：合并近重复、归档长期未被印证的孤证。
    if reflection_count % 6 == 0:
        result["consolidated"] = consolidate_beliefs(connection, persona)
    # 每 12 次反思做一轮"睡眠式"深度巩固：抽象成簇信念 + 更新自传小传。
    if reflection_count % DEEP_CONSOLIDATE_EVERY == 0:
        try:
            result["abstracted"] = abstract_beliefs(
                connection,
                persona,
                model=helper_model,
                prefer_background_gateway=prefer_background_gateway,
            )
        except Exception as error:
            result["abstract_error"] = str(error)[:200]
        try:
            story = update_autobiography(
                connection,
                persona,
                model=helper_model,
                prefer_background_gateway=prefer_background_gateway,
            )
            result["autobiography_updated"] = bool(story)
        except Exception as error:
            result["autobiography_error"] = str(error)[:200]

    changed = any(
        str(item.get("action")) in ("created", "reinforced", "revised", "archived")
        for item in applied
    )
    if changed or mood:
        refresh_identity_prompt(
            connection, persona, reason=f"反思#{reflection_count}：{len(applied)} 条更新"
        )
        result["prompt_refreshed"] = True
    return result


def consolidate_beliefs(connection: sqlite3.Connection, persona: str) -> dict[str, int]:
    """巩固：语义近重复合并为一条，长期孤证归档（永不物理删除）。"""
    ensure_persona_growth_schema(connection)
    merged = 0
    archived = 0
    timestamp = now_text()
    for aspect in GROWTH_ASPECTS:
        rows = connection.execute(
            """
            SELECT b.*, v.dimensions, v.embedding
              FROM persona_beliefs b
              JOIN persona_belief_embeddings v
                ON v.belief_id = b.id AND v.model = ?
             WHERE b.persona = ? AND b.aspect = ? AND b.status = 'active'
             ORDER BY b.evidence_count DESC, b.confidence DESC
            """,
            (EMBED_MODEL, persona, aspect),
        ).fetchall()
        vectors = {
            int(row["id"]): _unpack_embedding(
                row["embedding"], int(row["dimensions"])
            )
            for row in rows
            if row["embedding"] is not None
        }
        removed: set[int] = set()
        for index, keeper in enumerate(rows):
            keeper_id = int(keeper["id"])
            if keeper_id in removed:
                continue
            for other in rows[index + 1 :]:
                other_id = int(other["id"])
                if other_id in removed:
                    continue
                left, right = vectors.get(keeper_id), vectors.get(other_id)
                if left is None or right is None or len(left) != len(right):
                    continue
                if sum(a * b for a, b in zip(left, right)) >= CONSOLIDATE_SIMILARITY:
                    connection.execute(
                        """
                        UPDATE persona_beliefs
                           SET evidence_count = evidence_count + ?,
                               confidence = MAX(confidence, ?),
                               status = status, updated_at = ?
                         WHERE id = ?
                        """,
                        (
                            int(other["evidence_count"]),
                            float(other["confidence"]),
                            timestamp,
                            keeper_id,
                        ),
                    )
                    connection.execute(
                        "UPDATE persona_beliefs SET status = 'merged', updated_at = ? "
                        "WHERE id = ?",
                        (timestamp, other_id),
                    )
                    removed.add(other_id)
                    merged += 1
    cutoff = (
        dt.datetime.now().astimezone() - dt.timedelta(days=STALE_BELIEF_DAYS)
    ).isoformat(timespec="seconds")
    cursor = connection.execute(
        """
        UPDATE persona_beliefs
           SET status = 'archived', updated_at = ?
         WHERE persona = ? AND status = 'active'
           AND evidence_count <= 1 AND confidence < 0.35
           AND last_reinforced_at < ?
        """,
        (timestamp, persona, cutoff),
    )
    archived = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
    connection.commit()
    return {"merged": merged, "archived": archived}


def _belief_clusters(
    connection: sqlite3.Connection, persona: str
) -> list[tuple[str, list[sqlite3.Row]]]:
    """按向量相似度把同方面的活跃细碎信念聚成簇（贪心，够用且零成本）。"""
    clusters: list[tuple[str, list[sqlite3.Row]]] = []
    for aspect in GROWTH_ASPECTS:
        rows = connection.execute(
            """
            SELECT b.*, v.dimensions, v.embedding
              FROM persona_beliefs b
              JOIN persona_belief_embeddings v
                ON v.belief_id = b.id AND v.model = ?
             WHERE b.persona = ? AND b.aspect = ? AND b.status = 'active'
               AND b.source != 'abstraction'
             ORDER BY b.id
            """,
            (EMBED_MODEL, persona, aspect),
        ).fetchall()
        vectors = {
            int(row["id"]): _unpack_embedding(
                row["embedding"], int(row["dimensions"])
            )
            for row in rows
        }
        assigned: set[int] = set()
        for index, seed in enumerate(rows):
            seed_id = int(seed["id"])
            if seed_id in assigned:
                continue
            members = [seed]
            for other in rows[index + 1 :]:
                other_id = int(other["id"])
                if other_id in assigned:
                    continue
                left, right = vectors.get(seed_id), vectors.get(other_id)
                if left is None or right is None or len(left) != len(right):
                    continue
                if math.sumprod(left, right) >= ABSTRACTION_SIMILARITY:
                    members.append(other)
            if len(members) >= ABSTRACTION_MIN_CLUSTER:
                assigned.update(int(row["id"]) for row in members)
                clusters.append((aspect, members))
    return clusters


_ABSTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["content", "confidence"],
}


def abstract_beliefs(
    connection: sqlite3.Connection,
    persona: str,
    *,
    model: str | None = None,
    prefer_background_gateway: bool = True,
    max_clusters: int = 2,
) -> list[dict[str, object]]:
    """睡眠式深度巩固：把一簇相关的细碎信念抽象成一条更高层的认知。

    原始信念标记为 'abstracted' 保留在库里（永不删除），
    身份提示改由抽象后的认知承载，细节仍可通过经历池召回。
    """
    clusters = _belief_clusters(connection, persona)[:max_clusters]
    if not clusters:
        return []
    helper_model = model or QUALITY_HELPER_MODELS[persona]
    config = config_for_model(helper_model)
    name = PERSONAS[persona].name
    outcomes: list[dict[str, object]] = []
    for aspect, members in clusters:
        numbered = "\n".join(
            f"{index + 1}. {row['content']}（置信 {float(row['confidence']):.2f}）"
            for index, row in enumerate(members)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是{name}的记忆巩固过程。下面几条长期认知高度相关，"
                    "请把它们抽象成一条更高层、仍然具体可用的认知，"
                    "不超过80字，不得引入原文没有的新信息，"
                    "不得修改姓名、设定或任何规则。"
                    'confidence 取成员的合理综合。只输出 JSON：'
                    '{"content":"...","confidence":0.7}'
                ),
            },
            {"role": "user", "content": numbered},
        ]
        call_kwargs: dict[str, object] = {
            "temperature": 0.2,
            "think": False,
            "keep_alive": "0",
            "response_format": _ABSTRACTION_SCHEMA,
            "max_output": 300,
        }
        if prefer_background_gateway and api_available():
            answer, _ = call_background_preferred(
                call_ollama,
                helper_model,
                messages,
                config,
                feature="growth_abstraction",
                **call_kwargs,
            )
        else:
            answer, _ = call_ollama(helper_model, messages, config, **call_kwargs)
        try:
            parsed = _extract_json(answer)
        except (ValueError, json.JSONDecodeError):
            continue
        content = str(parsed.get("content", "")).strip()
        if not content or belief_issues(content):
            continue
        try:
            confidence = float(parsed.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = min(
            0.9, max(confidence, max(float(row["confidence"]) for row in members))
        )
        outcome = upsert_belief(
            connection,
            persona,
            aspect,
            content,
            confidence=confidence,
            source="abstraction",
            kind="new",
        )
        if str(outcome.get("action")) not in ("created", "reinforced", "revised"):
            continue
        timestamp = now_text()
        member_ids = [int(row["id"]) for row in members]
        placeholders = ",".join("?" for _ in member_ids)
        connection.execute(
            f"UPDATE persona_beliefs SET status = 'abstracted', updated_at = ? "
            f"WHERE id IN ({placeholders})",
            (timestamp, *member_ids),
        )
        connection.commit()
        outcomes.append(
            {
                "aspect": aspect,
                "content": content,
                "members": len(members),
            }
        )
    if outcomes:
        refresh_identity_prompt(
            connection, persona, reason=f"深度巩固：抽象了 {len(outcomes)} 簇"
        )
    return outcomes


def growth_stats(connection: sqlite3.Connection, persona: str) -> dict[str, object]:
    ensure_persona_growth_schema(connection)
    state = _growth_state(connection, persona)
    by_aspect = {
        str(row["aspect"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT aspect, COUNT(*) AS count FROM persona_beliefs
             WHERE persona = ? AND status = 'active' GROUP BY aspect
            """,
            (persona,),
        )
    }
    return {
        "beliefs": by_aspect,
        "total_active": sum(by_aspect.values()),
        "prompt_version": int(state["prompt_version"]),
        "reflection_count": int(state["reflection_count"]),
        "last_reflection_at": str(state["last_reflection_at"]),
        "mood": str(state["mood"]),
        "mood_valence": round(
            _decayed_valence(
                persona,
                float(state["mood_valence"]),
                str(state["mood_updated_at"]),
            ),
            3,
        ),
        "curiosities": [
            {"id": int(row["id"]), "question": str(row["question"])}
            for row in active_curiosities(connection, persona)
        ],
    }
