#!/usr/bin/env python3
"""双人格统一经历池。

普通聊天原文继续保存在 messages/message_embeddings；本模块保存茶话、文件观察、
屏幕观察等非会话经历。检索时由上层把两类结果合并成同一个人格记忆池。
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections.abc import Sequence

from api_long_chat import (
    EMBED_MODEL,
    PERSONAS,
    _pack_embedding,
    _unpack_embedding,
    call_embeddings,
    lexical_memory_score,
    memory_text_overlap,
    now_text,
    semantic_excerpt,
)
from memory_vector_ops import semantic_scores


MAX_EXPERIENCE_CHARS = 12_000
MAX_DAILY_SCREEN_DIGEST_CHARS = 6_000
DETAILED_SCREEN_RETENTION_DAYS = 30


def ensure_persona_memory_pool_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS persona_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL CHECK(persona IN ('aili', 'shaya')),
            source_type TEXT NOT NULL,
            source_key TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_recalled_at TEXT NOT NULL DEFAULT '',
            recall_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            UNIQUE(persona, source_type, source_key)
        );

        CREATE TABLE IF NOT EXISTS persona_experience_embeddings (
            experience_id INTEGER NOT NULL
                REFERENCES persona_experiences(id) ON DELETE CASCADE,
            persona TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(experience_id, model)
        );

        CREATE INDEX IF NOT EXISTS idx_persona_experiences_recent
            ON persona_experiences(persona, occurred_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_persona_experiences_source
            ON persona_experiences(persona, source_type, source_key);
        CREATE INDEX IF NOT EXISTS idx_persona_experience_embeddings
            ON persona_experience_embeddings(persona, model, experience_id);
        """
    )
    experience_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(persona_experiences)")
    }
    if "status" not in experience_columns:
        connection.execute(
            "ALTER TABLE persona_experiences "
            "ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
        )
    connection.commit()


def _metadata_text(metadata: dict[str, object] | None) -> str:
    return json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))


def _linked_lounge_is_quarantined(
    connection: sqlite3.Connection, metadata: dict[str, object] | None
) -> bool:
    """隔离过的茶话永远不能被迁移/回填重新激活。"""
    session_id = int((metadata or {}).get("lounge_session_id") or 0)
    if session_id <= 0:
        return False
    try:
        row = connection.execute(
            "SELECT quality_status FROM lounge_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return bool(row and str(row["quality_status"]) == "quarantined")


def add_persona_experience(
    connection: sqlite3.Connection,
    persona: str,
    source_type: str,
    source_key: str | int,
    title: str,
    content: str,
    *,
    occurred_at: str | None = None,
    importance: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> int:
    """写入或更新一条人格亲历；内容变化时自动让旧向量失效。"""
    if persona not in PERSONAS:
        raise ValueError("人格必须是 aili 或 shaya")
    value = str(content).replace("\x00", "").strip()[:MAX_EXPERIENCE_CHARS]
    if not value:
        raise ValueError("经历内容不能为空")
    ensure_persona_memory_pool_schema(connection)
    timestamp = now_text()
    occurred = occurred_at or timestamp
    existing = connection.execute(
        """
        SELECT id, content, title, occurred_at, importance, metadata, status
          FROM persona_experiences
         WHERE persona = ? AND source_type = ? AND source_key = ?
        """,
        (persona, str(source_type), str(source_key)),
    ).fetchone()
    metadata_text = _metadata_text(metadata)
    desired_status = (
        "quarantined"
        if _linked_lounge_is_quarantined(connection, metadata)
        else "active"
    )
    bounded_importance = max(0.0, min(float(importance), 1.0))
    if existing is None:
        cursor = connection.execute(
            """
            INSERT INTO persona_experiences(
                persona, source_type, source_key, title, content, occurred_at,
                importance, metadata, created_at, updated_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                persona,
                str(source_type)[:80],
                str(source_key)[:240],
                str(title)[:240],
                value,
                occurred,
                bounded_importance,
                metadata_text,
                timestamp,
                timestamp,
                desired_status,
            ),
        )
        experience_id = int(cursor.lastrowid)
    else:
        experience_id = int(existing["id"])
        changed = any(
            (
                str(existing["content"]) != value,
                str(existing["title"]) != str(title)[:240],
                str(existing["occurred_at"]) != occurred,
                float(existing["importance"]) != bounded_importance,
                str(existing["metadata"]) != metadata_text,
                str(existing["status"]) != desired_status,
            )
        )
        if changed:
            connection.execute(
                """
                UPDATE persona_experiences
                   SET title = ?, content = ?, occurred_at = ?, importance = ?,
                       metadata = ?, updated_at = ?, status = ?
                 WHERE id = ?
                """,
                (
                    str(title)[:240],
                    value,
                    occurred,
                    bounded_importance,
                    metadata_text,
                    timestamp,
                    desired_status,
                    experience_id,
                ),
            )
            connection.execute(
                "DELETE FROM persona_experience_embeddings WHERE experience_id = ?",
                (experience_id,),
            )
    connection.commit()
    return experience_id


def index_persona_experiences(
    connection: sqlite3.Connection,
    persona: str,
    *,
    embedding_keep_alive: str = "1m",
) -> int:
    """只为尚未索引或内容刚更新的经历建立向量。"""
    ensure_persona_memory_pool_schema(connection)
    rows = connection.execute(
        """
        SELECT e.id, e.source_type, e.title, e.content, e.occurred_at
          FROM persona_experiences e
          LEFT JOIN persona_experience_embeddings v
            ON v.experience_id = e.id AND v.model = ?
         WHERE e.persona = ? AND e.status = 'active' AND v.experience_id IS NULL
         ORDER BY e.id
        """,
        (EMBED_MODEL, persona),
    ).fetchall()
    indexed = 0
    for start in range(0, len(rows), 24):
        batch = rows[start : start + 24]
        documents = [
            (
                f"这是{PERSONAS[persona].name}自己的经历记忆，用于以后按主题检索。\n"
                f"时间：{row['occurred_at']}\n来源：{row['source_type']}\n"
                f"标题：{row['title']}\n内容：{semantic_excerpt(str(row['content']))}"
            )
            for row in batch
        ]
        vectors = call_embeddings(documents, keep_alive=embedding_keep_alive)
        timestamp = now_text()
        for row, vector in zip(batch, vectors):
            connection.execute(
                """
                INSERT OR REPLACE INTO persona_experience_embeddings(
                    experience_id, persona, model, dimensions, embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["id"]),
                    persona,
                    EMBED_MODEL,
                    len(vector),
                    sqlite3.Binary(_pack_embedding(vector)),
                    timestamp,
                ),
            )
            indexed += 1
        connection.commit()
    return indexed


def retrieve_persona_experiences(
    connection: sqlite3.Connection,
    persona: str,
    query: str,
    *,
    max_items: int = 5,
    max_chars: int = 1_500,
    min_score: float = 0.48,
    embedding_keep_alive: str = "1m",
    mood_hint: str = "",
) -> list[dict[str, object]]:
    """从指定人格自己的非会话经历里做语义召回，绝不跨人格。

    mood_hint 是心境一致性回忆：情绪明显时情景经历的检索方向轻微受
    当前心境影响（更接近人的回忆方式）；只作用于经历池，不影响事实检索。
    """
    if not query.strip() or max_chars <= 0:
        return []
    ensure_persona_memory_pool_schema(connection)
    index_persona_experiences(
        connection, persona, embedding_keep_alive=embedding_keep_alive
    )
    mood_line = f"\n{mood_hint.strip()}" if mood_hint.strip() else ""
    query_vector = call_embeddings(
        [
            "检索任务：寻找有助于当前判断或聊天的亲历，包括与主人交流、"
            "人格间对话、文件观察和屏幕观察。只按主题相关性检索。"
            + mood_line
            + "\n当前主题："
            + semantic_excerpt(query)
        ],
        keep_alive=embedding_keep_alive,
    )[0]
    semantic_by_id = semantic_scores(
        connection,
        f"experiences:{persona}",
        "persona_experience_embeddings",
        "experience_id",
        "persona = ? AND model = ?",
        (persona, EMBED_MODEL),
        query_vector,
    )
    rows = connection.execute(
        """
        SELECT e.*
          FROM persona_experience_embeddings v
          JOIN persona_experiences e ON e.id = v.experience_id
         WHERE v.persona = ? AND v.model = ? AND e.status = 'active'
         ORDER BY e.occurred_at DESC, e.id DESC
        """,
        (persona, EMBED_MODEL),
    ).fetchall()
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        semantic = semantic_by_id.get(int(row["id"]))
        if semantic is None:
            continue
        lexical = lexical_memory_score(query, str(row["content"]))
        # 重要经历略微加权，但相关性仍是第一条件。
        adjusted = (
            semantic
            + min(0.10, lexical * 0.10)
            + min(0.035, float(row["importance"]) * 0.035)
        )
        if semantic >= min_score or (
            semantic >= min_score - 0.06 and lexical >= 0.18
        ):
            scored.append((adjusted, row))
    scored.sort(
        key=lambda item: (item[0], str(item[1]["occurred_at"]), int(item[1]["id"])),
        reverse=True,
    )
    result: list[dict[str, object]] = []
    used = 0
    selected_texts: list[str] = []
    selected_groups: set[str] = set()
    for score, row in scored:
        remaining = max_chars - used
        if remaining < 100:
            break
        content = str(row["content"]).strip()
        source_type = str(row["source_type"])
        source_key = str(row["source_key"])
        group = (
            "lounge:" + source_key.split(":", 1)[0]
            if source_type.startswith("lounge_")
            else source_type + ":" + source_key
        )
        if group in selected_groups:
            continue
        if any(memory_text_overlap(content, old) >= 0.72 for old in selected_texts):
            continue
        clipped = content[: min(900, remaining)]
        result.append(
            {
                "experience_id": int(row["id"]),
                "source_type": row["source_type"],
                "source_key": row["source_key"],
                "title": row["title"],
                "content": clipped,
                "score": round(score, 4),
                "lexical_score": round(lexical_memory_score(query, content), 4),
                "occurred_at": row["occurred_at"],
                "importance": float(row["importance"]),
            }
        )
        used += len(clipped)
        selected_texts.append(content)
        selected_groups.add(group)
        if len(result) >= max_items:
            break
    if result:
        timestamp = now_text()
        ids = [int(item["experience_id"]) for item in result]
        placeholders = ",".join("?" for _ in ids)
        connection.execute(
            f"""
            UPDATE persona_experiences
               SET last_recalled_at = ?, recall_count = recall_count + 1
             WHERE id IN ({placeholders})
            """,
            (timestamp, *ids),
        )
        connection.commit()
    return result


def recent_persona_experiences(
    connection: sqlite3.Connection,
    persona: str,
    *,
    limit: int = 6,
    max_chars: int = 1_500,
) -> list[dict[str, object]]:
    ensure_persona_memory_pool_schema(connection)
    rows = connection.execute(
        """
        SELECT * FROM persona_experiences
         WHERE persona = ? AND status = 'active'
         ORDER BY occurred_at DESC, id DESC LIMIT ?
        """,
        (persona, max(1, min(limit, 30))),
    ).fetchall()
    result: list[dict[str, object]] = []
    used = 0
    for row in rows:
        remaining = max_chars - used
        if remaining < 100:
            break
        content = str(row["content"]).strip()[: min(900, remaining)]
        result.append(
            {
                "experience_id": int(row["id"]),
                "source_type": row["source_type"],
                "source_key": row["source_key"],
                "title": row["title"],
                "content": content,
                "occurred_at": row["occurred_at"],
                "importance": float(row["importance"]),
            }
        )
        used += len(content)
    return result


def format_persona_experiences(
    persona: str, items: Sequence[dict[str, object]]
) -> str:
    source_names = {
        "lounge_conversation": "与另一人格的交流",
        "file_observation": "本地文件观察",
        "screen_observation": "屏幕观察",
        "screen_daily_digest": "屏幕观察日记",
    }
    return "\n\n".join(
        f"[{PERSONAS[persona].name}记忆池｜"
        f"{source_names.get(str(item.get('source_type')), str(item.get('source_type')))}｜"
        f"{item.get('occurred_at', '时间未知')}｜{item.get('title', '')}]\n"
        f"{item.get('content', '')}"
        for item in items
    )


def append_screen_daily_digest(
    connection: sqlite3.Connection,
    persona: str,
    observation: str,
    occurred_at: str,
    screen_record_id: int,
) -> int:
    """详细屏幕记录只保留 30 天；按日摘要永久保留，避免长期膨胀。"""
    try:
        stamp = dt.datetime.fromisoformat(occurred_at)
    except ValueError:
        stamp = dt.datetime.now().astimezone()
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    day = stamp.date().isoformat()
    clock = stamp.strftime("%H:%M")
    ensure_persona_memory_pool_schema(connection)
    row = connection.execute(
        """
        SELECT content, status FROM persona_experiences
         WHERE persona = ? AND source_type = 'screen_daily_digest' AND source_key = ?
        """,
        (persona, day),
    ).fetchone()
    entry = f"[{clock}] {observation.strip()}"
    # 已隔离的旧日记可能含有错误推测；新的可靠观察必须从干净日记重新开始。
    combined = (
        (str(row["content"]).strip() + "\n" + entry).strip()
        if row and str(row["status"]) == "active"
        else entry
    )
    if len(combined) > MAX_DAILY_SCREEN_DIGEST_CHARS:
        combined = combined[-MAX_DAILY_SCREEN_DIGEST_CHARS:]
    experience_id = add_persona_experience(
        connection,
        persona,
        "screen_daily_digest",
        day,
        f"{day} 屏幕观察日记",
        combined,
        occurred_at=stamp.replace(hour=23, minute=59, second=59).isoformat(
            timespec="seconds"
        ),
        importance=0.55,
        metadata={"latest_screen_record_id": screen_record_id, "retention": "daily"},
    )
    cutoff = (
        dt.datetime.now().astimezone()
        - dt.timedelta(days=DETAILED_SCREEN_RETENTION_DAYS)
    ).isoformat(timespec="seconds")
    connection.execute(
        """
        DELETE FROM persona_experiences
         WHERE persona = ? AND source_type = 'screen_observation' AND occurred_at < ?
        """,
        (persona, cutoff),
    )
    connection.commit()
    return experience_id


def memory_pool_stats(
    connection: sqlite3.Connection, persona: str
) -> dict[str, object]:
    ensure_persona_memory_pool_schema(connection)
    chat_messages = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM messages m
            JOIN sessions s ON s.id = m.session_id WHERE s.persona = ?
            """,
            (persona,),
        ).fetchone()[0]
    )
    experience_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM persona_experiences "
            "WHERE persona = ? AND status = 'active'",
            (persona,),
        ).fetchone()[0]
    )
    indexed_chat = int(
        connection.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE persona = ?",
            (persona,),
        ).fetchone()[0]
    )
    indexed_experiences = int(
        connection.execute(
            "SELECT COUNT(*) FROM persona_experience_embeddings WHERE persona = ?",
            (persona,),
        ).fetchone()[0]
    )
    sources = {
        str(row["source_type"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT source_type, COUNT(*) AS count FROM persona_experiences
             WHERE persona = ? AND status = 'active' GROUP BY source_type
            """,
            (persona,),
        )
    }
    return {
        "chat_messages": chat_messages,
        "experiences": experience_count,
        "total_items": chat_messages + experience_count,
        "indexed_items": indexed_chat + indexed_experiences,
        "sources": sources,
        "quarantined_items": int(
            connection.execute(
                "SELECT COUNT(*) FROM persona_experiences "
                "WHERE persona = ? AND status = 'quarantined'",
                (persona,),
            ).fetchone()[0]
        ),
    }
