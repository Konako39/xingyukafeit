#!/usr/bin/env python3
"""记忆向量引擎：fp16 存储 + 进程内矩阵缓存 + numpy 加速批量打分。

设计目标（无限记忆下的成本控制，质量不降）：
  - 磁盘：新向量一律 fp16（体积减半）；旧 fp32 透明兼容并可后台迁移。
    归一化向量的 fp16 点积误差 ~1e-3，远小于所有检索门槛间距。
  - 速度：优先 numpy 矩阵乘（10 万条 1024 维 ≈ 12ms）；没有 numpy 时
    回退 C 实现的 math.sumprod（同规模 ≈ 1.2s），绝不再用纯 Python 逐项乘。
  - 缓存：每张向量表按人格缓存一份内存矩阵，用 (行数, 最大ID, 最新时间)
    校验有效性，追加增量加载，其余情况整表重建；打分结果以 {id: 语义分}
    返回，行级过滤（status/persona 等）仍由上层 SQL 决定，缓存脏行无害。
"""

from __future__ import annotations

import array
import math
import sqlite3
import struct
import threading

try:  # numpy 是可选加速层，Homebrew: brew install numpy
    import numpy as _np
except ImportError:  # pragma: no cover - 环境差异
    _np = None

HAVE_NUMPY = _np is not None

# 超过这个行数就不整表驻留内存，改为分块流式打分（仍走 numpy/sumprod）。
MAX_CACHED_ROWS = 300_000
_CHUNK_ROWS = 8_192

_CACHE_LOCK = threading.Lock()
_CACHES: dict[str, "_CacheEntry"] = {}


def pack_embedding(vector) -> bytes:
    """新向量统一按 fp16 落盘。"""
    values = [float(value) for value in vector]
    return struct.pack(f"{len(values)}e", *values)


def unpack_embedding(blob: bytes, dimensions: int) -> list[float]:
    """按声明维度识别 fp32（4B/维，历史数据）或 fp16（2B/维，新数据）。"""
    size = len(blob)
    if size == 4 * dimensions:
        vector = array.array("f")
        vector.frombytes(blob)
        return list(vector)
    if size == 2 * dimensions:
        return list(struct.unpack(f"{dimensions}e", blob))
    raise ValueError(f"向量长度 {size} 与维度 {dimensions} 不匹配")


def dot(left, right) -> float:
    return math.sumprod(left, right)


class _CacheEntry:
    __slots__ = ("validity", "ids", "matrix", "vectors", "dimensions")

    def __init__(self, dimensions: int) -> None:
        self.validity: tuple = ()
        self.ids: list[int] = []
        self.matrix = None  # numpy fp32 矩阵（有 numpy 时）
        self.vectors: list[array.array] = []  # 回退：array('f') 列表
        self.dimensions = dimensions


def _decode_to_float32(blob: bytes, dimensions: int):
    if _np is not None:
        if len(blob) == 2 * dimensions:
            return _np.frombuffer(blob, dtype=_np.float16).astype(_np.float32)
        return _np.frombuffer(blob, dtype=_np.float32).copy()
    vector = array.array("f")
    if len(blob) == 2 * dimensions:
        vector.fromlist(list(struct.unpack(f"{dimensions}e", blob)))
    else:
        vector.frombytes(blob)
    return vector


def _validity(
    connection: sqlite3.Connection, table: str, id_column: str,
    where_sql: str, params: tuple,
) -> tuple:
    row = connection.execute(
        f"SELECT COUNT(*), COALESCE(MAX({id_column}), 0), COALESCE(MAX(created_at), '')"
        f" FROM {table} WHERE {where_sql}",
        params,
    ).fetchone()
    return (int(row[0]), int(row[1]), str(row[2]))


def _load_rows(
    connection: sqlite3.Connection, table: str, id_column: str,
    where_sql: str, params: tuple, after_id: int, dimensions: int,
) -> tuple[list[int], list]:
    rows = connection.execute(
        f"SELECT {id_column}, dimensions, embedding FROM {table}"
        f" WHERE {where_sql} AND {id_column} > ? ORDER BY {id_column}",
        (*params, after_id),
    ).fetchall()
    ids: list[int] = []
    vectors = []
    for row in rows:
        if int(row[1]) != dimensions:
            continue
        ids.append(int(row[0]))
        vectors.append(_decode_to_float32(bytes(row[2]), dimensions))
    return ids, vectors


def _scores_from_entry(entry: _CacheEntry, query_vector) -> dict[int, float]:
    if not entry.ids:
        return {}
    if _np is not None:
        query = _np.asarray(list(query_vector), dtype=_np.float32)
        values = entry.matrix @ query
        return dict(zip(entry.ids, values.tolist()))
    return {
        item_id: math.sumprod(query_vector, vector)
        for item_id, vector in zip(entry.ids, entry.vectors)
    }


def _stream_scores(
    connection: sqlite3.Connection, table: str, id_column: str,
    where_sql: str, params: tuple, query_vector,
) -> dict[int, float]:
    """超大表不驻留内存，分块流式打分。"""
    dimensions = len(query_vector)
    result: dict[int, float] = {}
    last_id = 0
    query = (
        _np.asarray(list(query_vector), dtype=_np.float32) if _np is not None else None
    )
    while True:
        rows = connection.execute(
            f"SELECT {id_column}, dimensions, embedding FROM {table}"
            f" WHERE {where_sql} AND {id_column} > ? ORDER BY {id_column} LIMIT ?",
            (*params, last_id, _CHUNK_ROWS),
        ).fetchall()
        if not rows:
            return result
        for row in rows:
            last_id = int(row[0])
            if int(row[1]) != dimensions:
                continue
            vector = _decode_to_float32(bytes(row[2]), dimensions)
            if query is not None:
                result[last_id] = float(vector @ query)
            else:
                result[last_id] = math.sumprod(query_vector, vector)


def semantic_scores(
    connection: sqlite3.Connection,
    cache_name: str,
    table: str,
    id_column: str,
    where_sql: str,
    params: tuple,
    query_vector,
) -> dict[int, float]:
    """返回 {行ID: 与查询向量的点积}；维度不符的行自动跳过。

    上层拿到分数后仍用自己的 SQL 决定哪些行参与结果（status、persona、
    时间窗等过滤都在上层），因此缓存里多出的陈旧 ID 不会造成脏读。
    """
    dimensions = len(query_vector)
    validity = _validity(connection, table, id_column, where_sql, params)
    if validity[0] > MAX_CACHED_ROWS:
        return _stream_scores(
            connection, table, id_column, where_sql, params, query_vector
        )
    try:
        # 缓存键必须绑定具体数据库文件，测试临时库与正式库绝不共享缓存。
        database_file = str(
            connection.execute("PRAGMA database_list").fetchone()[2]
        )
    except (sqlite3.Error, TypeError, IndexError):
        database_file = "unknown"
    cache_name = f"{database_file}|{cache_name}"
    with _CACHE_LOCK:
        entry = _CACHES.get(cache_name)
        if entry is None or entry.dimensions != dimensions:
            entry = _CacheEntry(dimensions)
            _CACHES[cache_name] = entry
        if entry.validity != validity:
            cached_count = len(entry.ids)
            cached_max = entry.ids[-1] if entry.ids else 0
            incremental = (
                validity[0] > cached_count and validity[1] > cached_max
            )
            new_ids, new_vectors = _load_rows(
                connection, table, id_column, where_sql, params,
                cached_max if incremental else 0, dimensions,
            )
            if incremental and cached_count + len(new_ids) != validity[0]:
                # 除了追加还有删改，放弃增量，整表重建。
                incremental = False
                new_ids, new_vectors = _load_rows(
                    connection, table, id_column, where_sql, params, 0, dimensions
                )
            if incremental:
                entry.ids.extend(new_ids)
                if _np is not None:
                    stacked = (
                        _np.stack(new_vectors)
                        if new_vectors
                        else _np.empty((0, dimensions), dtype=_np.float32)
                    )
                    entry.matrix = (
                        stacked
                        if entry.matrix is None
                        else _np.vstack((entry.matrix, stacked))
                    )
                else:
                    entry.vectors.extend(new_vectors)
            else:
                entry.ids = new_ids
                if _np is not None:
                    entry.matrix = (
                        _np.stack(new_vectors)
                        if new_vectors
                        else _np.empty((0, dimensions), dtype=_np.float32)
                    )
                else:
                    entry.vectors = new_vectors
            entry.validity = validity
        return _scores_from_entry(entry, query_vector)


def invalidate_cache(cache_name: str | None = None) -> None:
    with _CACHE_LOCK:
        if cache_name is None:
            _CACHES.clear()
        else:
            _CACHES.pop(cache_name, None)


def migrate_embeddings_to_fp16(connection: sqlite3.Connection) -> dict[str, int]:
    """把历史 fp32 向量原地转成 fp16，体积减半；可重复运行，幂等。"""
    converted: dict[str, int] = {}
    tables = (
        ("message_embeddings", "message_id"),
        ("persona_experience_embeddings", "experience_id"),
        ("persona_belief_embeddings", "belief_id"),
        ("lounge_embeddings", "id"),
    )
    for table, id_column in tables:
        try:
            rows = connection.execute(
                f"SELECT {id_column} AS row_id, dimensions, embedding FROM {table}"
            ).fetchall()
        except sqlite3.Error:
            continue
        count = 0
        for row in rows:
            dimensions = int(row["dimensions"])
            blob = bytes(row["embedding"])
            if len(blob) != 4 * dimensions:
                continue  # 已经是 fp16 或异常行
            vector = array.array("f")
            vector.frombytes(blob)
            connection.execute(
                f"UPDATE {table} SET embedding = ? WHERE {id_column} = ?",
                (sqlite3.Binary(pack_embedding(vector)), int(row["row_id"])),
            )
            count += 1
        if count:
            connection.commit()
            converted[table] = count
    if converted:
        invalidate_cache()
    return converted
