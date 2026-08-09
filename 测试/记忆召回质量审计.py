#!/usr/bin/env python3
"""在真实数据库上测两套人格池的召回、隔离、更新与坏记忆排除。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "应用" / "后端"))

from api_long_chat import open_database, retrieve_persona_history  # noqa: E402
from lounge_service import ensure_lounge_schema  # noqa: E402
from persona_memory_pool import retrieve_persona_experiences  # noqa: E402


CASES = (
    {"persona": "aili", "query": "我之前告诉过你我叫什么名字？", "expected": {73}},
    {"persona": "aili", "query": "那个额度监控到底监控什么？", "expected": {103, 105}},
    {
        "persona": "aili",
        "query": "我说N1成绩什么时候出？",
        "expected": {173, 175},
        "preferred": 175,
        "stale": 173,
    },
    {"persona": "shaya", "query": "我的N1考试现在是什么情况？", "expected": {157, 159, 161}},
    {"persona": "shaya", "query": "之前问你的何が疲れた是什么意思？", "expected": {147, 151, 153}},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "数据" / "对话记忆.sqlite3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    connection = open_database(args.db)
    ensure_lounge_schema(connection)
    before = int(connection.execute("SELECT COALESCE(MAX(id),0)+1 FROM messages").fetchone()[0])
    results: list[dict[str, object]] = []
    for case in CASES:
        hits = retrieve_persona_history(
            connection,
            str(case["persona"]),
            str(case["query"]),
            before_message_id=before,
            max_items=5,
            max_chars=2_400,
            embedding_keep_alive="5m",
        )
        ids = [int(item["message_id"]) for item in hits]
        expected = set(case["expected"])
        found = expected & set(ids)
        preferred = int(case.get("preferred", 0) or 0)
        stale = int(case.get("stale", 0) or 0)
        temporal_ok = (
            not preferred
            or (
                preferred in ids
                and (stale not in ids or ids.index(preferred) < ids.index(stale))
            )
        )
        results.append(
            {
                **case,
                "expected": sorted(expected),
                "hit_ids": ids,
                "passed": bool(found) and temporal_ok,
                "temporal_resolution_passed": temporal_ok,
                "reciprocal_rank": (
                    1 / min(ids.index(item) + 1 for item in found) if found else 0
                ),
                "hits": hits,
            }
        )
    isolation_checks = []
    for persona, query, forbidden in (
        ("aili", "何が疲れた怎么翻译", {147, 151, 153}),
        ("shaya", "主人叫什么名字", {73}),
    ):
        hits = retrieve_persona_history(
            connection,
            persona,
            query,
            before_message_id=before,
            max_items=8,
            max_chars=2_400,
            embedding_keep_alive="5m",
        )
        ids = {int(item["message_id"]) for item in hits}
        isolation_checks.append(
            {
                "persona": persona,
                "forbidden_ids": sorted(forbidden),
                "hit_ids": sorted(ids),
                "passed": not bool(ids & forbidden),
            }
        )
    experience_checks = []
    for persona in ("aili", "shaya"):
        hits = retrieve_persona_experiences(
            connection,
            persona,
            "桌面右侧的录屏文件是不是在整理",
            max_items=8,
            max_chars=2_400,
            embedding_keep_alive="0",
        )
        experience_checks.append(
            {
                "persona": persona,
                "passed": not hits,
                "active_bad_hits": hits,
                "quarantined_count": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM persona_experiences "
                        "WHERE persona=? AND status='quarantined'",
                        (persona,),
                    ).fetchone()[0]
                ),
            }
        )
    connection.close()
    all_checks = [bool(item["passed"]) for item in results + isolation_checks + experience_checks]
    report = {
        "passed": sum(all_checks),
        "total": len(all_checks),
        "recall_cases": results,
        "persona_isolation": isolation_checks,
        "quarantine_exclusion": experience_checks,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
