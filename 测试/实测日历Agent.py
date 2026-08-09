#!/usr/bin/env python3
"""日历 Agent 真机端到端验证：自然语言 → 真实系统日历 加/查/删。

需要：Ollama 正在运行 + 日历助手已授权
（首次运行会弹系统授权窗，或到 系统设置 > 隐私与安全性 > 日历 中允许）。

流程完全可逆：添加一条「星语茶话屋测试」日程，查询确认，再用自然语言删除。
人格记忆写入临时数据库，不污染正式数据。
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "应用" / "后端"))

from api_long_chat import PERSONAS, config_for_model, open_database  # noqa: E402
import persona_agent  # noqa: E402


def main() -> int:
    print("== 1. 检查日历助手与授权 ==")
    try:
        payload = persona_agent._run_calendar_helper(["calendars"], timeout=90)
    except persona_agent.CalendarToolError as error:
        print(f"失败：{error}")
        print("请先允许日历访问后重试。")
        return 1
    calendars = payload.get("calendars") or []
    print(f"可见日历 {len(calendars)} 个："
          + "、".join(str(item.get('title')) for item in calendars[:8]))

    model = PERSONAS["aili"].models["4b"]
    config = config_for_model(model)
    tomorrow = (dt.datetime.now() + dt.timedelta(days=1))
    date_label = f"{tomorrow.month}月{tomorrow.day}日"

    with tempfile.TemporaryDirectory() as tmp:
        connection = open_database(Path(tmp) / "e2e.sqlite3")

        def run(text: str) -> dict:
            print(f"\n== 主人说：{text}")
            outcome = persona_agent.handle_agent_request(
                connection, "aili", model, config, text,
                on_status=lambda notice: print(f"   [状态] {notice}"),
            )
            if outcome is None:
                print("   （未识别为日程指令）")
                return {}
            print("   工具结果：" + str(outcome.get("tool_context", ""))[:400])
            return outcome

        added = run(f"帮我在{date_label}下午三点加一个「星语茶话屋测试」日程")
        if not added.get("performed"):
            print("\n添加未完成，终止。")
            connection.close()
            return 1
        run(f"{date_label}我有什么安排？")
        deleted = run(f"把{date_label}的星语茶话屋测试日程删掉")
        connection.close()
        if not deleted.get("performed"):
            print("\n注意：测试日程可能没删掉，请到日历里手动清理「星语茶话屋测试」。")
            return 1
    print("\n全部通过：自然语言加日程、查日程、删日程都已在真实日历上验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
