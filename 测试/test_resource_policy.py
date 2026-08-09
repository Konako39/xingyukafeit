from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


BACKEND = Path(__file__).resolve().parents[1] / "应用" / "后端"
sys.path.insert(0, str(BACKEND))

import lounge_service  # noqa: E402
import memory_api_server  # noqa: E402


class ResourcePolicyTests(unittest.TestCase):
    def test_foreground_chat_ignores_stale_long_keep_alive(self):
        with mock.patch.object(
            memory_api_server,
            "resource_snapshot",
            return_value={"fullscreen_active": False},
        ):
            parameters = memory_api_server.generation_parameters(
                {"quality_mode": "balanced", "keep_alive": "30m"}
            )
        self.assertEqual(parameters["keep_alive"], "1m")

    def test_fullscreen_chat_releases_immediately(self):
        with mock.patch.object(
            memory_api_server,
            "resource_snapshot",
            return_value={"fullscreen_active": True},
        ):
            parameters = memory_api_server.generation_parameters(
                {"quality_mode": "balanced"}
            )
        self.assertEqual(parameters["keep_alive"], "0")

    def test_ultimate_ignores_local_quality_modes(self):
        with mock.patch.object(
            memory_api_server,
            "resource_snapshot",
            return_value={"fullscreen_active": False},
        ):
            parameters = memory_api_server.generation_parameters(
                {
                    "model": "ultimate:aili",
                    "quality_mode": "deep",
                    "think": True,
                }
            )
        self.assertEqual(parameters["quality_mode"], "ultimate")
        self.assertFalse(parameters["think"])

    def test_background_checks_are_low_frequency(self):
        self.assertGreaterEqual(lounge_service.SCHEDULER_POLL_SECONDS, 180)
        self.assertGreaterEqual(memory_api_server.MEMORY_REFRESH_RETRY_SECONDS, 180)

    def test_llm_cannot_end_lounge_before_five_rounds(self):
        self.assertGreaterEqual(lounge_service.MIN_LOUNGE_COMPLETE_ROUNDS, 5)
        self.assertEqual(
            lounge_service.MIN_LOUNGE_MESSAGES_BEFORE_DECISION,
            lounge_service.MIN_LOUNGE_COMPLETE_ROUNDS * 2,
        )

    def test_conversation_memory_is_paginated_and_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "memory.sqlite3"
            connection = memory_api_server.open_database(database)
            try:
                for index in range(45):
                    session = memory_api_server.create_session(
                        connection,
                        "huihui_ai/qwen3.5-abliterated:9b-16k",
                        title=f"会话 {index}",
                        persona="aili",
                    )
                    memory_api_server.append_message(
                        connection,
                        int(session["id"]),
                        "user",
                        "特殊检索词" if index == 17 else f"消息 {index}",
                    )
            finally:
                connection.close()

            handler = object.__new__(memory_api_server.MemoryAPIHandler)
            handler.server = SimpleNamespace(database_path=str(database))
            first, total = handler.list_conversations(limit=40)
            remainder, _ = handler.list_conversations(limit=40, offset=40)
            matches, matched_total = handler.list_conversations(
                limit=40, query="特殊检索词"
            )

            self.assertEqual(total, 45)
            self.assertEqual(len(first), 40)
            self.assertEqual(len(remainder), 5)
            self.assertEqual(matched_total, 1)
            self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
