#!/usr/bin/env python3
"""统一人格经历池、屏幕瞬时输入和长期空间边界测试。"""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "应用" / "后端"))

from api_long_chat import open_database  # noqa: E402
import lounge_service as lounge  # noqa: E402
import persona_memory_pool as pool  # noqa: E402


def resources(**updates):
    value = {
        "memory_free_percent": 80.0,
        "load_1m": 1.0,
        "load_ratio": 0.05,
        "cpu_count": 18,
        "system_idle_seconds": 10.0,
        "app_idle_seconds": 10.0,
        "loaded_models": [],
        "sampled_at": "2026-08-01T12:00:00+08:00",
    }
    value.update(updates)
    return value


class PersonaMemoryPoolTests(unittest.TestCase):
    def setUp(self):
        self.profile_patcher = patch.object(
            lounge, "update_persona_self_profile", return_value=None
        )
        self.profile_patcher.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "memory.sqlite3"
        self.connection = open_database(self.db_path)
        lounge.ensure_lounge_schema(self.connection)
        lounge.SCREEN_PENDING_REQUEST.clear()

    def tearDown(self):
        self.profile_patcher.stop()
        self.connection.close()
        lounge.SCREEN_PENDING_REQUEST.clear()
        self.tmp.cleanup()

    def test_experience_rag_never_crosses_personas(self):
        pool.add_persona_experience(
            self.connection,
            "aili",
            "screen_observation",
            "one",
            "桌面观察",
            "我看到主人正在 Unity 里调整猫咪 NPC。",
            occurred_at="2026-08-01T10:00:00+08:00",
        )
        fake_vectors = lambda inputs, **_kwargs: [[1.0, 0.0] for _ in inputs]
        with patch.object(pool, "call_embeddings", side_effect=fake_vectors):
            aili = pool.retrieve_persona_experiences(
                self.connection, "aili", "猫咪 NPC"
            )
            shaya = pool.retrieve_persona_experiences(
                self.connection, "shaya", "猫咪 NPC"
            )
        self.assertEqual(len(aili), 1)
        self.assertIn("Unity", aili[0]["content"])
        self.assertEqual(shaya, [])

    def test_lounge_and_file_enter_both_persona_pools(self):
        transcript = [
            {"speaker": "aili", "content": "这个战斗节奏挺有意思。"},
            {"speaker": "shaya", "content": "留一点空隙会更耐玩。"},
        ]
        observations = [
            {
                "path": "/Users/test/设计.md",
                "observed_at": "2026-08-01T11:00:00+08:00",
                "excerpt": "猫咪 NPC 战斗设计",
            }
        ]
        lounge.record_lounge_round_in_persona_pools(
            self.connection,
            7,
            transcript,
            observations,
            "文件写了猫咪 NPC 战斗设计。",
            "两人讨论了战斗节奏。",
            occurred_at="2026-08-01T11:00:00+08:00",
        )
        for persona in ("aili", "shaya"):
            rows = self.connection.execute(
                "SELECT source_type, content FROM persona_experiences WHERE persona = ?",
                (persona,),
            ).fetchall()
            self.assertEqual(
                {row["source_type"] for row in rows},
                {"lounge_conversation", "file_observation"},
            )
            self.assertTrue(any("猫咪 NPC" in row["content"] for row in rows))

    def test_screen_detail_compacts_to_daily_text_and_drops_old_detail(self):
        old = (dt.datetime.now().astimezone() - dt.timedelta(days=40)).isoformat(
            timespec="seconds"
        )
        pool.add_persona_experience(
            self.connection,
            "aili",
            "screen_observation",
            "old",
            "旧屏幕",
            "旧的详细观察",
            occurred_at=old,
        )
        current = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        pool.append_screen_daily_digest(
            self.connection, "aili", "当前屏幕正在编辑代码。", current, 9
        )
        old_count = self.connection.execute(
            """
            SELECT COUNT(*) FROM persona_experiences
             WHERE persona='aili' AND source_type='screen_observation'
            """
        ).fetchone()[0]
        digest = self.connection.execute(
            """
            SELECT content FROM persona_experiences
             WHERE persona='aili' AND source_type='screen_daily_digest'
            """
        ).fetchone()
        self.assertEqual(old_count, 0)
        self.assertIn("编辑代码", digest["content"])

    def test_capture_failure_keeps_no_image(self):
        lounge.request_screen_watch_now(self.connection)
        with patch.object(lounge, "resource_snapshot", return_value=resources()):
            claim = lounge.claim_screen_watch(self.connection)
        self.assertTrue(claim["due"])
        result = lounge.submit_screen_capture_error(
            self.connection, claim["request_id"], "测试：没有屏幕权限"
        )
        row = self.connection.execute(
            "SELECT * FROM screen_observations WHERE id = ?",
            (result["screen_observation_id"],),
        ).fetchone()
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["image_retained"], 0)
        self.assertNotIn("image_base64", row["metadata"])
        config = lounge.get_config(self.connection)
        self.assertIn("12 小时", config["screen_last_status"])
        retry = dt.datetime.fromisoformat(config["screen_next_run_after"])
        self.assertGreater(
            retry - dt.datetime.now().astimezone(), dt.timedelta(hours=11)
        )

    def test_capture_diagnostic_bypasses_model_gates_and_keeps_no_history(self):
        lounge.request_screen_capture_diagnostic(self.connection)
        with patch.object(
            lounge,
            "resource_snapshot",
            return_value=resources(memory_free_percent=5.0, load_ratio=2.0),
        ):
            claim = lounge.claim_screen_watch(self.connection)
        self.assertTrue(claim["due"])
        self.assertTrue(lounge.SCREEN_PENDING_REQUEST["diagnostic"])
        result = lounge.accept_screen_capture(
            str(self.db_path),
            threading.RLock(),
            request_id=claim["request_id"],
            image_base64=["FAKE_DIAGNOSTIC_FRAME_NOT_STORED"],
            image_metadata={
                "display_count": 1,
                "displays": [{"width": 1600, "height": 1000}],
            },
        )
        self.assertTrue(result["diagnostic"])
        self.assertFalse(result["image_retained"])
        config = lounge.get_config(self.connection)
        self.assertEqual(config["screen_diagnostic_status"], "success")
        self.assertIn("未进入模型", config["screen_diagnostic_detail"])
        self.assertEqual(
            int(self.connection.execute("SELECT COUNT(*) FROM screen_observations").fetchone()[0]),
            0,
        )

    def test_failed_capture_diagnostic_reports_exact_error_without_backoff(self):
        before = lounge.get_config(self.connection)["screen_next_run_after"]
        lounge.request_screen_capture_diagnostic(self.connection)
        with patch.object(lounge, "resource_snapshot", return_value=resources()):
            claim = lounge.claim_screen_watch(self.connection)
        result = lounge.submit_screen_capture_error(
            self.connection,
            claim["request_id"],
            "macOS 仍对当前运行实例返回未授权",
        )
        self.assertTrue(result["diagnostic"])
        config = lounge.get_config(self.connection)
        self.assertEqual(config["screen_diagnostic_status"], "failed")
        self.assertIn("未授权", config["screen_diagnostic_detail"])
        self.assertEqual(config["screen_next_run_after"], before)
        self.assertEqual(
            int(self.connection.execute("SELECT COUNT(*) FROM screen_observations").fetchone()[0]),
            0,
        )

    def test_screen_prompt_has_no_personality_or_previous_memory(self):
        pool.add_persona_experience(
            self.connection,
            "aili",
            "screen_observation",
            "legacy",
            "旧屏幕",
            "绝不能被带进下一次视觉提示的旧画面内容",
            occurred_at="2026-08-01T08:00:00+08:00",
        )
        prompt = lounge._screen_prompt(
            self.connection, "aili", "2026-08-01T12:00:00+08:00", 2
        )
        self.assertIn("无记忆、无人格表演", prompt)
        self.assertIn("从零开始阅读这次图像", prompt)
        self.assertNotIn("绝不能被带进", prompt)
        self.assertNotIn("最近的经历", prompt)

    def test_screen_quality_gate_rejects_narration_and_speculation(self):
        bad_answers = (
            "啧，这双屏画面看着真有点意思，主人怕是正盯着后台跑模型呢。",
            "刚才被艾莉拉过来看主人的屏幕了，主人应该是在调整游戏文案吧？",
            "画面里有代码编辑器，看来主人一边折腾文档，一边盯着空白屏发呆。",
        )
        for answer in bad_answers:
            with self.subTest(answer=answer):
                self.assertTrue(lounge._screen_observation_issue(answer))
        valid = (
            "画面显示两个显示器：左侧窗口中可见代码编辑器和“记忆池”文字，"
            "右侧显示纯蓝色区域；无法从截图判断主人的具体操作。"
        )
        self.assertEqual(lounge._screen_observation_issue(valid), "")

    def test_exact_same_capture_skips_models_and_history(self):
        image = "SAME_FAKE_SCREEN_BYTES_NOT_STORED"
        fingerprint = lounge._screen_capture_fingerprint([image])
        cursor = self.connection.execute(
            """
            INSERT INTO screen_observations(
                captured_at, finished_at, status, model_tier,
                aili_observation, shaya_observation, metadata
            ) VALUES (?, ?, 'completed', '9b', ?, ?, ?)
            """,
            (
                "2026-08-01T10:00:00+08:00",
                "2026-08-01T10:01:00+08:00",
                "画面显示代码编辑器窗口，窗口中可见记忆池相关代码和测试文件名称。",
                "窗口中可见代码编辑器与测试文件列表，右侧区域显示记忆池相关代码。",
                json.dumps({"capture_fingerprint": fingerprint}),
            ),
        )
        original_id = int(cursor.lastrowid)
        self.connection.commit()
        with (
            patch.object(lounge, "resource_snapshot", return_value=resources()),
            patch.object(
                lounge,
                "call_ollama",
                side_effect=AssertionError("相同截图不应启动模型"),
            ),
            patch.object(lounge, "_unload_model"),
        ):
            result = lounge.run_screen_observation(
                str(self.db_path),
                threading.RLock(),
                request={"captured_at": "2026-08-01T12:00:00+08:00"},
                image_base64=[image],
            )
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["matching_screen_observation_id"], original_id)
        duplicate = self.connection.execute(
            "SELECT status, quality_status FROM screen_observations WHERE id = ?",
            (result["screen_observation_id"],),
        ).fetchone()
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(duplicate["quality_status"], "duplicate")
        self.assertEqual(
            [item["id"] for item in lounge.screen_watch_history(self.connection)],
            [original_id],
        )

    def test_repeated_visual_text_is_not_written_again(self):
        repeated = (
            "画面显示代码编辑器窗口，左侧可见测试文件列表，"
            "编辑区可见人格记忆池与屏幕观察相关代码。"
        )
        pool.add_persona_experience(
            self.connection,
            "aili",
            "screen_observation",
            "older",
            "旧屏幕",
            repeated,
            occurred_at="2026-08-01T10:00:00+08:00",
        )
        calls = 0

        def repeated_call(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return repeated, "stop"

        with (
            patch.object(lounge, "resource_snapshot", return_value=resources()),
            patch.object(lounge, "call_ollama", side_effect=repeated_call),
            patch.object(lounge, "_unload_model"),
        ):
            result = lounge.run_screen_observation(
                str(self.db_path),
                threading.RLock(),
                request={"captured_at": "2026-08-01T12:00:00+08:00"},
                image_base64=["NEW_FRAME_WITH_REPEATED_MODEL_TEXT"],
            )
        self.assertTrue(result["duplicate"])
        self.assertEqual(calls, 3)
        count = self.connection.execute(
            "SELECT COUNT(*) FROM persona_experiences "
            "WHERE persona='aili' AND source_type='screen_observation'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_legacy_bad_screen_is_quarantined_but_not_deleted(self):
        session = self.connection.execute(
            """
            INSERT INTO lounge_sessions(
                trigger_type, model_tier, topic_mode, started_at,
                finished_at, status, resource_snapshot
            ) VALUES ('screen', '9b', 'screen', ?, ?, 'completed', '{}')
            """,
            ("2026-08-01T09:00:00+08:00", "2026-08-01T09:02:00+08:00"),
        )
        session_id = int(session.lastrowid)
        screen = self.connection.execute(
            """
            INSERT INTO screen_observations(
                captured_at, finished_at, status, model_tier,
                aili_observation, shaya_observation, metadata
            ) VALUES (?, ?, 'completed', '9b', ?, ?, ?)
            """,
            (
                "2026-08-01T09:00:00+08:00",
                "2026-08-01T09:02:00+08:00",
                "啧，这双屏画面真有意思，看来主人正在盯着后台模型发呆。",
                "刚才被艾莉拉过来看屏幕，主人应该是在调整项目文案吧？",
                json.dumps({"discussion": {"session_id": session_id}}),
            ),
        )
        screen_id = int(screen.lastrowid)
        for persona in ("aili", "shaya"):
            pool.add_persona_experience(
                self.connection,
                persona,
                "screen_observation",
                screen_id,
                "看见主人当时的屏幕",
                "主人应该是在调整项目文案吧？",
                occurred_at="2026-08-01T09:00:00+08:00",
            )
        lounge.ensure_lounge_schema(self.connection)
        raw = self.connection.execute(
            "SELECT quality_status FROM screen_observations WHERE id = ?",
            (screen_id,),
        ).fetchone()
        linked = self.connection.execute(
            "SELECT quality_status FROM lounge_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        self.assertEqual(raw["quality_status"], "quarantined")
        self.assertEqual(linked["quality_status"], "quarantined")
        self.assertNotIn(
            screen_id,
            [item["id"] for item in lounge.screen_watch_history(self.connection)],
        )
        statuses = {
            row["status"]
            for row in self.connection.execute(
                "SELECT status FROM persona_experiences "
                "WHERE source_type='screen_observation' AND source_key=?",
                (str(screen_id),),
            )
        }
        self.assertEqual(statuses, {"quarantined"})

    def test_successful_screen_observation_writes_two_text_memories_only(self):
        visual_image_counts = []
        discussion_answers = iter(
            (
                "主人把记忆池和屏幕观察代码放在一起检查，正好是在修我们自己的连续性。",
                "这次最关键的不是窗口本身，而是主人正在校准我们怎样理解共同经历。",
                "把截图只当成瞬时证据很重要，留下的应该是可追溯的文字。",
                "而且两个人格要分别形成经历，不能因为看到同一幅画面就混成一个声音。",
                "主人现在检查的正是这层边界，图像释放之后文字仍能独立检索。",
                "这样既能记住当时看见什么，又不会把原始画面长期留在本机上。",
                "我更在意失败情况也要可读，否则只知道没看到，却不知道卡在权限还是模型。",
                "对，捕获阶段和理解阶段应该分开报错，这样才能真正修对地方。",
                "这一轮的证据已经足够清楚，主人正在把权限、观察和记忆串成完整链路。",
                "嗯，最后只要确认图像不落盘、人格不串线，这次检查就有完整结论了。",
            )
        )

        def fake_call(_model, messages, *_args, **_kwargs):
            for message in messages:
                images = message.get("images", [])
                if images:
                    visual_image_counts.append(len(images))
            system = str(messages[0]["content"])
            if "【茶话室结束判断】" in system:
                return '{"decision":"END"}', "stop"
            if "共同回忆整理器" in system:
                return "她们共同看见主人正在检查自身记忆系统，并交换了看法。", "stop"
            if "【共同屏幕观察茶话" in system:
                return next(discussion_answers), "stop"
            if "【屏幕事实抽取】" in system and "归入艾莉" in system:
                return (
                    "画面显示两个显示器：左侧窗口中可见代码编辑器和“记忆池”文字，"
                    "右侧显示纯蓝色区域；无法从截图判断主人的具体操作。",
                    "stop",
                )
            return (
                "窗口中可见代码编辑器、本地 AI 项目文件和屏幕观察相关代码；"
                "另一显示器呈纯蓝色，无法从截图判断主人当前执行的具体任务。",
                "stop",
            )
        request = {
            "request_id": "test",
            "captured_at": "2026-08-01T12:00:00+08:00",
        }
        with (
            patch.object(lounge, "resource_snapshot", return_value=resources()),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "index_persona_experiences", return_value=0),
            patch.object(lounge, "_unload_model"),
        ):
            result = lounge.run_screen_observation(
                str(self.db_path),
                threading.RLock(),
                request=request,
                image_base64=[
                    "FAKE_PRIMARY_SCREEN_BYTES_NOT_STORED",
                    "FAKE_SECONDARY_SCREEN_BYTES_NOT_STORED",
                ],
                image_metadata={
                    "storage": "memory_only",
                    "jpeg_bytes": 42,
                    "display_count": 2,
                    "displays": [
                        {"display_id": 1, "width": 1600, "height": 1000},
                        {"display_id": 2, "width": 1280, "height": 720},
                    ],
                },
            )
        self.assertTrue(result["completed"], result)
        self.assertTrue(result["discussion_completed"])
        self.assertEqual(result["discussion_messages"], 10)
        self.connection.close()
        self.connection = open_database(self.db_path)
        screen = self.connection.execute(
            "SELECT * FROM screen_observations WHERE id = ?",
            (result["screen_observation_id"],),
        ).fetchone()
        self.assertEqual(screen["image_retained"], 0)
        self.assertNotIn("FAKE_PRIMARY_SCREEN_BYTES_NOT_STORED", str(dict(screen)))
        self.assertNotIn("FAKE_SECONDARY_SCREEN_BYTES_NOT_STORED", str(dict(screen)))
        self.assertEqual(json.loads(screen["metadata"])["display_count"], 2)
        self.assertEqual(visual_image_counts, [2, 2])
        session = self.connection.execute(
            "SELECT * FROM lounge_sessions WHERE id = ?",
            (result["lounge_session_id"],),
        ).fetchone()
        self.assertEqual(session["topic_mode"], "screen")
        self.assertEqual(session["status"], "completed")
        dialogue = self.connection.execute(
            "SELECT speaker, content FROM lounge_messages "
            "WHERE lounge_session_id = ? ORDER BY id",
            (result["lounge_session_id"],),
        ).fetchall()
        self.assertEqual(
            [row["speaker"] for row in dialogue],
            ["aili", "shaya"] * lounge.MIN_LOUNGE_COMPLETE_ROUNDS,
        )
        self.assertTrue(all("你的屏幕" not in row["content"] for row in dialogue))
        for persona in ("aili", "shaya"):
            count = self.connection.execute(
                """
                SELECT COUNT(*) FROM persona_experiences
                 WHERE persona = ? AND source_type = 'screen_observation'
                """,
                (persona,),
            ).fetchone()[0]
            self.assertEqual(count, 1)
            sources = {
                row["source_type"]: int(row["count"])
                for row in self.connection.execute(
                    "SELECT source_type, COUNT(*) AS count FROM persona_experiences "
                    "WHERE persona = ? AND status = 'active' GROUP BY source_type",
                    (persona,),
                )
            }
            self.assertEqual(sources.get("lounge_message", 0), 0)
            self.assertEqual(sources["lounge_conversation"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
