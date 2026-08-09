#!/usr/bin/env python3
"""茶话室的调度、记忆隔离和只读观察回归测试。"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "应用" / "后端"))

from api_long_chat import format_retrieved_history, get_persona_memory, open_database  # noqa: E402
import lounge_service as lounge  # noqa: E402


def snapshot(**updates):
    value = {
        "memory_free_percent": 82.0,
        "load_1m": 1.0,
        "load_ratio": 0.05,
        "cpu_count": 18,
        "system_idle_seconds": 2_000.0,
        "app_idle_seconds": 2_000.0,
        "loaded_models": [],
        "sampled_at": "2026-07-31T20:00:00+08:00",
    }
    value.update(updates)
    return value


class LoungeTests(unittest.TestCase):
    def setUp(self):
        self.profile_patcher = patch.object(
            lounge, "update_persona_self_profile", return_value=None
        )
        self.profile_patcher.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.database_path = Path(self.tmp.name) / "memory.sqlite3"
        self.connection = open_database(self.database_path)
        lounge.ensure_lounge_schema(self.connection)

    def tearDown(self):
        self.profile_patcher.stop()
        self.connection.close()
        self.tmp.cleanup()

    def test_default_policy_is_low_frequency(self):
        config = lounge.get_config(self.connection)
        self.assertEqual(config["idle_minutes"], 15)
        self.assertEqual(config["min_interval_minutes"], 180)
        self.assertEqual(config["max_interval_minutes"], 360)
        self.assertEqual(config["max_daily_rounds"], 4)
        self.assertTrue(config["screen_watch_enabled"])
        self.assertEqual(config["screen_min_interval_minutes"], 60)
        self.assertEqual(config["screen_max_interval_minutes"], 180)
        self.assertEqual(config["screen_max_daily"], 6)

    def test_screen_watch_has_independent_resource_and_frequency_gates(self):
        lounge.update_config(
            self.connection,
            {
                "screen_min_interval_minutes": 90,
                "screen_max_interval_minutes": 240,
                "screen_max_daily": 3,
            },
        )
        config = lounge.get_config(self.connection)
        self.assertEqual(config["screen_min_interval_minutes"], 90)
        self.assertEqual(config["screen_max_interval_minutes"], 240)
        self.assertEqual(config["screen_max_daily"], 3)
        ok, reason, _ = lounge.screen_watch_eligibility(
            self.connection,
            manual=True,
            snapshot=snapshot(system_idle_seconds=5),
        )
        self.assertTrue(ok, reason)
        ok, reason, _ = lounge.screen_watch_eligibility(
            self.connection,
            manual=True,
            snapshot=snapshot(memory_free_percent=30),
        )
        self.assertFalse(ok)
        self.assertIn("内存", reason)
        ok, reason, _ = lounge.screen_watch_eligibility(
            self.connection,
            manual=True,
            snapshot=snapshot(loaded_models=["qwen3.5:27b"]),
        )
        self.assertFalse(ok)
        self.assertIn("27B", reason)

    def test_fullscreen_foreground_blocks_background_ai_and_forces_manual_4b(self):
        fullscreen = snapshot(
            fullscreen_active=True,
            state_known=True,
            frontmost_app="全屏游戏",
            frontmost_bundle="example.game",
            frontmost_window_ratio=1.0,
        )
        ok, reason, tier, _ = lounge.evaluate_eligibility(
            self.connection, snapshot=fullscreen
        )
        self.assertFalse(ok)
        self.assertEqual(tier, "4b")
        self.assertIn("全屏游戏", reason)
        ok, reason, tier, _ = lounge.evaluate_eligibility(
            self.connection, manual=True, snapshot=fullscreen
        )
        self.assertTrue(ok, reason)
        self.assertEqual(tier, "4b")
        for manual in (False, True):
            ok, reason, _ = lounge.screen_watch_eligibility(
                self.connection, manual=manual, snapshot=fullscreen
            )
            self.assertFalse(ok)
            self.assertIn("全屏", reason)

    def test_entering_fullscreen_downgrades_then_finishes_background_chat(self):
        fullscreen = snapshot(
            fullscreen_active=True,
            frontmost_app="全屏视频",
        )
        action, reason, _ = lounge._runtime_resource_action("9b", fullscreen)
        self.assertEqual(action, "downgrade")
        self.assertIn("全屏视频", reason)
        action, reason, _ = lounge._runtime_resource_action("4b", fullscreen)
        self.assertEqual(action, "finish")
        self.assertIn("全屏视频", reason)

    def test_resource_gate_and_tier_selection(self):
        ok, _, tier, _ = lounge.evaluate_eligibility(
            self.connection, snapshot=snapshot()
        )
        self.assertTrue(ok)
        self.assertEqual(tier, "9b")
        ok, reason, tier, _ = lounge.evaluate_eligibility(
            self.connection,
            snapshot=snapshot(memory_free_percent=55.0, load_ratio=0.2),
        )
        self.assertTrue(ok)
        self.assertEqual(tier, "4b")
        ok, reason, _, _ = lounge.evaluate_eligibility(
            self.connection,
            snapshot=snapshot(loaded_models=["qwen3.5:27b"]),
        )
        self.assertFalse(ok)
        self.assertIn("27B", reason)
        ok, reason, _, _ = lounge.evaluate_eligibility(
            self.connection,
            snapshot=snapshot(system_idle_seconds=10),
        )
        self.assertFalse(ok)
        self.assertIn("空闲", reason)

    def test_runtime_resource_policy_downgrades_then_gracefully_finishes(self):
        action, reason, _ = lounge._runtime_resource_action(
            "9b", snapshot(memory_free_percent=40.0, load_ratio=0.2)
        )
        self.assertEqual(action, "downgrade")
        self.assertIn("内存", reason)
        action, reason, _ = lounge._runtime_resource_action(
            "4b", snapshot(memory_free_percent=30.0, load_ratio=0.9)
        )
        self.assertEqual(action, "finish")
        self.assertIn("CPU", reason)
        with patch.object(lounge, "user_chat_activity_marker", return_value=100.0):
            safe, reason = lounge._still_safe(
                100.0,
                manual=False,
                snapshot=snapshot(memory_free_percent=5.0, load_ratio=2.0),
            )
        self.assertTrue(safe)
        self.assertEqual(reason, "")

    def test_quality_helpers_and_escalation_never_cross_persona_families(self):
        installed = {
            "qwen3:14b",
            "huihui_ai/qwen3-abliterated:14b-v2",
            "qwen3.5:27b",
            "huihui_ai/qwen3.5-abliterated:27b",
        }
        with patch.object(lounge, "installed_models", return_value=installed):
            self.assertEqual(lounge._quality_helper_if_available("shaya"), "qwen3:14b")
            self.assertEqual(
                lounge._quality_helper_if_available("aili"),
                "huihui_ai/qwen3-abliterated:14b-v2",
            )
            self.assertEqual(
                lounge._generation_model_for_attempt(
                    "shaya", "qwen3.5:4b-16k", 3
                ),
                "qwen3.5:27b",
            )
            self.assertEqual(
                lounge._generation_model_for_attempt(
                    "aili", "huihui_ai/qwen3.5-abliterated:4b-16k", 3
                ),
                "huihui_ai/qwen3.5-abliterated:27b",
            )
            self.assertEqual(
                lounge._generation_model_for_attempt(
                    "aili",
                    "huihui_ai/qwen3.5-abliterated:4b-16k",
                    3,
                    allow_escalation=False,
                ),
                "huihui_ai/qwen3.5-abliterated:4b-16k",
            )

    def test_running_9b_round_downgrades_to_4b_and_completes(self):
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "9b"},
        )
        self.connection.close()
        resource_calls = 0
        models_used = []
        visible_answers = iter(
            (
                "这种时候慢一点也没关系，先把眼前的话说完整。",
                "嗯，换个轻一点的档位继续，反而更从容。",
            )
        )

        def changing_resources():
            nonlocal resource_calls
            resource_calls += 1
            return snapshot(
                memory_free_percent=70.0 if resource_calls == 1 else 40.0,
                load_ratio=0.1,
            )

        def fake_call(model, messages, *_args, **_kwargs):
            models_used.append(model)
            system = str(messages[0]["content"])
            if "【茶话室结束判断】" in system:
                return "END", "stop"
            if "共同回忆整理器" in system:
                return "她们在资源收紧后换用轻量模型完成了交流。", "stop"
            return next(visible_answers), "stop"

        with (
            patch.object(lounge, "resource_snapshot", side_effect=changing_resources),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "_unload_model"),
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
            patch.object(lounge, "MIN_LOUNGE_MESSAGES_BEFORE_DECISION", 2),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertTrue(result["completed"])
        self.assertEqual(result["tier"], "4b")
        self.assertTrue(result["resource_events"])
        self.assertTrue(models_used)
        self.assertTrue(all("4b" in model for model in models_used))
        self.connection = open_database(self.database_path)
        row = self.connection.execute(
            "SELECT model_tier, status FROM lounge_sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(row["model_tier"], "9b→4b")
        self.assertEqual(row["status"], "completed")

    def test_failed_lounge_always_unloads_every_selected_model(self):
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "4b"},
        )
        self.connection.close()
        with (
            patch.object(lounge, "resource_snapshot", return_value=snapshot()),
            patch.object(lounge, "call_ollama", side_effect=RuntimeError("测试生成失败")),
            patch.object(lounge, "_unload_model") as unload,
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertFalse(result["completed"])
        self.assertFalse(lounge.RUN_LOCK.locked())
        self.assertEqual(
            {call.args[0] for call in unload.call_args_list},
            {
                lounge.PERSONAS["aili"].models["4b"],
                lounge.PERSONAS["shaya"].models["4b"],
            },
        )
        self.connection = open_database(self.database_path)

    def test_file_observation_is_read_only_and_not_repeated(self):
        source = Path(self.tmp.name) / "项目计划.md"
        source.write_text("# 项目计划\n只读检查。", encoding="utf-8")
        before = source.stat()
        with patch.object(lounge, "_spotlight_candidates", return_value=[source]):
            candidates = lounge.select_file_candidates(
                self.connection, [self.tmp.name]
            )
        self.assertEqual(len(candidates), 1)
        observed = lounge.observe_candidate(candidates[0])
        self.assertIn("只读检查", observed["excerpt"])
        self.connection.execute(
            """
            INSERT INTO lounge_sessions(trigger_type, model_tier, started_at, status)
            VALUES ('manual', '4b', '2026-07-31T20:00:00+08:00', 'completed')
            """
        )
        lounge._insert_observations(self.connection, 1, [observed])
        with patch.object(lounge, "_spotlight_candidates", return_value=[source]):
            repeated = lounge.select_file_candidates(
                self.connection, [self.tmp.name]
            )
        after = source.stat()
        self.assertEqual(repeated, [])
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
        self.assertEqual(before.st_size, after.st_size)

    def test_exchange_is_separate_from_user_profiles(self):
        aili_before = str(get_persona_memory(self.connection, "aili")["memory"])
        shaya_before = str(get_persona_memory(self.connection, "shaya")["memory"])
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "4b"},
        )
        self.connection.close()
        visible_answers = iter(
            (
                "沙雅，有时候留一点遗憾的结局反而更让人记得住。",
                "艾莉，我也觉得，太圆满反而容易看完就放下。",
                "沙雅，那种没说尽的感觉，会让人自己补完很久。",
                "艾莉，不过收得住也很重要，不然只剩故弄玄虚。",
                "要是前面的线索够扎实，留白才会让人愿意反复想。",
                "是，没有支撑的空白只会像是忘了写完。",
                "所以好的遗憾不是少一块，而是把最后一步留给读者。",
                "而且那一步最好没有唯一答案，否则只是延迟公布。",
                "这样看，圆满和遗憾不是反义词，关键是情绪有没有到位。",
                "嗯，该落地的落地，该余音的余音，这样收尾就够了。",
            )
        )
        decision_calls = 0
        visible_calls = 0
        decision_after_visible = []

        def fake_call(_model, messages, *_args, **_kwargs):
            nonlocal decision_calls, visible_calls
            system = str(messages[0]["content"])
            if "【茶话室结束判断】" in system:
                decision_calls += 1
                decision_after_visible.append(visible_calls)
                return "END", "stop"
            if "共同回忆整理器" in system:
                return "她们聊了有遗憾的结局为什么更让人记住。", "stop"
            visible_calls += 1
            return next(visible_answers), "stop"

        with (
            patch.object(lounge, "resource_snapshot", return_value=snapshot()),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "_unload_model"),
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertTrue(result["completed"])
        self.connection = open_database(self.database_path)
        lounge.ensure_lounge_schema(self.connection)
        rows = self.connection.execute(
            "SELECT speaker, content FROM lounge_messages ORDER BY id"
        ).fetchall()
        self.assertEqual(
            [row["speaker"] for row in rows],
            ["aili", "shaya"] * lounge.MIN_LOUNGE_COMPLETE_ROUNDS,
        )
        self.assertEqual(decision_after_visible, [10, 10])
        self.assertEqual(str(get_persona_memory(self.connection, "aili")["memory"]), aili_before)
        self.assertEqual(str(get_persona_memory(self.connection, "shaya")["memory"]), shaya_before)
        context = lounge.get_lounge_context(self.connection, "aili")
        self.assertIn("遗憾的结局", context)
        self.assertIn("故弄玄虚", context)
        for persona in ("aili", "shaya"):
            experience = self.connection.execute(
                """
                SELECT content FROM persona_experiences
                 WHERE persona = ? AND source_type = 'lounge_conversation'
                """,
                (persona,),
            ).fetchone()
            self.assertIsNotNone(experience)
            self.assertIn("遗憾的结局", experience["content"])
            self.assertIn("故弄玄虚", experience["content"])

    def test_models_can_continue_beyond_four_messages_and_end_themselves(self):
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "4b"},
        )
        self.connection.close()
        # 每一条都要真正推进话题；用“第 N 个角度”只换序号的旧样本
        # 本来就应该被现在的附和/停滞质量门拦下。
        visible_answers = iter(
            (
                "我先挑个最根本的：结局能不能站住，得看人物动机有没有铺够。",
                "动机够了还不行，前面的时间顺序也要让观众拼得回去。",
                "要是旁白把答案全说了，那些铺垫反而没什么用。",
                "我倒想让配角提前做个看似无关的选择，最后再显出它的分量。",
                "那就需要一个能被记住的道具，不然观众很容易漏掉那次选择。",
                "道具出现的场景也得变，同一个地方前后对照会比台词更有力。",
                "前段保留环境音，后段突然抽掉音乐，也能让那个对照更明显。",
                "不过别把安静当万能药，情绪还没到位就留白只会显得断了。",
                "所以前面至少要回收两条伏笔，再把最后一条留给观众。",
                "我不赞成硬塞反转，最后那条线索可以有两种解释，但不能推翻旧事实。",
                "这样的话，中段就得再紧一点，不然观众到结尾已经没耐心拼线索了。",
                "可以把最大的外部冲突提前解决，把最后几分钟留给人物自己的选择。",
                "那个选择必须付出代价，要是什么都没失去，收束就会轻。",
                "代价也不用当场解释，让观众重看时才发现前面那个小动作已经预告了它。",
                "我们的分歧其实只在最后一幕：你想留问号，我更想给一个可验证的落点。",
                "那就让事实落地、感情保留问号，至少观众不会觉得作者忘了写完。",
            )
        )
        decisions = iter(
            ["CONTINUE", "CONTINUE"] * 3 + ["END", "END"]
        )

        def fake_call(_model, messages, *_args, **_kwargs):
            system = str(messages[0]["content"])
            if "【茶话室结束判断】" in system:
                return next(decisions), "stop"
            if "滚动上下文整理器" in system:
                return "她们沿着同一话题交换了多个不同角度，仍保留自然承接点。", "stop"
            if "共同回忆整理器" in system:
                return "她们把一个分歧自然留到了下次。", "stop"
            return next(visible_answers), "stop"

        with (
            patch.object(lounge, "resource_snapshot", return_value=snapshot()),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "_unload_model"),
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertTrue(result["completed"])
        self.assertEqual(result["messages"], 16)
        self.assertEqual(result["termination_reason"], "双方选择自然收尾")
        self.connection = open_database(self.database_path)
        self.assertEqual(
            int(self.connection.execute("SELECT COUNT(*) FROM lounge_messages").fetchone()[0]),
            16,
        )
        for persona in ("aili", "shaya"):
            self.assertEqual(
                int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM persona_experiences "
                        "WHERE persona = ? AND source_type = 'lounge_message'",
                        (persona,),
                    ).fetchone()[0]
                ),
                16,
            )

    def test_recalled_history_contains_time(self):
        text = format_retrieved_history(
            [
                {
                    "session_title": "测试",
                    "role": "user",
                    "content": "记住这件事",
                    "created_at": "2026-07-31T19:20:00+08:00",
                }
            ]
        )
        self.assertIn("2026-07-31T19:20:00+08:00", text)

    def test_lounge_context_does_not_hijack_small_talk(self):
        self.connection.execute(
            """
            INSERT INTO lounge_notes(content, source_paths, confidence, created_at)
            VALUES ('讨论了额度监控的窗口布局', '[]', 'conversation',
                    '2026-07-31T20:00:00+08:00')
            """
        )
        self.connection.commit()
        with patch.object(lounge, "retrieve_lounge_memory", return_value=""):
            self.assertEqual(
                lounge.get_relevant_lounge_context(self.connection, "aili", "你好"),
                "",
            )
            self.assertEqual(
                lounge.get_relevant_lounge_context(self.connection, "aili", "嗯……"),
                "",
            )
            self.assertIn(
                "额度监控",
                lounge.get_relevant_lounge_context(
                    self.connection, "aili", "你们刚才聊了什么？"
                ),
            )
            self.assertIn(
                "额度监控",
                lounge.get_relevant_lounge_context(
                    self.connection, "aili", "额度监控窗口怎么调整"
                ),
            )

    def test_lounge_rag_indexes_and_recalls_old_chat(self):
        self.connection.execute(
            """
            INSERT INTO lounge_sessions(
                trigger_type, model_tier, topic_mode, started_at, status
            ) VALUES ('auto', '4b', 'free', '2026-07-01T20:00:00+08:00', 'completed')
            """
        )
        self.connection.execute(
            """
            INSERT INTO lounge_messages(
                lounge_session_id, speaker, content, created_at
            ) VALUES (1, 'aili', '以前聊过主人游戏里的猫咪NPC。',
                      '2026-07-01T20:01:00+08:00')
            """
        )
        self.connection.commit()
        fake_vectors = lambda inputs, **_kwargs: [[1.0, 0.0] for _ in inputs]
        with patch.object(lounge, "call_embeddings", side_effect=fake_vectors):
            recalled = lounge.retrieve_lounge_memory(
                self.connection, "猫咪NPC后来怎么样"
            )
        self.assertIn("猫咪NPC", recalled)
        count = self.connection.execute(
            "SELECT COUNT(*) FROM lounge_embeddings"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_lounge_normalization_is_short_and_does_not_force_names(self):
        raw = (
            "沙雅，哼，我有三个方案：\n"
            "1. 先继续研究布局。\n2. 再整理实现细节。\n3. 最后给主人写报告。"
        )
        answer = lounge._normalize_lounge_answer(raw, "shaya")
        self.assertFalse(answer.startswith("沙雅"))
        self.assertNotIn("\n", answer)
        self.assertLessEqual(len(answer), 121)
        self.assertEqual(
            lounge._strip_questions_from_reply(
                "我也觉得这种结局更耐想。你是不是也这么觉得？别急着下结论。"
            ),
            "我也觉得这种结局更耐想。别急着下结论。",
        )
        self.assertEqual(
            lounge._limit_reply_to_one_question(
                "你也注意到这个细节了吗？还想继续往下看吗？我觉得先聊这一点就够了。"
            ),
            "你也注意到这个细节了吗？我觉得先聊这一点就够了。",
        )
        self.assertEqual(
            lounge._resolve_persona_pronouns("沙雅最近有点忙。你先说。", "aili", "shaya"),
            "我最近有点忙。你先说。",
        )
        self.assertIn(
            "连续追问",
            lounge._lounge_answer_issue(
                "这事挺有意思的，你觉得呢？我还想听听。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=2,
            ),
        )
        self.assertIn(
            "连续盘问",
            lounge._lounge_answer_issue(
                "你喜欢哪一种？为什么会这样想？",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=3,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "编造现实经历",
            lounge._lounge_answer_issue(
                "我刚路过社区花园，还看到一棵老树。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=2,
            ),
        )

    def test_grounded_chat_blocks_owner_swap_externalized_identity_and_repetition(self):
        self.assertIn(
            "替另一个人格说话",
            lounge._lounge_answer_issue(
                "我：这份 README 是我们自己的。 沙雅：对。 我：那继续看。",
                speaker="aili", other="shaya", owner_names=set(),
                topic_mode="file",
            ),
        )
        self.assertIn(
            "收件人",
            lounge._lounge_answer_issue(
                "嘿，主人！看到这份 README 了没？",
                speaker="aili", other="shaya", owner_names=set(),
                topic_mode="file",
            ),
        )
        self.assertIn(
            "错认成主人",
            lounge._lounge_answer_issue(
                "主人别吐槽了，这明明是系统标签。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="screen",
            ),
        )
        self.assertIn(
            "是否在场",
            lounge._lounge_answer_issue(
                "这个得等主人回来再商量。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="file",
            ),
        )
        self.assertIn(
            "程序自动选出",
            lounge._lounge_answer_issue(
                "主人真会挑时间看这个 README 呢。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="file",
            ),
        )
        self.assertIn(
            "错归给另一个人格",
            lounge._lounge_answer_issue(
                "别光在那分析 App profile 了，赶紧给我透个底。",
                speaker="aili", other="shaya", owner_names=set(),
                topic_mode="screen",
            ),
        )
        self.assertIn(
            "附和",
            lounge._lounge_answer_issue(
                "这样逻辑闭环才稳当。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="screen",
                prior_messages=["屏幕里的工作区触发条件需要继续收紧。"],
            ),
        )
        self.assertIn(
            "没有执行工具",
            lounge._lounge_answer_issue(
                "我这就去查查后台日志，确认数据有没有问题。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="screen",
            ),
        )
        self.assertIn(
            "额度进度不能证明网络",
            lounge._lounge_answer_issue(
                "进度条只剩38%，要确认是不是网络波动导致的延迟。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="screen",
            ),
        )
        self.assertIn(
            "错归",
            lounge._lounge_answer_issue(
                "嘿，我刚看到你的屏幕，你正在用 Safari 整理截图吗？",
                speaker="aili", other="shaya", owner_names=set(),
                topic_mode="screen",
            ),
        )
        self.assertIn(
            "冒认",
            lounge._lounge_answer_issue(
                "嗯，我在整理昨天的截屏，准备把文件分门别类。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="screen",
            ),
        )
        self.assertIn(
            "自身系统",
            lounge._lounge_answer_issue(
                "双人格、跨模型切换和 9B 记忆整理器，感觉这个系统挺会搞社交。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="file",
            ),
        )
        self.assertIn(
            "重复",
            lounge._lounge_answer_issue(
                "双人格和跨模型切换确实细致，那个 9B 记忆整理器也挺懂技术。",
                speaker="shaya", other="aili", owner_names=set(),
                topic_mode="file",
                prior_messages=[
                    "这个 README 写了双人格、跨模型切换和 9B 记忆整理器，技术挺细。"
                ],
            ),
        )

    def test_legacy_grounding_errors_are_quarantined_without_deleting_history(self):
        cursor = self.connection.execute(
            """
            INSERT INTO lounge_sessions(
                trigger_type, model_tier, topic_mode, started_at, finished_at, status
            ) VALUES ('auto', '4b', 'screen', ?, ?, 'completed')
            """,
            ("2026-08-01T02:05:00+08:00", "2026-08-01T02:06:00+08:00"),
        )
        session_id = int(cursor.lastrowid)
        self.connection.execute(
            """
            INSERT INTO lounge_messages(
                lounge_session_id, speaker, content, model, created_at
            ) VALUES (?, 'aili', ?, 'test', ?)
            """,
            (
                session_id,
                "我刚看到你的屏幕，你正在用 Safari 整理截图吗？",
                "2026-08-01T02:05:10+08:00",
            ),
        )
        lounge.record_lounge_message_in_persona_pools(
            self.connection,
            session_id,
            1,
            "aili",
            "我刚看到你的屏幕，你正在用 Safari 整理截图吗？",
            occurred_at="2026-08-01T02:05:10+08:00",
        )
        lounge.ensure_lounge_schema(self.connection)
        session = self.connection.execute(
            "SELECT quality_status, quality_reason FROM lounge_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        self.assertEqual(session["quality_status"], "quarantined")
        self.assertIn("错", session["quality_reason"])
        self.assertEqual(
            int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM lounge_messages WHERE lounge_session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            ),
            1,
        )
        statuses = {
            row["status"]
            for row in self.connection.execute(
                "SELECT status FROM persona_experiences "
                "WHERE json_extract(metadata, '$.lounge_session_id') = ?",
                (session_id,),
            )
        }
        self.assertEqual(statuses, {"quarantined"})
        # 日后回填或重建记忆池时，同一条经历也不得被 UPSERT 激活。
        lounge.record_lounge_message_in_persona_pools(
            self.connection,
            session_id,
            1,
            "aili",
            "我刚看到你的屏幕，你正在用 Safari 整理截图吗？",
            occurred_at="2026-08-01T02:05:10+08:00",
        )
        statuses_after_backfill = {
            row["status"]
            for row in self.connection.execute(
                "SELECT status FROM persona_experiences "
                "WHERE json_extract(metadata, '$.lounge_session_id') = ?",
                (session_id,),
            )
        }
        self.assertEqual(statuses_after_backfill, {"quarantined"})
        self.assertNotIn("你的屏幕", lounge.get_lounge_context(self.connection, "aili"))
        self.assertIn(
            "旧记忆",
            lounge._lounge_answer_issue(
                "今天额度面板又出现了新动向。",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=1,
                topic_mode="memory",
            ),
        )
        self.assertIn(
            "编造现实经历",
            lounge._lounge_answer_issue(
                "刚才瞥见桌面堆了一批新截图。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=2,
                topic_mode="memory",
            ),
        )
        self.assertIn(
            "编造现实经历",
            lounge._lounge_answer_issue(
                "我最近在追一部很有意思的动漫。",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=1,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "现实作品名",
            lounge._lounge_answer_issue(
                "不如聊聊《葬送的星尘》？",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=1,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "虚构的现实经历",
            lounge._lounge_answer_issue(
                "我最近发现了一家咖啡馆，想找个时间去试试。",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=1,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "归给主人",
            lounge._lounge_answer_issue(
                "主人眼光真好，我也喜欢这个想法。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=2,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "现实邀约",
            lounge._lounge_answer_issue(
                "我们一起去试试吧。",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=3,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "线下行动",
            lounge._lounge_answer_issue(
                "那我就跟着你去散个步，边走边聊。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=4,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "待办计划",
            lounge._lounge_answer_issue(
                "正好把手头的待办事项理一理，咱们一起捋顺计划。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=4,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "工作会议",
            lounge._lounge_answer_issue(
                "咱们还是先把正事办妥，接下来有什么计划？",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=3,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "现实媒体经历",
            lounge._lounge_answer_issue(
                "你最近有在看什么书或电影吗？",
                speaker="aili",
                other="shaya",
                owner_names=set(),
                turn=3,
                topic_mode="free",
            ),
        )
        self.assertIn(
            "服务对象",
            lounge._lounge_answer_issue(
                "我的主要任务是帮你把事办好，偶尔也会给你找资料。",
                speaker="shaya",
                other="aili",
                owner_names=set(),
                turn=4,
                topic_mode="free",
            ),
        )

    def test_shaya_refusal_escalates_within_her_own_model_family(self):
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "4b"},
        )
        self.connection.close()
        official_calls = 0

        def fake_call(model, messages, *_args, **_kwargs):
            nonlocal official_calls
            system = str(messages[0]["content"])
            if "【茶话室结束判断】" in system:
                return "END", "stop"
            if "共同回忆整理器" in system:
                return "她们讨论了如何分清明示与推测。", "stop"
            if model == "qwen3.5:4b-16k":
                official_calls += 1
                if official_calls <= 2:
                    return "用户您好，请先授权。", "stop"
                return "艾莉，我认为这条观察需要分清明示和推测。", "stop"
            if "班长型" in system:
                return "艾莉，我认为把明示和推测分开会更稳妥。", "stop"
            return "沙雅，这个话题里有一条值得慢慢想的线索。", "stop"

        with (
            patch.object(lounge, "resource_snapshot", return_value=snapshot()),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "_unload_model"),
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertTrue(result["completed"])
        self.connection = open_database(self.database_path)
        fallback = self.connection.execute(
            """
            SELECT model, metadata FROM lounge_messages
             WHERE speaker = 'shaya' ORDER BY id LIMIT 1
            """
        ).fetchone()
        self.assertEqual(fallback["model"], "qwen3.5:9b-16k")
        self.assertIn('"fallback": true', fallback["metadata"])

    def test_user_message_interrupts_before_next_turn(self):
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "4b"},
        )
        self.connection.close()
        marker = [100.0]

        def fake_call(_model, messages, *_args, **_kwargs):
            if "【茶话室结束判断】" not in str(messages[0]["content"]):
                marker[0] = 101.0
            return "沙雅，这个话题还挺值得慢慢想。", "stop"

        with (
            patch.object(lounge, "resource_snapshot", return_value=snapshot()),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "_unload_model"),
            patch.object(lounge, "user_chat_activity_marker", side_effect=lambda: marker[0]),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertTrue(result["started"])
        self.assertFalse(result["completed"])
        self.assertIn("用户打断了对话", result["reason"])
        self.connection = open_database(self.database_path)
        messages = int(
            self.connection.execute("SELECT COUNT(*) FROM lounge_messages").fetchone()[0]
        )
        self.assertEqual(messages, 1)

    def test_general_user_return_does_not_interrupt_active_lounge(self):
        with (
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
            patch.object(lounge, "user_activity_marker", return_value=999.0),
        ):
            safe, reason = lounge._still_safe(
                100.0,
                manual=False,
                snapshot=snapshot(system_idle_seconds=0),
            )
        self.assertTrue(safe)
        self.assertEqual(reason, "")

        with patch.object(lounge, "user_chat_activity_marker", return_value=101.0):
            safe, reason = lounge._still_safe(
                100.0,
                manual=False,
                snapshot=snapshot(system_idle_seconds=9_999),
            )
        self.assertFalse(safe)
        self.assertIn("用户打断了对话", reason)

    def test_interrupted_chat_is_offered_to_the_next_round_once(self):
        interrupted_at = "2026-08-01T10:00:00+08:00"
        cursor = self.connection.execute(
            """
            INSERT INTO lounge_sessions(
                trigger_type, model_tier, topic_mode, started_at, finished_at,
                status, summary, termination_reason
            ) VALUES ('auto', '4b', 'free', ?, ?, 'interrupted', ?, ?)
            """,
            (
                interrupted_at,
                interrupted_at,
                "主人发来消息，用户打断了对话",
                "主人发来消息，用户打断了对话",
            ),
        )
        source_id = int(cursor.lastrowid)
        self.connection.executemany(
            """
            INSERT INTO lounge_messages(
                lounge_session_id, speaker, content, model, created_at
            ) VALUES (?, ?, ?, 'test-model', ?)
            """,
            (
                (source_id, "shaya", "我觉得这个话题还有一层。", interrupted_at),
                (source_id, "aili", "那一层是什么，先说来听听？", interrupted_at),
            ),
        )
        lounge.update_config(
            self.connection,
            {"inspect_files": False, "model_strategy": "4b"},
        )
        self.connection.commit()
        self.connection.close()
        saw_old_line = False

        def fake_call(_model, messages, *_args, **_kwargs):
            nonlocal saw_old_line
            system = str(messages[0]["content"])
            if "【茶话室结束判断】" in system:
                return "END", "stop"
            if "共同回忆整理器" in system:
                return "她们续上了之前被打断的话题。", "stop"
            joined = "\n".join(str(item.get("content", "")) for item in messages)
            saw_old_line = saw_old_line or "那一层是什么" in joined
            return (
                "就是结局留白不等于什么都不交代。"
                if "班长型" in system
                else "对，线索得给够，剩下的才叫留白。"
            ), "stop"

        with (
            patch.object(lounge, "resource_snapshot", return_value=snapshot()),
            patch.object(lounge, "call_ollama", side_effect=fake_call),
            patch.object(lounge, "_unload_model"),
            patch.object(lounge, "user_chat_activity_marker", return_value=100.0),
        ):
            result = lounge.run_lounge_round(
                str(self.database_path), threading.RLock(), manual=True
            )
        self.assertTrue(result["completed"])
        self.assertEqual(result["topic_mode"], "resume")
        self.assertTrue(saw_old_line)
        self.connection = open_database(self.database_path)
        resumed = self.connection.execute(
            "SELECT * FROM lounge_sessions WHERE id = ?",
            (result["session_id"],),
        ).fetchone()
        self.assertEqual(int(resumed["resume_source_session_id"]), source_id)
        first = self.connection.execute(
            "SELECT speaker FROM lounge_messages WHERE lounge_session_id = ? ORDER BY id LIMIT 1",
            (result["session_id"],),
        ).fetchone()
        self.assertEqual(first["speaker"], "shaya")
        self.assertIsNone(lounge._pending_interrupted_lounge(self.connection))


if __name__ == "__main__":
    unittest.main(verbosity=2)
