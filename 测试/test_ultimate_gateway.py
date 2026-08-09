from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


BACKEND = Path(__file__).resolve().parents[1] / "应用" / "后端"
sys.path.insert(0, str(BACKEND))

import deepseek_gateway as gateway  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.headers = {"x-request-id": "test-request"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *_args):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class UltimateGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "usage.sqlite3"
        self.config = SimpleNamespace(num_predict=512)

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_usage_drives_cost_and_cache_stats(self):
        response = {
            "id": "completion-test",
            "choices": [
                {
                    "message": {"content": "完成"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1_000,
                "prompt_cache_hit_tokens": 800,
                "prompt_cache_miss_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 1_100,
            },
        }
        metrics: dict[str, object] = {}
        with mock.patch.object(gateway, "load_api_key", return_value="test-key"), mock.patch.object(
            gateway.urllib.request, "urlopen", return_value=FakeResponse(response)
        ):
            answer, reason = gateway.call_ultimate(
                "ultimate:aili",
                [{"role": "user", "content": "测试"}],
                self.config,
                database_path=self.db,
                metrics=metrics,
            )
        self.assertEqual(answer, "完成")
        self.assertEqual(reason, "stop")
        self.assertEqual(metrics["total_tokens"], 1_100)
        self.assertAlmostEqual(float(metrics["cost_cny"]), 0.000416)
        summary = gateway.usage_summary(self.db)
        self.assertEqual(summary["today"]["tokens"], 1_100)
        self.assertEqual(summary["today"]["cache_hit_rate"], 0.8)

    def test_exhausted_background_quota_falls_back_without_remote_call(self):
        connection = gateway._connect(self.db)
        try:
            connection.execute(
                """
                INSERT INTO ultimate_usage(
                    occurred_at, local_day, scope, feature, provider_model,
                    total_tokens, status
                ) VALUES (?, ?, 'background', 'test', ?, 150000, 'completed')
                """,
                ("2026-01-01T00:00:00+08:00", gateway._local_day(), gateway.PROVIDER_MODEL),
            )
            connection.commit()
        finally:
            connection.close()
        calls: list[str] = []

        def local(model, _messages, _config, **_kwargs):
            calls.append(model)
            return "本地回答", "stop"

        with mock.patch.object(gateway, "api_available", return_value=True), mock.patch.object(
            gateway, "call_ultimate"
        ) as remote:
            answer, _ = gateway.call_background_preferred(
                local,
                "qwen3.5:9b-16k",
                [{"role": "user", "content": "后台任务"}],
                self.config,
                database_path=self.db,
            )
        self.assertEqual(answer, "本地回答")
        self.assertEqual(calls, ["qwen3.5:9b-16k"])
        remote.assert_not_called()

    def test_background_images_never_reach_remote(self):
        calls: list[str] = []

        def local(model, _messages, _config, **_kwargs):
            calls.append(model)
            return "本地视觉", "stop"

        with mock.patch.object(gateway, "api_available", return_value=True), mock.patch.object(
            gateway, "call_ultimate"
        ) as remote:
            answer, _ = gateway.call_background_preferred(
                local,
                "qwen3.5:9b-16k",
                [{"role": "user", "content": "读图", "images": ["base64"]}],
                self.config,
                database_path=self.db,
            )
        self.assertEqual(answer, "本地视觉")
        self.assertEqual(calls, ["qwen3.5:9b-16k"])
        remote.assert_not_called()

    def test_background_request_uses_stable_prefix(self):
        first = gateway._cache_friendly_messages(
            [{"role": "user", "content": "A"}], background=True
        )
        second = gateway._cache_friendly_messages(
            [{"role": "user", "content": "B"}], background=True
        )
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[-1], second[-1])

    def test_ultimate_always_uses_single_flash_mode(self):
        response = {
            "id": "completion-flash",
            "choices": [
                {"message": {"content": "Flash"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        with mock.patch.object(gateway, "load_api_key", return_value="test-key"), mock.patch.object(
            gateway.urllib.request, "urlopen", return_value=FakeResponse(response)
        ) as urlopen:
            gateway.call_ultimate(
                "ultimate:aili",
                [{"role": "user", "content": "测试"}],
                self.config,
                database_path=self.db,
                think=True,
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], gateway.PROVIDER_MODEL)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)


if __name__ == "__main__":
    unittest.main()
