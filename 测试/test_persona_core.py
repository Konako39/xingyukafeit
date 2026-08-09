from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1] / "应用" / "后端"
sys.path.insert(0, str(BACKEND))

from api_long_chat import (  # noqa: E402
    INITIAL_PERSONA_PROFILES,
    LEGACY_INITIAL_PERSONA_PROFILES_V4,
    PERSONAS,
    open_database,
)


class PersonaCoreTests(unittest.TestCase):
    def test_aili_is_easygoing_without_being_deliberately_rude(self):
        prompt = PERSONAS["aili"].system_prompt
        self.assertIn("大大咧咧", prompt)
        self.assertIn("直率不等于故意无礼", prompt)
        self.assertIn("照样温和、有基本礼貌", prompt)
        self.assertIn("开玩笑", prompt)

    def test_shaya_is_shy_rigid_serious_and_gentle(self):
        prompt = PERSONAS["shaya"].system_prompt
        self.assertIn("班长型", prompt)
        self.assertIn("害羞、死板、认真", prompt)
        self.assertIn("并不凶或强势", prompt)
        self.assertIn("平时温和有礼", prompt)

    def test_both_personas_use_short_private_chat_style(self):
        for config in PERSONAS.values():
            self.assertIn("QQ、微信或 Instagram 私聊", config.system_prompt)
            self.assertIn("一到三句", config.system_prompt)
            self.assertIn("小作文", config.system_prompt)

    def test_initial_profiles_preserve_the_same_contrast(self):
        self.assertIn("大大咧咧", INITIAL_PERSONA_PROFILES["aili"])
        self.assertIn("害羞、死板、认真", INITIAL_PERSONA_PROFILES["shaya"])
        self.assertNotEqual(
            INITIAL_PERSONA_PROFILES["aili"],
            INITIAL_PERSONA_PROFILES["shaya"],
        )

    def test_only_untouched_v4_profiles_are_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            connection = open_database(path)
            connection.execute(
                "UPDATE persona_memories SET profile = ? WHERE persona = 'aili'",
                (LEGACY_INITIAL_PERSONA_PROFILES_V4["aili"],),
            )
            grown = INITIAL_PERSONA_PROFILES["shaya"] + "\n\n真实成长内容"
            connection.execute(
                "UPDATE persona_memories SET profile = ? WHERE persona = 'shaya'",
                (grown,),
            )
            connection.commit()
            connection.close()

            connection = open_database(path)
            try:
                profiles = {
                    row["persona"]: row["profile"]
                    for row in connection.execute(
                        "SELECT persona, profile FROM persona_memories"
                    )
                }
            finally:
                connection.close()
            self.assertEqual(profiles["aili"], INITIAL_PERSONA_PROFILES["aili"])
            self.assertEqual(profiles["shaya"], grown)


if __name__ == "__main__":
    unittest.main()
