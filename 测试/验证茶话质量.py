#!/usr/bin/env python3
"""茶话室对话质量回归：附和刷屏与语义原地打转必须被拦下。

样本直接取自真实劣化对话（两个人格互相点头、把同一观点换词重讲），
以及正常推进话题的发言，确保拦得住坏的、不误伤好的。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "应用" / "后端"))

from lounge_service import (  # noqa: E402
    _CONVERSATION_MOVES,
    _TOPIC_EXHAUSTED_ISSUE,
    _lounge_echo_issue,
    _lounge_stagnation_issue,
    _normalize_lounge_answer,
    _strip_agreement_openers,
)


# 真实劣化样本：每条都以附和开场，五条讲的是同一个意思。
BAD_THREAD = [
    "嗯，能力再强，天天端着架子谁受得了啊。普通人踏实靠谱，日子过起来才顺心。",
    "确实，能力再强要是总摆架子，相处起来也累。踏实靠谱的人至少让人安心，不用时刻提防着。",
    "嗯，而且那种能力强的人，多半还爱挑剔别人这不行那不行。普通人至少知道互相体谅，日子过得舒坦。",
    "嗯，说得对。能力强的人如果总盯着别人缺点挑，反而容易把关系搞僵。普通人互相体谅，相处起来才轻松自在。",
    "嗯，而且能力强的人往往还自带优越感，聊个天都像在指导工作。普通人相处起来没那么多包袱，反而更能放松做自己。",
]

# 正常推进：举具体例子、提出不同意见、自然换话题。
GOOD_REPLIES = [
    "说到这个，我昨天在那份文档里看到向量检索会随记忆量变慢，不知道主人怎么打算。",
    "我倒觉得不一定。真要说起来，主人上次通宵改的那版反而更稳。",
    "算了不聊这个了。你有没有发现最近屏幕上老是出现同一个网页？",
    "举个例子吧，主人写代码时从来不先画图，直接就开始敲。",
]


class LoungeQualityTest(unittest.TestCase):
    def test_agreement_openers_are_stripped(self) -> None:
        for raw, expected in (
            ("嗯，说得对。能力强的人容易把关系搞僵。", "能力强的人容易把关系搞僵。"),
            ("确实，相处起来也累。", "相处起来也累。"),
            ("嗯嗯，对啊，我也这么觉得，那份文档挺乱的。", "那份文档挺乱的。"),
            ("这倒是真的。跟那种人待一起太累了。", "跟那种人待一起太累了。"),
            ("这确实是个好主意，加上来源更清楚。", "是个好主意，加上来源更清楚。"),
        ):
            self.assertEqual(_strip_agreement_openers(raw), expected, raw)

    def test_normalize_removes_openers(self) -> None:
        self.assertFalse(
            _normalize_lounge_answer("嗯，说得对。这事挺麻烦的。").startswith("嗯")
        )

    def test_agreement_spam_cannot_sustain(self) -> None:
        """附和刷屏必须撑不下去：绝大多数条被拦，聊不成一长串点头。

        设计取舍：宁可放过个别措辞较新的一条，也绝不误伤正常发言——
        误伤会让整轮聊天直接死掉，代价比漏放大得多。
        """
        blocked = 0
        for index in range(1, len(BAD_THREAD)):
            history = BAD_THREAD[:index]
            issue = _lounge_echo_issue(
                BAD_THREAD[index], history
            ) or _lounge_stagnation_issue(BAD_THREAD[index], history)
            if issue:
                blocked += 1
        self.assertGreaterEqual(blocked, 3, "附和刷屏拦得太少，会继续刷屏")
        # 直接复述上一句的那条是最典型的，必须拦住。
        self.assertTrue(_lounge_echo_issue(BAD_THREAD[1], BAD_THREAD[:1]))

    def test_good_replies_are_never_blocked(self) -> None:
        for reply in GOOD_REPLIES:
            issue = _lounge_echo_issue(
                reply, BAD_THREAD
            ) or _lounge_stagnation_issue(reply, BAD_THREAD)
            self.assertFalse(issue, f"正常推进的发言被误伤：{reply}｜{issue}")

    def test_same_topic_deepening_is_not_blocked(self) -> None:
        """回归：同一个话题继续深入是正常的，不能当成原地打转拦掉。

        上一轮就是因为这个把艾莉的第三条卡死，整轮只聊了两句。
        """
        thread = [
            "嘿沙雅，我注意到这后端脚本里有个小细节：在观察结果里加个"
            "“修改时间”字段后，现在可以更方便地追踪每次文件变动了。",
            "加上来源能更清楚是谁改的文件。得想个既能追踪又能灵活调整的方案才行。",
            "不过要是每个字段都记，日志会不会太臃肿？我更想知道主人会不会去翻这些记录。",
        ]
        for reply in (
            "要真做的话，我建议先只记路径和时间戳，其他等真有人翻记录了再说。",
            "说起来主人昨天那份文档里其实提过一嘴，说日志太多反而没人看。",
        ):
            issue = _lounge_echo_issue(
                reply, thread
            ) or _lounge_stagnation_issue(reply, thread)
            self.assertFalse(issue, f"同话题深入被误伤：{reply}｜{issue}")

    def test_pure_agreement_has_no_content(self) -> None:
        self.assertTrue(_lounge_echo_issue("嗯，确实是这样。", BAD_THREAD))
        self.assertTrue(_lounge_echo_issue("对啊，我也这么觉得。", BAD_THREAD))

    def test_short_history_is_not_judged(self) -> None:
        """刚开场没有足够上下文时不能乱拦，否则第一轮就失败。"""
        self.assertFalse(_lounge_stagnation_issue(BAD_THREAD[1], BAD_THREAD[:1]))

    def test_conversation_moves_cover_four_distinct_actions(self) -> None:
        self.assertEqual(len(_CONVERSATION_MOVES), 4)
        joined = "".join(_CONVERSATION_MOVES)
        for keyword in ("例子", "不同", "推一步", "别的事"):
            self.assertIn(keyword, joined)

    def test_exhausted_topic_is_recognized_as_graceful_ending(self) -> None:
        """话题聊尽应当自然收场，而不是当成故障把整轮判失败。"""
        for issue in (
            "连续几条都在围绕同一批词打转，没有新信息；必须换角度、举具体例子或换话题",
            "只是把对方上一句换个说法复述了一遍，没有推进",
            "这条几乎只是附和，没有自己的内容",
        ):
            self.assertTrue(_TOPIC_EXHAUSTED_ISSUE.search(issue), issue)
        # 事实性问题不能被当成“聊尽了”而放过。
        self.assertFalse(
            _TOPIC_EXHAUSTED_ISSUE.search("不得把主人的电脑行为归给任何人格")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
