"""跨层共用的枚举。

题型放在 schema 层而不是 service 层，是因为它是**跨层词汇表**：service 解析时判它、
agent 按它分流、frontend 展示时翻成人话。谁都不该私自定义第二套。
"""

from enum import Enum

__all__ = ["QuestionType"]

_TYPE_LABELS = {
    "choice": "选择题",
    "fill": "填空题",
    "answer": "解答题",
}


class QuestionType(str, Enum):
    """题型。

    取值用英文小写并且混入 ``str``，是为了三件事同时成立：进 JSON 是人读的字符串、
    进数据库不用转换、和字面量比较时 ``QuestionType.CHOICE == "choice"`` 直接为真。
    """

    CHOICE = "choice"
    FILL = "fill"
    ANSWER = "answer"

    @property
    def label(self) -> str:
        """中文名，只给展示层用；逻辑判断一律用枚举本身。"""
        return _TYPE_LABELS[self.value]
