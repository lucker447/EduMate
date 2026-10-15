"""试卷解答 agent 的产出。

``answer`` 和 ``choice`` 看起来重复，其实是**两种读法**：``answer`` 是给人看的最终答案
（选择题就是"A"，填空题是填入的内容，解答题是一句话结论），``choice`` 是给程序比对用的
结构化选项字母。选择题上两者通常一致，但把判定押在"解析自然语言里的字母"上很容易被
"A、B 都有可能，选 A" 这类句子带偏。
"""

from pydantic import Field, field_validator

from edu_mate.core.schema._base import Base, to_float

__all__ = ["Answer"]


class Answer(Base):
    """一道题的作答结果。"""

    answer: str = Field(
        description="最终答案，按题型给形态：选择题写选项字母，填空题写填入内容，"
        "解答题写一句话结论"
    )
    choice: str | None = Field(
        default=None,
        description="选择题的选项字母，如 \"A\"；非选择题为 null",
    )
    steps: list[str] = Field(
        default_factory=list,
        description="解题步骤，一步一项；解答题必须完整到符合得分要求",
    )
    reasoning: str = Field(
        default="",
        description="为什么这么做，给核验 agent 看的推理过程；不输出给学生",
    )
    confidence: float | None = Field(
        default=None,
        description="自评把握，0~1",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value):
        """接受 ``0.8`` / ``"0.8"`` / ``"80%"`` / ``"high"``。

        认不出来的值**不报错**，降级为 ``None`` —— 把握度只是参考，不该因为模型
        用了个没见过的说法就让整道题的解析失败。
        """
        return to_float(value)

    @field_validator("choice", mode="before")
    @classmethod
    def _normalize_choice(cls, value):
        """把 ``"a"`` / ``"A."`` / ``"选项A"`` 统一成 ``"A"``，认不出就原样留着。

        不在这里丢弃非法值：真出现 ``"AB"`` 这种多选结果时，保留原样比强行截成
        ``"A"`` 更诚实，核验 agent 也能顺着它发现问题。
        """
        if not isinstance(value, str):
            return value
        text = value.strip().strip(".(（)）")
        if text.startswith("选项"):
            text = text[2:].strip()
        if len(text) <= 2 and text.isalpha() and text.isascii():
            return text.upper()
        return value
