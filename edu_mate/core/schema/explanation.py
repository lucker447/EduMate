"""试卷讲解 agent 与讲解评分 agent 的数据结构。

**首轮只定义、不实现**：本轮只打通「解答→核验」闭环。先把讲解这半边的位置占住，
是为了让 `QuestionRecord` 的形状一次定死 —— 下一轮往图里加两条边就能接上，
不用回头改已经落盘的 JSON 结构。
"""

from pydantic import Field, field_validator

from edu_mate.core.schema._base import Base, to_bool, to_float

__all__ = ["Explanation", "ExplanationRecord", "Grade", "GradeItem"]


class Explanation(Base):
    """一份课堂讲解稿。"""

    content: str = Field(default="", description="完整讲解稿正文（Markdown）")
    steps: list[str] = Field(
        default_factory=list,
        description="给老师照着讲的步骤，按课堂上实际说的顺序排",
    )
    knowledge_points: list[str] = Field(
        default_factory=list,
        description="这道题串起了哪些知识点",
    )
    blackboard: str = Field(default="", description="板书内容；没有就留空")


class GradeItem(Base):
    """单个维度的评分。"""

    dimension: str = Field(description="维度，如 专业性 / 全面性 / 通俗易懂性 / 逻辑性")
    score: float = Field(default=0.0, description="该维度得分，0~100")
    comment: str = Field(default="", description="扣分理由；满分也要写清好在哪里")


class Grade(Base):
    """对一份讲解稿的评分结论。"""

    items: list[GradeItem] = Field(default_factory=list)
    total: float = Field(default=0.0, description="总分，0~100")
    passed: bool = Field(default=False, description="是否达到交付标准")
    summary: str = Field(default="", description="总体评价与主要不足")

    @field_validator("passed", mode="before")
    @classmethod
    def _coerce_passed(cls, value):
        """与 ``Verification.passed`` 同理：认不出来就报错，不猜。"""
        out = to_bool(value)
        if out is None:
            raise ValueError("passed 只能是 true/false，收到 %r" % (value,))
        return out

    @field_validator("total", mode="before")
    @classmethod
    def _coerce_total(cls, value):
        return to_float(value) or 0.0


class ExplanationRecord(Base):
    """一道题跑完「讲解→评分」循环后的全部结果。"""

    explanation: Explanation | None = None
    grade: Grade | None = None
    rounds: int = 0
    passed: bool = False
    exhausted: bool = False
    error: str | None = None
