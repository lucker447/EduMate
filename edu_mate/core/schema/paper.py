"""整卷级别的结果对象，也是落盘 JSON 的根。

``PaperResult`` 是**唯一**会被 ``model_dump_json`` 写出去的类型，`frontend` 的报告
和以后可能的数据库入库都从它出发。
"""

from pydantic import Field

from edu_mate.core.schema._base import Base
from edu_mate.core.schema.explanation import ExplanationRecord
from edu_mate.core.schema.question import PageRef, Question
from edu_mate.core.schema.verification import VerificationRecord

__all__ = ["PaperResult", "QuestionRecord"]


class QuestionRecord(Base):
    """一道题在整卷结果里的完整记录。

    题目原文和解答结果**放在一起**，JSON 才是自解释的：截图、检索、人工复核都不用
    回头再去找那份 md。代价是体积 —— ``source_md`` 会重复一份题干，这是有意为之。
    """

    question: Question
    record: VerificationRecord = Field(
        default_factory=VerificationRecord,
        description="「解答→核验」循环的结果",
    )
    explanation: ExplanationRecord | None = Field(
        default=None,
        description="「讲解→评分」循环的结果；首轮恒为 null，下一轮接上",
    )


class PaperResult(Base):
    """一份试卷的最终产物。"""

    name: str = Field(description="试卷名，一般是识别产物 md 的文件名主干")
    page_refs: list[PageRef] = Field(
        default_factory=list,
        description="原卷各页；只存路径，不存图片",
    )
    records: list[QuestionRecord] = Field(
        default_factory=list,
        description="按 question.order 排好序",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="整卷级别的提醒，如某页识别失败、某题解析不出选项",
    )
    meta: dict = Field(
        default_factory=dict,
        description="模型名、轮次上限、耗时、生成时间等；也是以后接数据库和知识库的入口",
    )
