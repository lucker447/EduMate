"""数据模型层：各层共用的词汇表。

和 ``service`` / ``agent`` / ``prompt`` 平级。这一层的规矩是**无 IO、无模型调用、无业务
逻辑** —— 只有字段定义和脏数据归一。因此它可以被任何一层 import 而不引入依赖方向问题，
也可以被离线测试单独覆盖。

``explanation`` 里的类型首轮只定义不使用，占位是为了让 ``QuestionRecord`` 的形状一次定死。
"""

from edu_mate.core.schema._base import Base
from edu_mate.core.schema.answer import Answer
from edu_mate.core.schema.enums import QuestionType
from edu_mate.core.schema.explanation import (
    Explanation,
    ExplanationRecord,
    Grade,
    GradeItem,
)
from edu_mate.core.schema.paper import PaperResult, QuestionRecord
from edu_mate.core.schema.question import Option, PageRef, Question, TileRef
from edu_mate.core.schema.verification import (
    BLOCKING,
    MINOR,
    Verification,
    VerificationIssue,
    VerificationRecord,
    to_severity,
)

__all__ = [
    "Answer",
    "BLOCKING",
    "Base",
    "Explanation",
    "ExplanationRecord",
    "Grade",
    "GradeItem",
    "MINOR",
    "Option",
    "PageRef",
    "PaperResult",
    "Question",
    "QuestionRecord",
    "QuestionType",
    "TileRef",
    "Verification",
    "VerificationIssue",
    "VerificationRecord",
    "to_severity",
]
