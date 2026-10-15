"""题目本身，以及它和原卷页面的对应关系。

这一层只描述"卷面上有什么"，不含任何解答信息 —— 解答在 ``answer.py``，
两者分开是为了让「解析试卷」这件事在没有任何模型参与时就能独立完成和验证。
"""

from pydantic import Field

from edu_mate.core.schema._base import Base
from edu_mate.core.schema.enums import QuestionType

__all__ = ["Option", "PageRef", "Question", "TileRef"]


class Option(Base):
    """选择题的一个选项。"""

    label: str = Field(description="选项标号，如 \"A\"")
    text: str = Field(
        default="",
        description="选项内容；为空说明该选项是图，内容要看配图",
    )


class PageRef(Base):
    """原卷的一页。

    存**路径**而不是图片本身：这东西会进状态、进日志、进 JSON。
    """

    stem: str = Field(description="页面文件名的主干，如 \"2\"；md 里的页码标记用的就是它")
    path: str = Field(description="页面图片路径，可能相对也可能绝对")


class TileRef(Base):
    """整页按固定网格切出来的一块，给"小图密集"的页面用。"""

    page_stem: str
    path: str
    row: int = Field(description="从 0 数")
    col: int = Field(description="从 0 数")


class Question(Base):
    """一道题，以及它在原卷里的位置。

    题号有两个：``number`` 是**原卷印的**，照抄不改；``order`` 是它在本次解析结果里的
    先后。之所以不能只留一个，是因为 ``order`` 才是**重排依据** —— LangGraph 用
    ``operator.add`` 合并并行分支时，记录的到达顺序必然和题目顺序不一致（其实也每次不同），
    只有 ``order`` 是稳定的。而 ``number`` 可能带小问号（"16(1)"），拿来排序会得出
    "16(1) 在 16(2) 后面"这种错误结果。
    """

    qid: str = Field(description="稳定标识，如 \"q9\"；同一份卷子里唯一")
    order: int = Field(description="全卷序，从 1 开始；并行合并后靠它重排")
    number: str = Field(description="原卷题号，照抄，如 \"9\" 或 \"16(1)\"")
    section: str = Field(default="", description="所属大题标题，如 \"二、填空题\"")
    qtype: QuestionType = QuestionType.ANSWER
    text: str = Field(default="", description="题干（含小题号），不含选项")
    options: list[Option] = Field(
        default_factory=list,
        description="选择题的选项；非选择题为空",
    )
    page_stems: list[str] = Field(
        default_factory=list,
        description="这道题涉及的所有页，跨页题会有多个，如第 9 题是 [\"1\", \"2\"]",
    )
    needs_figure: bool = Field(
        default=False,
        description="题干或选项里有配图（原 md 中的 ``![图: …]`` 占位符）",
    )
    has_image_options: bool = Field(
        default=False,
        description="选项是图，必须切块投喂才能看清",
    )
    option_page: str = Field(
        default="",
        description="选项落在哪一页。跨页题里它和题干页不是同一页（第 9 题题干在第 1 页、"
        "四个选项图在第 2 页），切块只切这一页 —— 题干页已有文字，再压一张整页图进去"
        "只会稀释注意力",
    )
    source_md: str = Field(
        default="",
        description="题干在原卷 md 里的切片（含页码标记与配图占位符）。"
        "**不含选项** —— 选项被抽进了 options 字段，因为选项是结构化的、要单独判。"
        "拼提示词时选项要另行渲染（见 prompt._common.options_block），"
        "只喂 source_md 等于让模型看不到选项说什么。",
    )
