"""答案核验 agent 的产出，以及「单题解答→核验」这个循环的最终结果。

核验是**对抗式**的：要求它先独立重算再看解答，抗的是"看到答案就顺着答案找理由"这个
锚定效应。所以这里刻意不设计"核验结果是否采纳"这类字段 —— 模型只能给出 ``passed``
和具体意见，怎么处理由外层图和 ``max_rounds`` 决定。

**问题要分「阻断」和「不影响结论」两类，这是实测逼出来的。** 第一版只有一个不带分类的
``issues``，结果核验把自己当成了阅卷老师：它挑出的毛病（某一步只说"可证"没展开、选项图
画得比例失真、退化的端点情形没提）条条属实，但**条条不影响答案对错** —— 于是两道答案
本来就正确、且核验自己独立算出来的答案也相同的题，被连续打回三轮、烧掉六次请求，
最后交上来的答案一字未改。核验的职责是回答"这个答案还能不能直接信"，不是"这份卷子
能不能得满分"。所以 ``severity`` 是**判据**而不是形容词：只有 ``blocking`` 才拦得住答案。

``passed`` 由 :attr:`Verification.has_blocking` **推导**出来，不采信模型自己写的值：

- 模型写 ``true`` 却列了 ``blocking`` 的问题 —— 这正是最初踩过的"轻易放行"，
  照它的字面值走就会把错答案放过去；
- 模型写 ``false`` 而问题全是 ``minor`` —— 就是上面那个空转的死循环。

它原本的说法留在 :attr:`Verification.declared_passed` 里，只看不判：两者不一致本身就是
"提示词没把判据教明白"的信号，丢了就没法发现。
"""

from pydantic import Field, field_validator, model_validator

from edu_mate.core.schema._base import Base, to_bool
from edu_mate.core.schema.answer import Answer

__all__ = [
    "BLOCKING",
    "MINOR",
    "Verification",
    "VerificationIssue",
    "VerificationRecord",
    "to_severity",
]

BLOCKING = "blocking"
"""答案错，或关键推理错到可能让答案错。**没通过的唯一理由。**"""

MINOR = "minor"
"""不影响这个答案对错的表述问题：列出来给人看，但不拦。"""

# 模型爱用同义词，别让它因为措辞不同就丢掉分类 —— 但**方向必须偏保守**：认不出来一律
# 当阻断。反过来（认不出来当 minor）会让一个没写分类的严重错误静悄悄地通过，这种错没人
# 会去查；而多报一次阻断，人看一眼报告就知道了。
_SEVERITY_WORDS = {
    "blocking": BLOCKING, "block": BLOCKING, "blocker": BLOCKING,
    "major": BLOCKING, "critical": BLOCKING, "serious": BLOCKING,
    "error": BLOCKING, "fatal": BLOCKING, "high": BLOCKING,
    "严重": BLOCKING, "致命": BLOCKING, "错误": BLOCKING, "阻断": BLOCKING,
    "重大": BLOCKING, "关键": BLOCKING,
    "minor": MINOR, "trivial": MINOR, "cosmetic": MINOR, "nit": MINOR,
    "nitpick": MINOR, "info": MINOR, "informational": MINOR, "style": MINOR,
    "low": MINOR, "presentation": MINOR,
    "轻微": MINOR, "瑕疵": MINOR, "提示": MINOR, "建议": MINOR,
    "表述": MINOR, "次要": MINOR, "不影响": MINOR,
}


def to_severity(value) -> str:
    """归一 ``severity``。

    **认不出来返回** ``blocking``，这一点和 ``to_bool`` 的"返回 None、交给调用方决定"
    刻意不同 —— 这里的默认值只有一个合理方向，把选择权散给各调用方只会被误用。
    """
    text = str(value or "").strip().lower()
    return _SEVERITY_WORDS.get(text, BLOCKING)


class VerificationIssue(Base):
    """一处具体问题。"""

    kind: str = Field(
        description="问题类别，如 计算错误 / 边界漏项 / 步骤缺失 / 选项与内容不符 / 依据不足"
    )
    detail: str = Field(description="具体哪里不对")
    suggestion: str = Field(default="", description="该怎么改")
    severity: str = Field(
        default=BLOCKING,
        description="``blocking``（答案可能因此错）或 ``minor``（不影响答案对错）。"
        "**默认取最严重的那个**：模型漏填分类时按阻断处理，宁可多报不可漏放",
    )

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value):
        return to_severity(value)


class Verification(Base):
    """对一份解答的核验结论。"""

    passed: bool = Field(
        default=False,
        description="是否通过核验。**由 issues 里的 severity 推导**，不取模型返回值",
    )
    issues: list[VerificationIssue] = Field(
        default_factory=list,
        description="核验提出的问题。通过时也可以非空 —— 只要里面全是 minor 就行",
    )
    declared_passed: bool | None = Field(
        default=None,
        description="模型**自己写的**那个 passed。只用于发现提示词没把判据教明白："
        "它和推导出的 passed 不一致时，说明模型对判据的理解和代码不一样",
    )
    round: int = Field(
        default=0,
        description="这是第几轮核验（从 1 开始）。由节点填，不取模型返回值",
    )
    independent_answer: str = Field(
        default="",
        description="核验者**自己**独立算出的答案",
    )
    independent_reasoning: str = Field(
        default="",
        description="独立推导过程。保留它不是为了好看 —— 抗锚定是否真的发生，只能靠"
        "这段文字来判断：它若和待核验的解答逐字雷同，说明核验只是在顺着答案找理由",
    )

    @property
    def blocking_issues(self) -> list:
        """会让这个答案不可信的 issue。**只有这些算没通过。**"""
        return [issue for issue in self.issues if issue.severity == BLOCKING]

    @property
    def minor_issues(self) -> list:
        """不影响结论的瑕疵。列出来是为了让人看得见核验确实读了，不是没看出问题。"""
        return [issue for issue in self.issues if issue.severity != BLOCKING]

    @property
    def has_blocking(self) -> bool:
        return any(issue.severity == BLOCKING for issue in self.issues)

    @field_validator("passed", mode="before")
    @classmethod
    def _discard_passed(cls, value):
        """``passed`` 是推导出来的，**送进来的值一律丢弃**。

        这里刻意**不报错**。这个值来自模型，脏得很（``"是"`` / ``"通过"`` / ``"符合"``
        都见过），而它马上就会被 :meth:`_derive_passed` 覆盖掉 —— 为一个注定被丢掉的值
        去触发一次"修复重试"，等于白花一次请求，还让一道好题因为一个无关字段而失败。
        真正要读模型说法的是 :attr:`declared_passed`，那里也不报错，见下。
        """
        return bool(to_bool(value))

    @field_validator("declared_passed", mode="before")
    @classmethod
    def _coerce_declared(cls, value):
        """同样用 ``to_bool`` 尽量认，**认不出来记 ``None`` 而不报错**。

        这个字段只用来观察"模型自己对判据是怎么理解的"，不影响任何判定 —— 为它中断一次
        核验不合算。``None`` 的语义是"它写了个读不出来的东西"，报告里照样看得出来。
        """
        return to_bool(value)

    @model_validator(mode="before")
    @classmethod
    def _capture_declared(cls, data):
        """模型输出里那个 ``passed`` 是它的**说法**，先原样存下来。

        不这么做的话，下面推导时它会当场被覆盖掉，"模型到底怎么说的"就查不到了。
        """
        if isinstance(data, dict) and "passed" in data:
            data = dict(data)
            data.setdefault("declared_passed", data["passed"])
        return data

    @model_validator(mode="after")
    def _derive_passed(self):
        """``passed`` 是**推导**出来的，不是模型说了算。

        这一条把"轻易放行"从提示词问题变成了**不可表达的状态**：模型可以在 ``issues``
        里随便写，但只要有一条 ``blocking``，构造出来的对象就不可能是通过的。
        """
        self.passed = not self.has_blocking
        return self


class VerificationRecord(Base):
    """一道题跑完「解答→核验」循环后的全部结果。

    这是**每题子图的出口对象**，也是 `agent` 层往 `paper` 层交的东西。
    """

    answer: Answer | None = Field(default=None, description="以最后一次解答为准")
    verification: Verification | None = Field(
        default=None,
        description="最后一次核验；连一轮都没跑完时为 null",
    )
    rounds: int = Field(default=0, description="实际跑了几轮")
    passed: bool = Field(default=False, description="最终是否通过核验")
    exhausted: bool = Field(
        default=False,
        description="轮次用满仍不通过。这种结果**照常输出**，但必须在报告里醒目标注",
    )
    error: str | None = Field(
        default=None,
        description="硬失败原因（JSON 解析失败、接口异常等）；非 null 时 answer 可能为空",
    )
