"""图上的状态。

两个状态，对应两层图：``PaperState`` 是整卷，``QuestionState`` 是**每题子图**。子图由
``Send`` 扇出，每个分支拿到的是一份隔离的 ``QuestionState`` —— 所以状态里可以放心地放
"当前题目"这种单数概念，不会有并发写同一个 key 的问题。

**只有 ``PaperState.records`` 用 ``operator.add``。** 别的 key 都是整份覆盖。
``operator.add`` 的合并顺序是**分支完成顺序**，并行下必然乱、而且每次不同，
所以下游必须按 ``question.order`` 重排，绝不能依赖到达次序 —— 这也是 ``Question``
要同时留 ``order`` 和 ``number`` 的原因：``number`` 可能带小问号（"16(1)"），拿去排序
会得出"16(1) 在 16(2) 后面"这种结果。

**状态里只放路径，不放图片二进制。** 图片会进日志、进 JSON、被复制到每个分支，
放进去只会让状态变得又大又没法读。
"""

import operator
from typing import Annotated, TypedDict

__all__ = ["PaperState", "QuestionState"]


class QuestionState(TypedDict, total=False):
    """单题子图的状态。``total=False`` 是因为节点只回自己改掉的 key。"""

    question: object
    page_refs: list
    images: list
    """已决定好的投喂图片路径。**解答和核验共用同一份** —— 两边要是看到不同的证据，
    核验核的就不是同一道题，独立重算也没有可比性。"""
    tiled: bool
    """``images`` 是不是同一页切出来的块。要一路传到提示词里。"""
    answer: object
    verification: object
    rounds: int
    """已经解答了几轮。由 ``solve`` 节点自增，路由函数只读不写。"""
    prior_issues: list
    """上一轮核验提出的问题，喂给下一轮核验要求逐条回应。"""
    passed: bool
    error: str | None
    record: object
    """``finalize`` 的产物（``VerificationRecord``）。"""


class PaperState(TypedDict, total=False):
    """整卷状态。"""

    name: str
    questions: list
    page_refs: list
    warnings: list
    meta: dict
    records: Annotated[list, operator.add]
    """并行分支各自追加一条，合并顺序 = 完成顺序，**是乱的**。"""

    ordered: list
    """``collect`` 按 ``question.order`` 重排后的结果，**没有 reducer**。

    重排结果不能写回 ``records``：那个 key 带 ``operator.add``，回一个列表进去只会被
    追加到原列表后面，每项变两份。所以另开一个 key。下游一律读 ``ordered``。
    """
