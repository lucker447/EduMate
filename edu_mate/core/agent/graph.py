"""图的装配：外层 map-reduce + 每题一个子图。

```
外层   START ─(Send 扇出)─▶ solve_one ×N ─▶ collect ─▶ END

子图   START ─▶ solve ─▶ verify ─▶ route ─┐
                 ▲                         │ 不通过且轮次没满
                 └─────────────────────────┘
                                           └─▶ finalize ─▶ END
```

**两个循环做成边，不是节点里的 while。** 代价是图上多几个节点，换来的是每一轮都能在
trace 里看见（哪一轮打回、为什么打回、打回了几次），而 `while` 只有一个不透明的节点。
路由函数只读状态、不改状态，所以"第几轮""用没用满""算不算 exhausted"这些判断只有
``finalize`` 一处定义。

**扇出用 ``Send``。** ``Send`` 的分支只拿到显式传进去的那几个 key，不是全量状态 ——
正好用来把题目互相隔离，省得一道题的处理污染另一道。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from edu_mate.core.agent.nodes import (
    SUBGRAPH_RECURSION_LIMIT,
    Deps,
    default_call_llm,
    make_nodes,
    wrap_record,
)
from edu_mate.core.agent.state import PaperState, QuestionState
from edu_mate.core.schema import VerificationRecord

__all__ = ["build_graph", "build_question_graph"]

# 外层超步上限。扇出、汇聚各算一步，实际只走 3、4 步；50 是给"将来在收尾前加节点"留的。
OUTER_RECURSION_LIMIT = 50


def build_question_graph(deps: Deps):
    """单题子图：「解答→核验」循环。"""
    nodes = make_nodes(deps)

    builder = StateGraph(QuestionState)
    builder.add_node("solve", nodes["solve"])
    builder.add_node("verify", nodes["verify"])
    builder.add_node("finalize", nodes["finalize"])

    builder.add_edge(START, "solve")
    # 解答失败就没有可核验的对象，直接收尾 —— 不能让 verify 拿到一个 None 去编。
    builder.add_conditional_edges(
        "solve", nodes["route_solve"], {"verify": "verify", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "verify", nodes["route_verify"], {"solve": "solve", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile()


def build_graph(deps: Deps | None = None):
    """整卷图。返回 ``(编译好的图, deps)``。

    连 ``deps`` 一起返回，是因为调用方通常需要 ``deps.max_rounds`` 去判退出码、
    写 ``meta``；让它自己再拼一份 Deps 出来，两份就有走偏的可能。
    """
    if deps is None:
        deps = Deps()

    subgraph = build_question_graph(deps)

    async def solve_one(payload):
        """跑一道题的完整子图。

        子图是**在节点里显式 ainvoke** 的，不是把编译结果直接当 ``Send`` 的目标 ——
        后者在嵌套时有已知的坑，而且显式调用更好单测（能单独喂一个 payload 进去看输出）。

        限额**在子图自己的 config 里再传一遍**：子图用的是自己的 ``recursion_limit``，
        父级那份管不到它。
        """
        question = payload["question"]
        result = await subgraph.ainvoke(
            payload, config={"recursion_limit": SUBGRAPH_RECURSION_LIMIT}
        )
        record = result.get("record")
        if record is None:
            # 子图跑完却没有 record，说明装配漏了边。宁可出一条写明原因的记录，
            # 也不要静默少一道题 —— 少题比错题更难发现。
            record = VerificationRecord(error="子图没有产出结果（装配错误）")
        return {"records": [wrap_record(question, record)]}

    def dispatch(state):
        """``Send`` 扇出。

        **空的 ``Send`` 列表会静默结束整个 run**（一个分支都不发，连收尾节点都不跑），
        所以没有题目时必须路由到一个真实节点。正常路径下 ``runner`` 会先拦住空卷，
        这里只是不让一个畸形状态把整台图变成"什么都没发生"。
        """
        questions = state.get("questions") or []
        if not questions:
            return "collect"
        return [
            Send("solve_one", {"question": q, "page_refs": state.get("page_refs") or []})
            for q in questions
        ]

    def collect(state):
        """收尾：重排 + 汇总。

        **重排这一步不能省。** ``records`` 的 reducer 是 ``operator.add``，合并顺序就是
        分支完成顺序 —— 并行下必然乱，而且每次运行都不一样。只有 ``question.order``
        是稳定的，按它排。

        **排序结果写到 ``ordered`` 而不是 ``records``**：``records`` 带 ``operator.add``
        reducer，往它回一个排好序的列表只会被**追加**到原列表后面，得到一个每项两份的结果。
        换个 key 是最省事的正解，也比"回空再让别处去读"清楚。
        """
        records = list(state.get("records") or [])
        records.sort(key=lambda item: item.question.order)
        return {"ordered": records}

    builder = StateGraph(PaperState)
    builder.add_node("solve_one", solve_one)
    builder.add_node("collect", collect)
    builder.add_conditional_edges(START, dispatch, ["solve_one", "collect"])
    # 扇出的 N 个 solve_one 在这里自动汇聚：全部跑完才轮到 collect。
    builder.add_edge("solve_one", "collect")
    builder.add_edge("collect", END)
    return builder.compile(), deps
