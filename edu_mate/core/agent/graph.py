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

**运行时依赖走 ``config["configurable"]``，不建 Deps 对象。** 节点签名一律
``(state, config)``，参数在**调用时**传进来，而不是创建节点时闭包捕获 —— 所以节点就是
普通函数，不需要工厂。``configurable`` 不参与状态合并、不进 checkpoint，正适合放
函数和回调这类没法序列化的东西。

可用的 configurable key：

==================  ==================================
``call_llm``        模型入口，``async (images, prompt) -> str``
``max_rounds``      每题最多几轮「解答→核验」，默认 3
``subject``         学科，进提示词，默认「中学」
``tile_dir``        切块缓存放哪；不给则由 tiler 落系统临时目录
``on_event``        进度回调 ``on_event(kind, **info)``
==================  ==================================

测试时只要 ``graph.ainvoke(state, config={"configurable": {"call_llm": fake}})``，
整套失败分支都能在不调 API 的前提下跑完。
"""

import logging
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from edu_mate.core.agent.state import PaperState, QuestionState
from edu_mate.core.prompt import render_repair, render_solve, render_verify
from edu_mate.core.schema import (
    Answer,
    QuestionRecord,
    Verification,
    VerificationRecord,
)
from edu_mate.core.service import llm, tracing
from edu_mate.core.service.jsonio import JsonParseError, parse_into
from edu_mate.core.service.tiler import decide_feed, page_map_from

__all__ = [
    "OUTER_RECURSION_LIMIT",
    "SUBGRAPH_RECURSION_LIMIT",
    "build_graph",
    "build_question_graph",
    "check_node",
    "default_call_llm",
    "finalize_node",
    "make_call_llm",
    "solve_node",
    "wrap_record",
]

# 外层超步上限。扇出、汇聚各算一步，实际只走 3、4 步；50 是给"将来在收尾前加节点"留的。
OUTER_RECURSION_LIMIT = 50

# 单题子图的超步上限。最坏 3 轮 × (solve + verify) + finalize = 7 步，留到 25 很宽裕。
# **必须在子图自己的 config 里显式传**：子图用的是自己的限额，父级的只管父级 ——
# 这一点是实测出来的，不传就会用默认的 25，看着像没事，但对更长的循环会在中途断掉。
SUBGRAPH_RECURSION_LIMIT = 25

_log = logging.getLogger(__name__)


def _opt(config: RunnableConfig | None, key: str, default=None):
    """从 ``config["configurable"]`` 取运行时参数。

    统一走这一个函数，是为了让"没给 config / 没给这个 key"两种情况都退化成同一个
    默认值 —— 节点里就不会出现 ``config["configurable"]["x"]`` 那种一缺就 KeyError 的写法。
    """
    return (config or {}).get("configurable", {}).get(key, default)


def _emit(config: RunnableConfig | None, kind: str, **info) -> None:
    """进度回调。回调自己出错不能连累节点，所以整段吞掉只记日志。"""
    callback = _opt(config, "on_event")
    if callback is None:
        return
    try:
        callback(kind, **info)
    except Exception:  # noqa: BLE001
        _log.warning("on_event 回调出错，已忽略", exc_info=True)


def make_call_llm(model: str | None = None):
    """造一个模型入口：有图走多模态，没图走纯文本。

    合成一个函数是为了让 configurable 只需要一个注入点 —— 测试里换假 LLM 时不用去猜
    "这一轮到底会走哪个分支"。有图/无图的判断集中在这里，不散落到各个节点。
    """
    async def call_llm(images, prompt: str) -> str:
        if images:
            return await llm.achat_with_images(images, prompt, model=model)
        return await llm.achat_text(prompt, model=model)

    return call_llm


default_call_llm = make_call_llm()


def _call_llm(config: RunnableConfig | None) -> Callable[[list, str], Awaitable[str]]:
    """取模型入口；configurable 里没给就用真模型。"""
    return _opt(config, "call_llm") or default_call_llm


def _images_for(config: RunnableConfig | None, question, page_refs) -> tuple:
    """决定这道题喂哪些图。返回值里第二个是 ``tiled``，要原样传进提示词。"""
    page_map = page_map_from([ref.path for ref in page_refs])
    return decide_feed(question, page_map, out_dir=_opt(config, "tile_dir"))


async def _ask_parsed(config, images, prompt: str, model_cls, label: str):
    """发一次请求并解析成 Pydantic 对象；解析不了就**一次**修复重试。

    只修一次是刻意的：修复重试解决的是"格式没写对"，这种事给两次机会和给一次机会差别
    不大；而反复重试会把一道坏题的成本放大到整卷，还会把真正的问题（模型没读懂题）
    拖成"格式问题"反复重来。第二次还不行就抛出去，由节点记成 ``error``。

    修复**不带图**：要修的是格式，模型手上有自己刚写的那段文字，重排成合法 JSON
    不需要再看题。带图既贵，又多一次顺手改答案的机会。
    """
    call = _call_llm(config)
    raw = await call(images, prompt)
    try:
        return parse_into(model_cls, raw)
    except JsonParseError as exc:
        _emit(config, "repair", label=label, error=str(exc))
        repaired = await call([], render_repair(exc.raw or raw, exc.hint))
        return parse_into(model_cls, repaired)


async def solve_node(state: QuestionState, config: RunnableConfig) -> dict:
    """解答一轮。"""
    question = state["question"]
    rounds = (state.get("rounds") or 0) + 1

    # 一轮一个 span，名字里带轮次 —— 这是"看得见循环"的关键。切块（_images_for）
    # 也算在这一轮里，因为它确实是一次真实的耗时。
    with tracing.span(
        f"第 {rounds} 轮 · 解答",
        metadata={"round": rounds, "question": str(question.number)},
    ) as traced:
        images = state.get("images")
        tiled = state.get("tiled", False)
        if images is None:
            # 只在第一轮算一次。切块会落盘，后面每轮重算纯属浪费；更要紧的是**每轮看到的
            # 必须是同一份证据**，否则第 2 轮的解答和上一轮的核验对不上号。
            images, tiled = _images_for(config, question, state.get("page_refs") or [])
        images = list(images or [])
        # tiled 必须跟着**实际投喂的内容**走，不跟着意图走：一张图都没有时它就是 False，
        # 否则提示词会告诉模型"附件是同一页切成的 0 块"。
        tiled = bool(tiled) and bool(images)

        traced.update(
            input={
                "题号": question.number,
                "图片": len(images),
                "tiled": tiled,
                # 进入本节点时的完整 QuestionState：第 1 轮只有题面和页图，
                # 第 2 轮起还会带上上一轮的 answer / verification / prior_issues。
                "state": tracing.snapshot(state),
            }
        )

        prompt = render_solve(
            question,
            len(images),
            tiled=tiled,
            subject=_opt(config, "subject", "中学"),
        )
        _emit(config, "solve_start", question=question, round=rounds, images=len(images))

        try:
            answer = await _ask_parsed(config, images, prompt, Answer, "解答")
        except Exception as exc:  # noqa: BLE001 —— 兜住一切，见模块说明
            _emit(config, "solve_failed", question=question, round=rounds, error=str(exc))
            update = {
                "rounds": rounds,
                "images": images,
                "tiled": tiled,
                "error": "第 %d 轮解答失败：%s" % (rounds, exc),
            }
            # state_update 就是返回给 LangGraph 合并的那份增量，原样记下来。
            traced.update(
                output={"error": update["error"], "state_update": tracing.snapshot(update)},
                level="ERROR",
            )
            return update

        _emit(config, "solve_done", question=question, round=rounds, answer=answer)
        update = {
            "rounds": rounds,
            "images": images,
            "tiled": tiled,
            "answer": answer,
            "error": None,
        }
        # 增量（update）和合并后的状态（下一个节点的 input）能对上，才叫"看得见状态变化"。
        traced.update(
            output={"answer": answer.answer, "state_update": tracing.snapshot(update)}
        )
        return update


async def check_node(state: QuestionState, config: RunnableConfig) -> dict:
    """核验一轮。"""
    question = state["question"]
    answer = state.get("answer")
    rounds = state.get("rounds") or 0
    prior_issues = state.get("prior_issues") or []

    if answer is None:
        # 只在解答失败后才可能走到这里，正常路径由 solve_route 挡住了。
        return {"passed": False, "error": "没有解答可核验"}

    with tracing.span(
        f"第 {rounds} 轮 · 核验",
        metadata={"round": rounds, "question": str(question.number)},
    ) as traced:
        images = state.get("images") or []
        tiled = bool(state.get("tiled", False))
        traced.update(input={
            "图片": len(images),
            "tiled": tiled,
            "上一轮阻断问题": len(prior_issues),
            # 这一步读到的完整状态：包含本轮要核的 answer，以及上一轮的 prior_issues。
            "state": tracing.snapshot(state),
        })
        prompt = render_verify(
            question,
            answer,
            len(images),
            tiled=tiled,
            prior_issues=prior_issues or None,
        )
        _emit(config, "verify_start", question=question, round=rounds)

        try:
            result = await _ask_parsed(config, images, prompt, Verification, "核验")
        except Exception as exc:  # noqa: BLE001
            _emit(config, "verify_failed", question=question, round=rounds, error=str(exc))
            update = {
                "passed": False,
                "error": "第 %d 轮核验失败：%s" % (rounds, exc),
            }
            traced.update(
                output={"error": update["error"], "state_update": tracing.snapshot(update)},
                level="ERROR",
            )
            # **不把已有答案丢掉**：核验没跑成是"没验证过"，不是"答案错了"。
            # 记录里留着答案、passed=False、error 写明原因，报告如实标注即可。
            return update

        # round 由节点填，不取模型返回值：模型数不清自己是第几轮。
        result.round = rounds
        _emit(config, "verify_done", question=question, round=rounds,
              passed=result.passed, issues=len(result.issues),
              blocking=len(result.blocking_issues), minor=len(result.minor_issues))
        # 模型自己写的 passed 和按 severity 推导出来的不一致时，记一笔。这不是错，
        # 但它是**提示词没把判据教明白**的直接证据 —— 第一版就是靠这个不一致才发现的：
        # 模型写 passed=false，问题却全是"步骤没展开""图不精确"这类不影响结论的。
        if result.declared_passed is not None and bool(result.declared_passed) != result.passed:
            _emit(config, "verify_override", question=question, round=rounds,
                  declared=bool(result.declared_passed), derived=result.passed)
        update = {
            "verification": result,
            "passed": result.passed,
            # 只把 **blocking** 的带进下一轮。minor 是下一轮要解决不了的问题：重解三次也
            # 不会凭空长出那段推导，带上它们只会让重解去改措辞，并在下一轮被原样再念一遍。
            "prior_issues": list(result.blocking_issues),
        }
        traced.update(output={
            "passed": result.passed,
            "blocking": len(result.blocking_issues),
            "issues": [
                {"severity": issue.severity, "kind": issue.kind}
                for issue in result.issues
            ],
            "independent_answer": result.independent_answer,
            # 下一轮 solve 的 input.state 里会原样出现这份 verification / prior_issues。
            "state_update": tracing.snapshot(update),
        })
        # 故意**不返回** error：上面可能有一轮失败留下的 error，解答成功那一轮已经清掉了，
        # 这里再写一次不必要；而写 None 会把不该清的清掉。
        return update


def solve_route(state: QuestionState) -> str:
    """解答完之后往哪走。解答失败就没有可核验的对象，直接收尾。"""
    return "finalize" if state.get("error") else "verify"


def verify_route(state: QuestionState, config: RunnableConfig) -> str:
    """核验完之后往哪走。

    只有在**明确不通过**时才打回重解。没通过核验、但轮次已用满，就放行 ——
    这正是"照常输出、明确标注未通过核验"的语义所在。
    """
    if state.get("error"):
        return "finalize"
    if state.get("passed"):
        return "finalize"
    if (state.get("rounds") or 0) >= _opt(config, "max_rounds", 3):
        return "finalize"
    return "solve"


def finalize_node(state: QuestionState, config: RunnableConfig) -> dict:
    """收尾：把状态压成一个 ``VerificationRecord``。

    ``exhausted`` **只在这里算**，路由函数碰都不碰状态。这样"用尽放行并标注未通过"
    的语义只有一个地方定义，不会出现两处判断不一致。
    """
    rounds = state.get("rounds") or 0
    passed = bool(state.get("passed"))
    error = state.get("error") or None
    record = VerificationRecord(
        answer=state.get("answer"),
        verification=state.get("verification"),
        rounds=rounds,
        passed=passed,
        # 有 error 就是"没跑成"，不是"跑满了仍不合格" —— 两者在报告里要分开说，
        # 前者要人去查接口，后者要人去看题。合并成一个字段会把排查方向指错。
        exhausted=(not error) and (not passed)
        and rounds >= _opt(config, "max_rounds", 3),
        error=error,
    )
    _emit(config, "finalize", question=state.get("question"), record=record)
    return {"record": record}


def wrap_record(question, record) -> QuestionRecord:
    """把单题结果包成整卷记录。题面和解答放在一起，落盘的 JSON 才自解释。"""
    return QuestionRecord(question=question, record=record)


def build_question_graph() -> CompiledStateGraph:
    """单题子图：「解答→核验」循环。"""
    builder = StateGraph(QuestionState)
    builder.add_node("solve", solve_node)
    builder.add_node("verify", check_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "solve")
    # 解答失败就没有可核验的对象，直接收尾 —— 不能让 verify 拿到一个 None 去编。
    builder.add_conditional_edges(
        "solve", solve_route, {"verify": "verify", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "verify", verify_route, {"solve": "solve", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile()


def build_graph() -> CompiledStateGraph:
    """整卷图。运行时参数由 ``ainvoke(..., config=...)`` 传进来，装配时不需要任何依赖。"""
    subgraph = build_question_graph()

    async def solve_one(payload, config: RunnableConfig) -> dict:
        """跑一道题的完整子图。

        子图是**在节点里显式 ainvoke** 的，不是把编译结果直接当 ``Send`` 的目标 ——
        后者在嵌套时有已知的坑，而且显式调用更好单测（能单独喂一个 payload 进去看输出）。

        ``config`` 必须**原样传下去**：``configurable`` 里的 ``call_llm``/``max_rounds``
        等全在父级配置里，子图拿不到它就会退回默认值 —— 测试里就变成了真的去调 API。
        只覆盖 ``recursion_limit``，子图用的是自己的限额，父级那份管不到它。
        """
        question = payload["question"]
        # 一题一个 span，把下面每一轮的 span 收在一起；并行的题各是一棵子树。
        with tracing.span(
            f"题目 {question.number}",
            input={"题号": question.number, "题型": question.qtype.value},
        ) as traced:
            sub_config = {**config, "recursion_limit": SUBGRAPH_RECURSION_LIMIT}
            result = await subgraph.ainvoke(payload, config=sub_config)
            record = result.get("record")
            if record is None:
                # 子图跑完却没有 record，说明装配漏了边。宁可出一条写明原因的记录，
                # 也不要静默少一道题 —— 少题比错题更难发现。
                record = VerificationRecord(error="子图没有产出结果（装配错误）")
            traced.update(output={
                "轮次": record.rounds,
                "通过": record.passed,
                "exhausted": record.exhausted,
                "error": record.error,
                # 子图跑完时的终态 QuestionState（含 record）。finalize 是同步节点，
                # 没能在它自己身上埋点，所以这里补一份"它看到的最终状态"。
                "state": tracing.snapshot(result),
            })
            return {"records": [wrap_record(question, record)]}

    def dispatch(state):    # state为 PaperState 类型
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
    return builder.compile()
