"""图上的节点。

全部做成**工厂**（``make_x(deps)`` → 节点函数）：依赖从 ``Deps`` 注入，模型调用是其中
一个普通字段。这样整套「解答→核验」循环能在**不调一次 API** 的前提下跑完整 —— 假 LLM
可以按脚本吐错答案、吐坏 JSON、吐超时，把所有分支都走一遍。真去调 API 才试得出来的
分支，等于没试。

**节点绝不往外抛异常。** LangGraph 默认在节点异常时终止整个 run，一道题的接口抖动会
把整卷带走。所有异常都在节点内兜住、写进 ``error``、由路由送到 ``finalize``。
"""

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from edu_mate.core.prompt import render_repair, render_solve, render_verify
from edu_mate.core.schema import (
    Answer,
    QuestionRecord,
    Verification,
    VerificationRecord,
)
from edu_mate.core.service import llm
from edu_mate.core.service.jsonio import JsonParseError, parse_into
from edu_mate.core.service.tiler import decide_feed, page_map_from

__all__ = [
    "Deps",
    "SUBGRAPH_RECURSION_LIMIT",
    "default_call_llm",
    "make_call_llm",
    "make_nodes",
    "route_solve",
    "wrap_record",
]

# 单题子图的超步上限。最坏 3 轮 × (solve + verify) + finalize = 7 步，留到 25 很宽裕。
# **必须在子图自己的 config 里显式传**：子图用的是自己的限额，父级的只管父级 ——
# 这一点是实测出来的，不传就会用默认的 25，看着像没事，但对更长的循环会在中途断掉。
SUBGRAPH_RECURSION_LIMIT = 25

_log = logging.getLogger(__name__)


def make_call_llm(model: str | None = None):
    """造一个模型入口：有图走多模态，没图走纯文本。

    合成一个函数是为了让 ``Deps`` 只需要一个注入点 —— 测试里换假 LLM 时不用去猜
    "这一轮到底会走哪个分支"。有图/无图的判断集中在这里，不散落到各个节点。

    做成工厂而不是直接写个函数，是为了让"覆盖模型"这种参数不必在两处各实现一遍
    （``runner`` 传 ``--model``、``Deps`` 要默认值，两边都得有这条分派逻辑）。
    """
    async def call_llm(images, prompt: str) -> str:
        if images:
            return await llm.achat_with_images(images, prompt, model=model)
        return await llm.achat_text(prompt, model=model)

    return call_llm


default_call_llm = make_call_llm()


@dataclass
class Deps:
    """节点需要的一切外部依赖。

    ``call_llm`` 是唯一的模型入口，签名 ``async (images, prompt) -> str``。
    默认值放在类外，因为 dataclass 的默认值要能在类体里求值，而函数定义要写在前面。
    """

    call_llm: Callable[[list, str], Awaitable[str]] = default_call_llm
    max_rounds: int = 3
    subject: str = "中学"
    tile_dir: str | None = None
    on_event: Callable | None = field(default=None)


def _emit(deps: Deps, kind: str, **info) -> None:
    """进度回调。回调自己出错不能连累节点，所以整段吞掉只记日志。"""
    if deps.on_event is None:
        return
    try:
        deps.on_event(kind, **info)
    except Exception:  # noqa: BLE001
        _log.warning("on_event 回调出错，已忽略", exc_info=True)


def _images_for(deps: Deps, question, page_refs) -> tuple:
    """决定这道题喂哪些图。返回值里第二个是 ``tiled``，要原样传进提示词。"""
    page_map = page_map_from([ref.path for ref in page_refs])
    return decide_feed(question, page_map, out_dir=deps.tile_dir)


async def _ask_parsed(deps: Deps, images, prompt: str, model_cls, label: str):
    """发一次请求并解析成 Pydantic 对象；解析不了就**一次**修复重试。

    只修一次是刻意的：修复重试解决的是"格式没写对"，这种事给两次机会和给一次机会差别
    不大；而反复重试会把一道坏题的成本放大到整卷，还会把真正的问题（模型没读懂题）
    拖成"格式问题"反复重来。第二次还不行就抛出去，由节点记成 ``error``。

    修复**不带图**：要修的是格式，模型手上有自己刚写的那段文字，重排成合法 JSON
    不需要再看题。带图既贵，又多一次顺手改答案的机会。
    """
    raw = await deps.call_llm(images, prompt)
    try:
        return parse_into(model_cls, raw)
    except JsonParseError as exc:
        _emit(deps, "repair", label=label, error=str(exc))
        repaired = await deps.call_llm([], render_repair(exc.raw or raw, exc.hint))
        return parse_into(model_cls, repaired)


def _make_solve(deps: Deps):
    async def solve(state):
        question = state["question"]
        rounds = (state.get("rounds") or 0) + 1

        images = state.get("images")
        tiled = state.get("tiled", False)
        if images is None:
            # 只在第一轮算一次。切块会落盘，后面每轮重算纯属浪费；更要紧的是**每轮看到的
            # 必须是同一份证据**，否则第 2 轮的解答和上一轮的核验对不上号。
            images, tiled = _images_for(deps, question, state.get("page_refs") or [])
        images = list(images or [])
        # tiled 必须跟着**实际投喂的内容**走，不跟着意图走：一张图都没有时它就是 False，
        # 否则提示词会告诉模型"附件是同一页切成的 0 块"。
        tiled = bool(tiled) and bool(images)

        prompt = render_solve(question, len(images), tiled=tiled, subject=deps.subject)
        _emit(deps, "solve_start", question=question, round=rounds, images=len(images))

        try:
            answer = await _ask_parsed(deps, images, prompt, Answer, "解答")
        except Exception as exc:  # noqa: BLE001 —— 兜住一切，见模块说明
            _emit(deps, "solve_failed", question=question, round=rounds, error=str(exc))
            return {
                "rounds": rounds,
                "images": images,
                "tiled": tiled,
                "error": "第 %d 轮解答失败：%s" % (rounds, exc),
            }

        _emit(deps, "solve_done", question=question, round=rounds, answer=answer)
        return {
            "rounds": rounds,
            "images": images,
            "tiled": tiled,
            "answer": answer,
            "error": None,
        }

    return solve


def _make_verify(deps: Deps):
    async def verify(state):
        question = state["question"]
        answer = state.get("answer")
        rounds = state.get("rounds") or 0
        prior_issues = state.get("prior_issues") or []

        if answer is None:
            # 只在解答失败后才可能走到这里，正常路径由 route_solve 挡住了。
            return {"passed": False, "error": "没有解答可核验"}

        images = state.get("images") or []
        tiled = bool(state.get("tiled", False))
        prompt = render_verify(
            question, answer, len(images), tiled=tiled,
            prior_issues=prior_issues or None,
        )
        _emit(deps, "verify_start", question=question, round=rounds)

        try:
            result = await _ask_parsed(deps, images, prompt, Verification, "核验")
        except Exception as exc:  # noqa: BLE001
            _emit(deps, "verify_failed", question=question, round=rounds, error=str(exc))
            # **不把已有答案丢掉**：核验没跑成是"没验证过"，不是"答案错了"。
            # 记录里留着答案、passed=False、error 写明原因，报告如实标注即可。
            return {
                "passed": False,
                "error": "第 %d 轮核验失败：%s" % (rounds, exc),
            }

        # round 由节点填，不取模型返回值：模型数不清自己是第几轮。
        result.round = rounds
        _emit(deps, "verify_done", question=question, round=rounds,
              passed=result.passed, issues=len(result.issues),
              blocking=len(result.blocking_issues), minor=len(result.minor_issues))
        # 模型自己写的 passed 和按 severity 推导出来的不一致时，记一笔。这不是错，
        # 但它是**提示词没把判据教明白**的直接证据 —— 第一版就是靠这个不一致才发现的：
        # 模型写 passed=false，问题却全是"步骤没展开""图不精确"这类不影响结论的。
        if result.declared_passed is not None and bool(result.declared_passed) != result.passed:
            _emit(deps, "verify_override", question=question, round=rounds,
                  declared=bool(result.declared_passed), derived=result.passed)
        # 故意**不返回** error：上面可能有一轮失败留下的 error，解答成功那一轮已经清掉了，
        # 这里再写一次不必要；而写 None 会把不该清的清掉。
        return {
            "verification": result,
            "passed": result.passed,
            # 只把 **blocking** 的带进下一轮。minor 是下一轮要解决不了的问题：重解三次也
            # 不会凭空长出那段推导，带上它们只会让重解去改措辞，并在下一轮被原样再念一遍。
            "prior_issues": list(result.blocking_issues),
        }

    return verify


def route_solve(state):
    """解答完之后往哪走。解答失败就没有可核验的对象，直接收尾。"""
    return "finalize" if state.get("error") else "verify"


def _make_route_verify(deps: Deps):
    def route_verify(state):
        """核验完之后往哪走。

        只有在**明确不通过**时才打回重解。没通过核验、但轮次已用满，就放行 ——
        这正是"照常输出、明确标注未通过核验"的语义所在。
        """
        if state.get("error"):
            return "finalize"
        if state.get("passed"):
            return "finalize"
        if (state.get("rounds") or 0) >= deps.max_rounds:
            return "finalize"
        return "solve"

    return route_verify


def _make_finalize(deps: Deps):
    def finalize(state):
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
            exhausted=(not error) and (not passed) and rounds >= deps.max_rounds,
            error=error,
        )
        _emit(deps, "finalize", question=state.get("question"), record=record)
        return {"record": record}

    return finalize


def make_nodes(deps: Deps) -> dict:
    """按依赖造出这台图需要的全部节点和路由函数。

    ``solve_one`` 不在这里 —— 它要引用编译好的子图，由 ``graph`` 层补上。
    """
    return {
        "solve": _make_solve(deps),
        "verify": _make_verify(deps),
        "route_solve": route_solve,
        "route_verify": _make_route_verify(deps),
        "finalize": _make_finalize(deps),
    }


def wrap_record(question, record) -> QuestionRecord:
    """把单题结果包成整卷记录。题面和解答放在一起，落盘的 JSON 才自解释。"""
    return QuestionRecord(question=question, record=record)
