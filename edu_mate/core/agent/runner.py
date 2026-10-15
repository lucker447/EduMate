"""解答链路的对外唯一入口。

分工：``runner`` 负责"把一份卷子从头跑到尾"（识别 → 解析 → 编排 → 汇总），
``frontend`` 负责把它呈现出来（写文件、印进度、定退出码）。所以这里**不写文件、不打印**，
只回一个 ``PaperResult``。

**识别与解析能被跳过。** 已经识别过一次的卷子（``output/x.md`` 就在那儿）没有理由
再花一遍模型钱，``md=`` 就是为这条路留的。
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from edu_mate.core.agent.graph import OUTER_RECURSION_LIMIT, build_graph
from edu_mate.core.agent.nodes import Deps, make_call_llm
from edu_mate.core.schema import PageRef, PaperResult
from edu_mate.core.service.llm import DEFAULT_MODEL
from edu_mate.core.service.parse import parse_paper
from edu_mate.core.service.paper import recognize_paper

__all__ = ["NoQuestionsError", "arun_paper", "run_paper"]


class NoQuestionsError(RuntimeError):
    """卷子里一道题都没解析出来。

    这是**硬失败**，不是"跑完了但结果为空"：要么 md 是坏的，要么格式和 ``parse_paper``
    的假设不符。两种都得人去看，静默回一份空结果只会让人以为识别没问题。
    """


def _page_refs(page_paths) -> list:
    return [PageRef(stem=path.stem, path=str(path)) for path in page_paths]


def _coverage_warning(questions, page_refs) -> list:
    """识别出哪些题的原页对不上实际页图。

    对不上不会让流程失败 —— 题还能靠转写的文字作答，只是少了"底稿"这份证据。
    但这正是最需要人说一声的情形，所以要报出来。
    """
    have = {ref.stem for ref in page_refs}
    if not have:
        return []
    missing = set()
    for question in questions:
        for stem in question.page_stems:
            if stem not in have:
                missing.add(stem)
    if not missing:
        return []
    return [
        "以下页码在页图里找不到，相关题目只能依据转写文字作答：%s"
        % "、".join(sorted(missing))
    ]


async def arun_paper(
    name: str,
    md: str,
    page_paths=None,
    *,
    jobs: int = 4,
    max_rounds: int = 3,
    subject: str = "中学",
    model: str | None = None,
    call_llm=None,
    on_event=None,
    tile_dir=None,
) -> PaperResult:
    """``run_paper`` 的异步核心。已经识别好的 ``md`` 文本进来，``PaperResult`` 出去。"""
    page_paths = list(page_paths or [])
    page_refs = _page_refs(page_paths)

    warnings = []
    questions = parse_paper(md, warnings=warnings)
    if not questions:
        raise NoQuestionsError("从识别结果里解析不出任何题目")
    warnings.extend(_coverage_warning(questions, page_refs))

    # 限流**自己管**，不指望框架配置：Semaphore 包住每一次模型调用，语义直白、不随
    # LangGraph 版本变。config 里的 max_concurrency 同时传（实测它确实生效），
    # 两者叠加只会更保守，不会更激进。
    gate = asyncio.Semaphore(jobs) if jobs and jobs > 0 else None
    base = call_llm or make_call_llm(model)

    async def gated(images, prompt):
        if gate is None:
            return await base(images, prompt)
        async with gate:
            return await base(images, prompt)

    deps = Deps(
        call_llm=gated, max_rounds=max_rounds, subject=subject,
        tile_dir=tile_dir, on_event=on_event,
    )
    graph, _ = build_graph(deps)

    started = time.monotonic()
    state = await graph.ainvoke(
        {"name": name, "questions": questions, "page_refs": page_refs},
        config={
            "recursion_limit": OUTER_RECURSION_LIMIT,
            "max_concurrency": jobs if jobs and jobs > 0 else None,
        },
    )
    elapsed = time.monotonic() - started

    records = state.get("ordered") or []
    return PaperResult(
        name=name,
        page_refs=page_refs,
        records=records,
        warnings=warnings,
        meta={
            "model": model or DEFAULT_MODEL,
            "max_rounds": max_rounds,
            "jobs": jobs,
            "question_count": len(records),
            "elapsed_seconds": round(elapsed, 1),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


def run_paper(
    name: str,
    pages=None,
    *,
    md_path=None,
    out_dir=None,
    jobs: int = 4,
    max_rounds: int = 3,
    subject: str = "中学",
    model: str | None = None,
    call_llm=None,
    on_event=None,
    on_page=None,
    tile_dir=None,
) -> PaperResult:
    """跑一份试卷的解答链路。

    Args:
        name: 试卷名，一般是目录名；用 ``--md`` 复用时由 cli 的 ``_paper_name`` 推出来。
        pages: 页面图片路径（已排序）。``md_path`` 为 ``None`` 时会先识别它们。
        md_path: 复用已经识别好的 md，跳过识别这一步。给了它 ``pages`` 就只用来
            提供页图（解答要看原图，仍然需要）。
        out_dir: **这份试卷**的输出目录（cli 传进来的是 ``<输出根>/<试卷名>/``）。
            切块缓存放它的 ``.tiles/`` 下，一份试卷一个，清卷子时缓存跟着走；
            传 ``None`` 则落到 ``tiler.default_tile_dir()``（系统临时目录）。
        jobs: 并发上限。
        max_rounds: 每题「解答→核验」最多几轮。
        on_event: ``on_event(kind, **info)``，进度回调。走 stderr 还是日志由调用方定。
        on_page: 识别阶段的逐页回调 ``on_page(index, total, path, error)``。

    Raises:
        NoQuestionsError: 一道题都没解析出来。
        ValueError: 既没给 ``md_path`` 也没给 ``pages``。
    """
    page_paths = [Path(p) for p in (pages or [])]

    if md_path is not None:
        md = Path(md_path).read_text(encoding="utf-8")
    elif page_paths:
        md = recognize_paper(
            [str(p) for p in page_paths], model=model, on_page=on_page
        )
    else:
        raise ValueError("至少要给 pages 或 md_path 之一")

    if tile_dir is None and out_dir is not None:
        # 给了 -o 就把切块缓存放进输出目录，清输出时缓存跟着走。
        # 没给的话交给 tiler 的 default_tile_dir()（系统临时目录）—— 默认值只在那里
        # 定义一次，因为它才是**真正拼路径**的地方。以前这里另存一份，结果 cli 的
        # --dry-run 走的是 tiler 那份、落进了当前工作目录。
        tile_dir = str(out_dir)

    return asyncio.run(arun_paper(
        name, md, page_paths,
        jobs=jobs, max_rounds=max_rounds, subject=subject, model=model,
        call_llm=call_llm, on_event=on_event, tile_dir=tile_dir,
    ))
