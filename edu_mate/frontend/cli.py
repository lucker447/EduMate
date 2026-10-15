"""命令行入口：识别试卷图片，以及解答整卷。

frontend 层是面向用户的入口层，本模块是其中的命令行形态；以后若加 Web 界面，
应当新增 ``frontend/app.py`` 与本模块并列，而不是把逻辑堆在这里。

用法::

    # 识别（旧命令；不带 -o 时行为逐字节不变）
    python -m edu_mate.frontend.cli 1.png                      # 单张图 → 标准输出
    python -m edu_mate.frontend.cli 1.png -o output/           # 写成 output/1/paper.md
    python -m edu_mate.frontend.cli papers/ -o output/         # 一个目录 = 一份试卷
    python -m edu_mate.frontend.cli data/ -o output/           # data/ 下每份试卷各出一个目录
    python -m edu_mate.frontend.cli papers/ -o output/ --split # 退回"每张图各一份"

    # 解答（新子命令）
    python -m edu_mate.frontend.cli solve papers/ -o output/   # 解答整卷并核验
    python -m edu_mate.frontend.cli solve papers/ --dry-run    # 只解析，不调解答 API
    python -m edu_mate.frontend.cli solve --md output/<卷名>/paper.md --pages papers/

**一份试卷一个目录**：带 ``-o DIR`` 时，一份试卷的东西全落在 ``DIR/<试卷名>/`` 下，
目录里的文件名不再带试卷名前缀::

    output/2026-2027学年第一学期第6周限时作业/
    ├─ paper.md        识别稿
    ├─ answers.md      人读的答案卷
    ├─ solution.json   机器存档（PaperResult）
    └─ .tiles/         这份试卷的切块缓存

清一份试卷就是删一个目录，缓存也跟着走。``<试卷名>`` 的取法见 :func:`_paper_name`。

传目录时**一个目录算一份试卷**：目录里直接放着该试卷的分页图片，页序按文件名里的数字
自然排序（``2.png`` 排在 ``10.png`` 前面）。像 ``data/<试卷名>/1.png 2.png …`` 这种
结构，直接把 ``data/`` 整批跑掉即可。加 ``--split`` 可退回旧的"递归收集、每张图各一份"。

进度信息走 stderr，结果走 stdout，所以可以放心地 ``> result.md`` 重定向。

**两段式解析**：``main`` 先看 ``argv[0]`` 是不是 ``solve``，不是就把整个 argv 交给旧的
解析器。这样旧命令的参数一个都没变过 —— 加子命令最省事的写法（``add_subparsers``）
会把 ``paths`` 变成必填的位置参数挂在子命令下，旧命令的写法就全废了。
代价是：名叫 ``solve`` 的文件或目录再也不能直接当 PATH 传，得写 ``./solve``。
"""

import argparse
import re
import sys
import time
from pathlib import Path

from edu_mate.core.service import DEFAULT_MODEL, recognize_paper

__all__ = ["main"]

# 目录参数展开时认得的图片后缀
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _is_image(path):
    return path.suffix.lower() in IMAGE_SUFFIXES


def _natural_key(path):
    """按文件名里的数字排序，让 ``2.png`` 排在 ``10.png`` 前面。

    直接 ``sorted()`` 是字符串序，页码一上两位数就会错位。
    """
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def _pages_in(directory):
    """目录里直接放着的图片，按页码自然排序。"""
    return sorted(
        (f for f in directory.iterdir() if f.is_file() and _is_image(f)),
        key=_natural_key,
    )


def _collect_split(paths):
    """``--split`` 模式：递归收集图片，每张图各算一份。"""
    found = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(
                sorted(
                    (f for f in p.rglob("*") if f.is_file() and _is_image(f)),
                    key=_natural_key,
                )
            )
        else:
            found.append(p)
    return found


def _collect_papers(paths):
    """把命令行参数展开成"一份试卷 = 一个页面列表"。

    图片文件自身算一份（单页）；目录若直接含图片，它自己就是一份试卷，
    否则下钻，每个直接含图片的子目录各算一份。
    """
    papers = []
    for raw in paths:
        p = Path(raw)
        if not p.is_dir():
            papers.append((p.stem, [p]))
            continue

        pages = _pages_in(p)
        if pages:
            papers.append((p.name, pages))
            continue

        for child in sorted((c for c in p.iterdir() if c.is_dir()), key=_natural_key):
            pages = _pages_in(child)
            if pages:
                papers.append((child.name, pages))
    return papers


# 本工具写识别稿时用的固定文件名。子目录里不带试卷名前缀，靠这个常量把"这份 md 是不是
# 我们自己生成的"变成可判定的 —— 卷名推导要用它，见 _paper_name。
PAPER_MD_NAME = "paper.md"


def _paper_dir(out_dir, name):
    """一份试卷的目录 ``<DIR>/<试卷名>/``，顺手建出来。``out_dir`` 为 None 时返回 None。"""
    if out_dir is None:
        return None
    dest = Path(out_dir) / name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _paper_name(md_path, pages_arg=None):
    """复用 md 时这份试卷叫什么。

    识别稿现在固定叫 ``paper.md``，所以**不能**再拿 md 的文件名主干当卷名 —— 那样每份
    试卷都会叫 "paper"，子目录全撞成一个。改成优先往目录取名字：

    ==================  ==========================================
    md 叫 paper.md      用 md 的父目录名（本工具自己的布局，最权威）
    给了 --pages DIR     用 DIR 的目录名
    其余                 用 md 的文件名主干
    ==================  ==========================================

    ``paper.md`` 是本工具自己起的名字，看到它就知道名字该往目录取；别的名字是用户起的，
    就从文件名取。

    md 那一行排在 ``--pages`` **前面**，是为了「拿整卷的页图重跑几道题的夹具」这种用法：
    ``--md output/subset-x/paper.md --pages data/<整卷>/`` 应当落在 ``output/subset-x/``。
    要是按 ``--pages`` 取名，它会落进整卷的目录、把整卷那份结果覆盖掉。
    """
    md_path = Path(md_path)
    if md_path.name == PAPER_MD_NAME:
        # 裸写 "paper.md" 时父目录是 "."，它的 name 是空串，退回去取实际所在目录。
        return md_path.parent.name or md_path.resolve().parent.name
    if pages_arg:
        return Path(pages_arg).name
    return md_path.stem


def build_parser():
    parser = argparse.ArgumentParser(
        prog="edu_mate.frontend.cli",
        description="识别试卷图片，输出结构化 Markdown。传目录时一个目录算一份试卷。",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="试卷目录，或图片路径（可给多个）",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        metavar="DIR",
        help="把结果写成 <DIR>/<名称>/paper.md；不给则打印到标准输出",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="覆盖默认模型（默认 %s）" % DEFAULT_MODEL,
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="目录参数退回旧行为：递归收集图片，每张图各算一份（各出一个目录）",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="不打印逐页处理进度",
    )
    return parser


def build_solve_parser():
    parser = argparse.ArgumentParser(
        prog="edu_mate.frontend.cli solve",
        description="解答整卷并逐题核验：识别（或复用 md）→ 解析 → 每题跑「解答→核验」循环。",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="试卷目录或图片路径（可给多个）；用 --md 时不需要",
    )
    parser.add_argument(
        "-o", "--out-dir", metavar="DIR",
        help="输出到 <DIR>/<名称>/{answers.md,solution.json}；不给则打印到标准输出",
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="覆盖默认模型（默认 %s）" % DEFAULT_MODEL,
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3, metavar="N",
        help="每题「解答→核验」最多几轮；用满仍不通过就照常输出并标注（默认 3）",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, metavar="N",
        help="并发上限（默认 4）",
    )
    parser.add_argument(
        "--subject", default="中学", metavar="NAME",
        help="学科名，只影响提示词里的角色设定（默认 中学）",
    )
    parser.add_argument(
        "--md", default=None, metavar="FILE",
        help="复用已经识别好的 md，跳过识别这一步（省一次模型钱）；"
             "试卷名取 --pages 的目录名，md 叫 paper.md 时取它的父目录名",
    )
    parser.add_argument(
        "--pages", default=None, metavar="DIR",
        help="页图目录；配合 --md 使用，解答仍然要看原图",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只解析并估算请求数，不调解答 API",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="不打印逐题进度",
    )
    return parser


def _solve_targets(args):
    """展开成 ``[(名称, 页面路径列表, md 路径或 None)]``。

    ``--md`` 和 PATH 是两条互斥的路：给了 md 就不再需要识别，PATH 也就不必给了。
    """
    if args.md:
        md_path = Path(args.md)
        if not md_path.is_file():
            print("找不到 md：%s" % md_path, file=sys.stderr)
            return None
        pages = _pages_in(Path(args.pages)) if args.pages else []
        return [(_paper_name(md_path, args.pages), pages, md_path)]

    if not args.paths:
        print("要么给 PATH（试卷目录或图片），要么给 --md", file=sys.stderr)
        return None

    papers = _collect_papers(args.paths)
    return [(name, pages, None) for name, pages in papers]


def _print_event(printer, kind, **info):
    """把 runner 的事件翻成人话。未知事件不印 —— 宁可少一行，不要露出内部术语。"""
    if kind == "solve_start":
        printer("  第 %s 题 解答中（第 %d 轮，%d 张图）"
                % (info["question"].number, info["round"], info["images"]))
    elif kind == "verify_done":
        verdict = "通过" if info["passed"] else "不通过"
        detail = "（第 %d 轮" % info["round"]
        # 只有阻断性的才打回去，所以要把两类分开报 —— 一个"不通过"后面跟着 0 处阻断，
        # 人会以为程序坏了，其实是瑕疵不计入。
        if not info["passed"]:
            detail += "，%d 处阻断性问题" % (info.get("blocking") or 0)
        if info.get("minor"):
            detail += "，%d 处不影响结论的瑕疵" % info["minor"]
        printer("  第 %s 题 核验%s%s）"
                % (info["question"].number, verdict, detail))
    elif kind == "verify_override":
        # 模型自己写的结论和按 severity 推导出来的不一致。不是错误 —— 以推导为准，
        # 但要把这件事露出来：它反复出现就说明提示词里的判据没教到位。
        printer("  第 %s 题 核验自评%s，按 severity 判定为%s，以判定为准"
                % (info["question"].number,
                   "通过" if info["declared"] else "不通过",
                   "通过" if info["derived"] else "不通过"))
    elif kind == "repair":
        printer("  %s的输出不是合法 JSON，修复重试一次" % info["label"])
    elif kind == "solve_failed":
        printer("  第 %s 题 解答失败：%s" % (info["question"].number, info["error"]))
    elif kind == "verify_failed":
        printer("  第 %s 题 核验失败：%s" % (info["question"].number, info["error"]))


def _dry_run(name, pages, md_path, args, out_dir):
    """只解析，不调解答 API。打印题数、题型分布、预计请求数、哪些题要切块。

    **成本必须先有数**：二十几道题最坏要几百次请求，切块模式还会再翻。
    切块会真的落盘（确定性动作，不花钱），顺便把缓存热上。
    """
    from edu_mate.core.agent.runner import NoQuestionsError
    from edu_mate.core.schema import PageRef
    from edu_mate.core.service import parse_paper
    from edu_mate.core.service.tiler import decide_feed, page_map_from

    if md_path is not None:
        md = Path(md_path).read_text(encoding="utf-8")
    else:
        print("  （--dry-run 仍要先识别一次；想省这一步请先跑旧命令再带 --md）",
              file=sys.stderr)
        md = recognize_paper([str(p) for p in pages], model=args.model)

    warnings = []
    questions = parse_paper(md, warnings=warnings)
    if not questions:
        raise NoQuestionsError("从识别结果里解析不出任何题目")

    page_map = page_map_from(pages)
    dest = _paper_dir(out_dir, name)
    counts = {}
    tiled = []
    images_total = 0
    for question in questions:
        counts[question.qtype.label] = counts.get(question.qtype.label, 0) + 1
        images, is_tiled = decide_feed(question, page_map, out_dir=dest)
        images_total += len(images)
        if is_tiled:
            tiled.append((question.number, len(images)))

    # 最坏情况：每题 (N 轮解答 + 1 次修复) + N 轮核验。核验不修复（它没有修复重试）。
    per_question = 2 * args.max_rounds + 1
    print("\n=== %s（dry-run） ===" % name)
    print("题目数：%d" % len(questions))
    print("题型分布：%s" % "、".join("%s %d 题" % (k, v) for k, v in sorted(counts.items())))
    print("涉及页图：%d 页，%d 张（含切块）" % (len(pages), images_total))
    if tiled:
        print("需要切块的题（选项是图）：%s"
              % "、".join("第 %s 题 %d 块" % (n, c) for n, c in tiled))
    else:
        print("需要切块的题：无")
    print("预计请求数：%d 次（最坏 = %d 题 × %d 轮）" % (len(questions) * per_question,
                                                       len(questions), per_question))
    for warning in warnings:
        print("提醒：%s" % warning)
    return 0


def _run_solve(name, pages, md_path, args, out_dir):
    from edu_mate.core.agent.runner import run_paper
    from edu_mate.frontend.report import (
        dump_json,
        render_answers_md,
        status_code,
    )

    def printer(message):
        if not args.quiet:
            print(message, file=sys.stderr)

    def on_event(kind, **info):
        _print_event(printer, kind, **info)

    def on_page(index, total, path, error):
        if error is not None:
            print("  第 %d/%d 页识别失败：%s: %s"
                  % (index, total, type(error).__name__, error), file=sys.stderr)
        elif not args.quiet:
            print("  第 %d/%d 页识别完成 %s" % (index, total, path.name), file=sys.stderr)

    # 卷目录：这份试卷的所有落盘产物都在它下面。交给 runner 的是**它**而不是输出根，
    # 这样切块缓存才会落进 <卷目录>/.tiles/ 而不是所有试卷共用一个。
    dest = _paper_dir(out_dir, name)

    printer("=== %s ===" % name)
    if md_path is None:
        printer("识别中（共 %d 页）…" % len(pages))
    else:
        printer("复用 md：%s" % md_path)

    started = time.monotonic()
    result = run_paper(
        name, pages,
        md_path=md_path,
        out_dir=dest,
        jobs=args.jobs,
        max_rounds=args.max_rounds,
        subject=args.subject,
        model=args.model,
        on_event=on_event,
        on_page=on_page,
    )
    elapsed = time.monotonic() - started

    answers = render_answers_md(result)
    if dest:
        json_dest = dest / "solution.json"
        md_dest = dest / "answers.md"
        json_dest.write_text(dump_json(result), encoding="utf-8")
        md_dest.write_text(answers, encoding="utf-8")
        printer("完成（%.1fs）-> %s" % (elapsed, md_dest))
        printer("                  %s" % json_dest)
    else:
        print(answers)

    code = status_code(result)
    passed = sum(1 for item in result.records if item.record.passed)
    printer("共 %d 题，通过核验 %d 题，未通过 %d 题"
            % (len(result.records), passed, len(result.records) - passed))
    if code == 1:
        printer("有题目执行出错，见报告")
    elif code == 2:
        printer("有题目未通过核验，见报告")
    return code


def _solve_main(argv):
    args = build_solve_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_rounds < 1:
        print("--max-rounds 至少为 1", file=sys.stderr)
        return 1

    targets = _solve_targets(args)
    if targets is None:
        return 1
    if not targets:
        print("没有找到任何图片", file=sys.stderr)
        return 1

    from edu_mate.core.agent.runner import NoQuestionsError

    worst = 0
    for name, pages, md_path in targets:
        missing = [p for p in pages if not p.is_file()]
        for path in missing:
            print("跳过：文件不存在 %s" % path, file=sys.stderr)
        pages = [p for p in pages if p.is_file()]
        if not pages and md_path is None:
            print("%s：没有可用的页图，跳过" % name, file=sys.stderr)
            worst = _worse(worst, 1)
            continue

        try:
            if args.dry_run:
                code = _dry_run(name, pages, md_path, args, out_dir)
            else:
                code = _run_solve(name, pages, md_path, args, out_dir)
        except NoQuestionsError as exc:
            print("%s：%s" % (name, exc), file=sys.stderr)
            code = 1
        worst = _worse(worst, code)
    return worst


# 退出码的严重程度。**不能直接 max()**：数字上 2 > 1，但"未通过核验"（2）没有
# "执行出错"（1）急 —— 前者跑完了只是结论存疑，后者根本没跑成。整批跑的时候按这个
# 次序取最严重的那个，才不会因为最后一份是好的就把前面的硬失败盖掉。
_SEVERITY = {0: 0, 2: 1, 1: 2}


def _worse(a, b):
    return a if _SEVERITY.get(a, 2) >= _SEVERITY.get(b, 2) else b


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 两段式解析：只认第一个位置参数是不是 solve，其余原样交给旧解析器。
    if argv and argv[0] == "solve":
        return _solve_main(argv[1:])

    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        jobs = [(p.stem, [p]) for p in _collect_split(args.paths)]
    else:
        jobs = _collect_papers(args.paths)

    if not jobs:
        print("没有找到任何图片", file=sys.stderr)
        return 1

    total = len(jobs)
    failed = 0

    for index, (name, pages) in enumerate(jobs, 1):
        tag = "[%d/%d] %s" % (index, total, name)

        missing = [p for p in pages if not p.is_file()]
        if missing:
            for p in missing:
                print("%s 跳过：文件不存在 %s" % (tag, p), file=sys.stderr)
            failed += len(missing)
            pages = [p for p in pages if p.is_file()]
            if not pages:
                continue

        page_errors = []

        def report(i, n, path, error):
            if error is not None:
                page_errors.append(path)
                print(
                    "%s 第 %d/%d 页失败：%s: %s"
                    % (tag, i, n, type(error).__name__, error),
                    file=sys.stderr,
                )
            elif not args.quiet:
                print("%s 第 %d/%d 页完成 %s" % (tag, i, n, path.name), file=sys.stderr)

        if not args.quiet:
            print("%s 识别中（共 %d 页）…" % (tag, len(pages)), file=sys.stderr)

        started = time.monotonic()
        result = recognize_paper([str(p) for p in pages], model=args.model, on_page=report)
        elapsed = time.monotonic() - started

        failed += len(page_errors)

        if out_dir:
            dest = _paper_dir(out_dir, name) / PAPER_MD_NAME
            dest.write_text(result, encoding="utf-8")
            if not args.quiet:
                print("%s 完成（%.1fs）-> %s" % (tag, elapsed, dest), file=sys.stderr)
        else:
            if total > 1:
                print("\n===== %s =====" % name)
            print(result)

    if not args.quiet and total > 1:
        print("\n共 %d 份，失败 %d 页" % (total, failed), file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
