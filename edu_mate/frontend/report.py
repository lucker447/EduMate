"""把 ``PaperResult`` 呈现成人能读的答案卷，以及定退出码。

纯展示，不调模型、不碰磁盘（``dump_json`` 只负责序列化，写不写文件由调用方定）。

**「未通过核验」必须醒目。** 这是整个设计里最容易做错的地方：核验没拦住错答案、
轮次用满后照常输出 —— 如果这时报告看上去和通过了的一模一样，前面的核验环节就白做了，
用户会把一个连模型自己都没把握的答案当成确认过的答案。所以未通过的题在标题里打标记，
底下把核验意见逐条列出来。

**「执行出错」和「未通过核验」分开说。** 前者是没跑成（去查接口或输入），后者是跑成了
但结论存疑（去看题）。排查方向完全不同，合并成一种"有问题"会把人指错方向。

**核验意见里，「阻断性的」和「不影响结论的」也要分开说。** 理由同上一层：前者是"这个答案
别直接用"，后者是"写得不够漂亮"。混在一张清单里，"第 3 步没把点积写出来"会和"选项读错了"
看起来一样严重，而前者根本不影响答案能不能用。
"""

__all__ = [
    "EXIT_ERROR",
    "EXIT_EXHAUSTED",
    "EXIT_OK",
    "dump_json",
    "render_answers_md",
    "status_code",
]

EXIT_OK = 0
"""全部通过。"""
EXIT_ERROR = 1
"""有硬失败：某题解答/核验没跑成（``record.error`` 非空）。"""
EXIT_EXHAUSTED = 2
"""没有硬失败，但有题轮次用满仍不通过 —— 结果可用，但要人看一眼。"""


def status_code(result) -> int:
    """退出码。优先级：硬失败 > 未通过核验 > 全部通过。"""
    if any(item.record.error for item in result.records):
        return EXIT_ERROR
    if any(not item.record.passed for item in result.records):
        return EXIT_EXHAUSTED
    return EXIT_OK


def dump_json(result, indent: int = 2) -> str:
    """``PaperResult`` → JSON 文本。"""
    return result.model_dump_json(indent=indent)


def _header(item) -> str:
    question, record = item.question, item.record
    title = "### %s. （%s）" % (question.number, question.qtype.label)
    if record.error:
        return title + " ⛔️ 执行出错"
    if record.passed:
        return title
    if record.exhausted:
        return title + " ⚠️ 未通过核验（已用满 %d 轮）" % record.rounds
    return title + " ⚠️ 未通过核验"


def _reason(record) -> str:
    if record.exhausted:
        return "轮次用满仍不通过"
    if record.verification is None:
        return "没跑成核验"
    blocking = len(record.verification.blocking_issues)
    if not blocking:
        # 走不到这里：passed 是由 blocking 推导的，没有 blocking 就必然通过。
        # 留着兜底是为了万一 report 被喂了手工构造的 record，别在这里炸掉。
        return "核验未通过"
    return "%d 处阻断性问题" % blocking


def _problems(item) -> list:
    """核验意见，位置固定在看答案之前。

    **阻断性的和不影响结论的分开印。** 混在一起列的话，"第 3 步没把点积写出来"和
    "选项读错了"看上去一样严重，而前者根本不影响这个答案能不能用 —— 这恰好是
    核验第一版空转三轮的原因。分开之后，看报告的人一眼能分出哪些必须处理。

    不影响结论的瑕疵在**通过**的题上也会打印。这是刻意的：它证明核验确实读过并挑出了
    东西，而不是走个形式就说通过。
    """
    record = item.record
    lines = []
    if record.error:
        lines.append("> **执行出错**：%s" % record.error)
    verification = record.verification
    if verification is None:
        return lines

    for index, issue in enumerate(verification.blocking_issues, 1):
        lines.append("> %d. **[%s]** %s" % (index, issue.kind, issue.detail))
        if issue.suggestion:
            lines.append(">    - 建议：%s" % issue.suggestion)

    minor = verification.minor_issues
    if minor:
        lines.append("> **不影响结论的瑕疵**（%d 处，不构成不通过）：" % len(minor))
        for issue in minor:
            lines.append("> - [%s] %s" % (issue.kind, issue.detail))

    if verification.independent_answer:
        # 摆在这里，是为了让"核验到底有没有独立发生过"能被人查 —— 只看结论的话，
        # 一段抄自待核验解答的"独立推理"和真的重算看起来一模一样。
        lines.append("> 独立核验答案：%s" % verification.independent_answer)
    return lines


def _body(item) -> list:
    answer = item.record.answer
    if answer is None:
        return ["", "（没有产出答案）"]
    lines = ["", "**答案**：%s" % answer.answer]
    if answer.steps:
        lines += ["", "**解答步骤**："]
        lines += ["%d. %s" % (i, step) for i, step in enumerate(answer.steps, 1)]
    if item.record.rounds > 1:
        lines += ["", "_（这是第 %d 轮的结果）_" % item.record.rounds]
    return lines


def render_answers_md(result) -> str:
    """整卷答案卷（Markdown）。"""
    items = list(result.records)
    passed = [item for item in items if item.record.passed]
    errored = [item for item in items if item.record.error]
    doubtful = [item for item in items if not item.record.passed and not item.record.error]

    meta = result.meta or {}
    lines = ["# %s —— 答案与核验" % result.name, ""]
    lines.append(
        "共 %d 题，通过核验 %d 题，未通过 %d 题。"
        % (len(items), len(passed), len(items) - len(passed))
    )
    if meta.get("model"):
        lines.append(
            "模型 `%s`；每题最多 %s 轮；用时 %ss。"
            % (meta.get("model"), meta.get("max_rounds"), meta.get("elapsed_seconds"))
        )

    if errored:
        lines += ["", "## ⛔️ 执行出错的题", ""]
        lines.append("这些题**没跑完**，要查接口或输入，不是答案有争议。")
        lines += [
            "- 第 %s 题：%s" % (item.question.number, item.record.error) for item in errored
        ]

    if result.warnings:
        lines += ["", "## 提醒", ""]
        lines += ["- %s" % warning for warning in result.warnings]

    if doubtful:
        lines += ["", "## ⚠️ 未通过核验的题", ""]
        lines.append("答案**照常给出**，但没有通过核验，请以核验意见为准。")
        lines += [
            "- 第 %s 题（%s）" % (item.question.number, _reason(item.record))
            for item in doubtful
        ]

    lines += ["", "---", "", "## 逐题"]
    for item in items:
        lines += ["", _header(item)]
        lines += _problems(item)
        lines += _body(item)

    return "\n".join(lines).rstrip() + "\n"
