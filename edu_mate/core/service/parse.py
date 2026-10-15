"""整卷 Markdown → 题目列表。

**纯确定性解析，不调用任何模型。** 这一步能不能离线验证，直接决定了整条链路好不好调 ——
所以凡是能靠代码稳定判定的，都放在这里，不往提示词里塞。

真实产物里踩到的三个坑，都在这里处理：

1. **页码标记是"续题"不是"断题"。** 第 9 题的题干在第 1 页底部、四个选项图在第 2 页顶部，
   ``<!-- 第 2 页 -->`` 插在题目中间。所以标记只更新"当前在第几页"，不结束当前题目；
   题目结束时再看它落在哪几页，得出 ``page_stems = ["1", "2"]``。

2. **写在题目尾部的图可能是下一题的。** 第 13 题切片末尾挂着
   ``![图: 第14题配图, …]`` —— 这是原卷上"图在题目上方"的版面顺序造成的。
   不管它的话，第 13 题的提示词里会混进一段描述第 14 题抛物线的文字，是**主动的错信息**。
   这里按"尾部成组的配图行里，明确写了下一题题号的，还给下一题"来纠正。
   这个"第N题"的字样是模型自己加的（提示词只要求 ``![图: 描述]``），所以它只是
   **尽力而为的修正**：认不出来就按位置归属，也就是纠正前的行为，不会更糟。

3. **末尾的页码标记不属于当前题。** 第 16 题结束后才出现 ``<!-- 第 3 页 -->``，
   后面只有空行。标记跟着空行一起回缩掉，否则第 16 题会凭空多出一页。
"""

import re

from edu_mate.core.schema import Option, Question, QuestionType
from edu_mate.core.service.normalize import strip_notes

__all__ = ["parse_paper"]

# 成功的页码标记由 recognize_paper 插入，格式固定
_PAGE_MARK_RE = re.compile(r"^<!--\s*第\s*(.+?)\s*页\s*-->$")
# 识别失败的页面标记。冒号后半部分是失败原因，所以上面那条不会误match它
_PAGE_FAIL_RE = re.compile(r"^<!--\s*第\s*(.+?)\s*页[：:]识别失败")

# 大题标题。normalize 已统一成 "## "，这里只管取值
_SECTION_RE = re.compile(r"^##(?!#)[ \t]*(\S.*?)[ \t]*$")
# 题号行。normalize 会剥掉模型自造的伪序号，所以行首的数字就是原卷题号
_Q_START_RE = re.compile(r"^(\d{1,2})[.、．][ \t]*(\S.*)$")
# 选项行。内容可能为空 —— 选项是图时就是这样
_OPTION_RE = re.compile(r"^[ \t]*([A-Z])[.、．][ \t]*(.*)$")

# 配图占位符，由 OCR 提示词第七节强制输出
_FIG_RE = re.compile(r"^!\[图")
# 配图占位符里"第N题"的字样，用来把写错位置的图还给正确的题
_FIG_OWNER_RE = re.compile(r"^!\[图[：:]?[^\]]*?第\s*(\d+)\s*题")
# 图下方的"图 1" / "图2" 题注，跟着配图一起移动
_CAPTION_RE = re.compile(r"^图\s*\d+$")

_OPTION_LABELS = "ABCDEFGH"

# 大题标题里的题型关键词。判题型以它为准 —— 比"看题干里有没有下划线"可靠得多
_SECTION_TYPE_HINTS = (
    ("选择题", QuestionType.CHOICE),
    ("填空题", QuestionType.FILL),
    ("解答题", QuestionType.ANSWER),
)

# 填空下划线。只在没有大题标题兜底时才用它猜题型（提示词要求省略下划线，模型不一定照做）
_BLANK_MARK_RE = re.compile(r"_{2,}|＿{2,}")


def _section_type(section: str) -> QuestionType | None:
    """从大题标题判题型；判不出返回 ``None``，交给 ``_guess_type`` 兜底。"""
    for word, qtype in _SECTION_TYPE_HINTS:
        if word in section:
            return qtype
    return None


def _guess_type(body: str, options: list) -> QuestionType:
    """没有可用的大题标题时，靠形态猜题型。尽力而为。"""
    if options:
        return QuestionType.CHOICE
    if _BLANK_MARK_RE.search(body):
        return QuestionType.FILL
    return QuestionType.ANSWER


def _pick_option_run(hits: list) -> list:
    """从所有疑似选项行里，挑第一段「从 A 起、标号连续」的。

    要求从 A 起且连续，是为了不误伤题干里恰好以"大写字母 + 句号"开头的行
    （英文题干、字母编号的列举），也避免把某道小题内部的 A/B 当成整题的选项。
    至少两个才算一段。
    """
    runs = []
    run = []
    for item in hits:
        want = _OPTION_LABELS[len(run)] if len(run) < len(_OPTION_LABELS) else None
        if item[1] == want:
            run.append(item)
        else:
            if run:
                runs.append(run)
            run = [item] if item[1] == "A" else []
    if run:
        runs.append(run)

    for run in runs:
        if len(run) >= 2:
            return run
    return []


def _split_options(lines: list) -> tuple:
    """切出选项行。

    返回 ``(去掉选项后的正文行, [Option, …], 第一个选项行的下标或 None)``。
    下标用来定位"选项落在哪一页" —— 跨页题的选项可能和题干不同页。
    """
    hits = []
    for index, line in enumerate(lines):
        match = _OPTION_RE.match(line)
        if match:
            hits.append((index, match.group(1), match.group(2).strip()))

    run = _pick_option_run(hits)
    if not run:
        return lines, [], None

    taken = {index for index, _, _ in run}
    body = [line for index, line in enumerate(lines) if index not in taken]
    options = [Option(label=label, text=text) for _, label, text in run]
    return body, options, run[0][0]


def _looks_like_page_tail(line: str) -> bool:
    """这一行是不是"可以往后挪"的边缘行：空行、页码标记、配图、题注。"""
    stripped = line.strip()
    if not stripped:
        return True
    return bool(
        _PAGE_MARK_RE.match(stripped)
        or _FIG_RE.match(stripped)
        or _CAPTION_RE.match(stripped)
    )


def _split_figure_tail(lines: list, next_number: str) -> tuple:
    """把尾部属于**下一题**的配图还给下一题。

    只在"题目尾部连续的空行/配图/题注"里找，且必须明确写着下一题的题号。
    从最靠前的那张匹配的图开始，后面整段都跟着走 —— 图下面通常还跟着题注。
    """
    end = len(lines)
    while end > 0 and _looks_like_page_tail(lines[end - 1]):
        end -= 1

    for index in range(end, len(lines)):
        match = _FIG_OWNER_RE.match(lines[index].strip())
        if match and match.group(1) == next_number:
            return lines[:index], lines[index:]
    return lines, []


def _trim_tail(lines: list) -> list:
    """回缩掉尾部的空行与页码标记。

    第 16 题结束后才出现 ``<!-- 第 3 页 -->``，后面只有空行 —— 这个标记是**下一题**的
    开头，不是第 16 题的内容。不回缩的话第 16 题会凭空多出一页。
    """
    end = len(lines)
    # 只吃空行和页码标记 —— 配图和题注是题目的真内容，第 10 题的图就挂在题尾，
    # 一起吃掉就把图丢了。
    # 识别失败的标记同样是页边界，同样回缩：它属于"下一段内容之前"，不属于本题。
    while end > 0:
        stripped = lines[end - 1].strip()
        if not stripped or _PAGE_MARK_RE.match(stripped) or _PAGE_FAIL_RE.match(stripped):
            end -= 1
        else:
            break
    return lines[:end]


def _finish(pending: dict, order: int) -> Question:
    """把攒下的行整理成一道 ``Question``。"""
    lines = _trim_tail(pending["lines"])
    body_lines, options, first_option = _split_options(lines)

    # 选项所在页：第一个选项行之前最后出现的那条页码标记；没有就退回题干起始页
    option_page = pending["start_page"] or ""
    if first_option is not None:
        for line in lines[:first_option]:
            match = _PAGE_MARK_RE.match(line.strip())
            if match:
                option_page = match.group(1)

    stems = []
    if pending["start_page"]:
        stems.append(pending["start_page"])
    for line in body_lines:
        match = _PAGE_MARK_RE.match(line.strip())
        if match and match.group(1) not in stems:
            stems.append(match.group(1))

    body = "\n".join(body_lines).strip()
    qtype = pending["qtype"] or _guess_type(body, options)
    needs_figure = any(_FIG_RE.match(line.strip()) for line in lines)
    # 判据是**选项全部为空**，不要求同时有配图占位符：占位符丢了但选项确实是图的情况
    # 恰恰最需要切块投喂，这时候把 needs_figure 也算进条件，等于在最需要它的场合关掉它。
    has_image_options = (
        qtype is QuestionType.CHOICE
        and len(options) >= 2
        and all(not option.text for option in options)
    )

    head = "<!-- 第 %s 页 -->\n\n" % pending["start_page"] if pending["start_page"] else ""
    return Question(
        qid="q%d" % order,
        order=order,
        number=pending["number"],
        section=pending["section"],
        qtype=qtype,
        text=body,
        options=options,
        page_stems=stems,
        needs_figure=needs_figure,
        has_image_options=has_image_options,
        option_page=option_page if has_image_options else "",
        source_md=head + body,
    )


def parse_paper(md: str, warnings: list | None = None) -> list:
    """把整卷 Markdown 拆成题目列表。

    Args:
        md: ``recognize_paper()`` 的产物（或任何结构相同的 Markdown）。
        warnings: 可选列表；识别失败的页面会以人话追加进去，由调用方决定怎么呈现。

    Returns:
        按原卷顺序排列的 ``Question``，``order`` 从 1 开始且连续。
        解析不出任何题目时返回空列表 —— 这是**硬失败**，由调用方报错，不在这里抛异常。
    """
    lines = strip_notes(md or "").split("\n")

    collected = []
    section = ""
    qtype = None
    page = None
    pending = None

    for line in lines:
        match = _PAGE_MARK_RE.match(line.strip())
        if match:
            page = match.group(1)
            if pending is not None:
                pending["lines"].append(line)
            continue

        fail = _PAGE_FAIL_RE.match(line.strip())
        if fail:
            if warnings is not None:
                warnings.append("第 %s 页识别失败，该页题目可能有缺失" % fail.group(1))
            if pending is not None:
                pending["lines"].append(line)
            continue

        head = _SECTION_RE.match(line)
        if head:
            section = head.group(1).strip()
            qtype = _section_type(section)
            if pending is not None:
                collected.append(_finish(pending, len(collected) + 1))
                pending = None
            continue

        start = _Q_START_RE.match(line)
        if start:
            number = start.group(1)
            # 同一题号连着出现两次，是同一道题被页断开、下一页又印了一遍题号。
            # 合并而不是新建 —— 新建会得到两道同号题，而且第二道只有半截内容。
            if pending is not None and pending["number"] == number:
                pending["lines"].append(line)
                continue

            carry = []
            if pending is not None:
                pending["lines"], carry = _split_figure_tail(pending["lines"], number)
                collected.append(_finish(pending, len(collected) + 1))
            pending = {
                "number": number,
                "section": section,
                "qtype": qtype,
                "start_page": page,
                "lines": carry + [line],
            }
            continue

        # 卷首的试卷标题、姓名栏等：第一道题出现之前的内容一律不算题目
        if pending is not None:
            pending["lines"].append(line)

    if pending is not None:
        collected.append(_finish(pending, len(collected) + 1))

    return collected
