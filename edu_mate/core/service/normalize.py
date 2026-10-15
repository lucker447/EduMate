"""识别结果归一化。

把模型输出清洗成规范的 Markdown。这里做的变换必须是**确定性**的 ——
凡是能用代码稳定解决的格式问题，就不要指望 OCR 模型去遵守提示词。
"""

import re

__all__ = ["ensure_notes", "normalize_markdown", "strip_notes"]

# \[ ... \]  →  $$ ... $$   （独立公式）
_BLOCK_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
# \( ... \)  →  $ ... $     （行内公式）
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# 三个及以上连续换行 → 一个空行
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# 模型无法确认的字符所用的占位符，与提示词第五节保持一致
_PLACEHOLDER = "〔?〕"

# 大题序号："一、" 或 "第X部分"，多处复用
_SECTION_MARK = r"(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+部分)"

# 大题标题行：以 _SECTION_MARK 开头且尚未是标题的整行
# 用 lookahead 只匹配插入点，替换时直接补 "## " 前缀即可
_SECTION_HEAD_RE = re.compile(
    r"^(?![ \t]*#)(?=[ \t]*" + _SECTION_MARK + r")",
    re.MULTILINE,
)

# 模型自造的连续序号。它一定紧跟真正的题号（"2. 10."）或大题序号（"3. 二、填空题"）；
# 后面不是这两者时就不是伪序号，而是原卷题号本身，必须原样保留。
_FAKE_INDEX_RE = re.compile(
    r"^(?P<head>#{1,6}[ \t]*)?[ \t]*\d{1,2}[.、][ \t]*"
    r"(?=(?:\d{1,2}[.、])|" + _SECTION_MARK + r")",
    re.MULTILINE,
)

# 页脚页码，如"第2页 共6页"
_PAGE_FOOTER_RE = re.compile(r"第\s*\d+\s*页\s*(?:/|共)\s*\d+\s*页")
# 页脚行都很短，超过这个长度就不当页脚，避免误删正文
_FOOTER_MAX_LEN = 30

# 任意层级的标题行
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]*(.+?)[ \t]*$", re.MULTILINE)
# 以题号开头的整行，如 "17. 已知…"
_QUESTION_START_RE = re.compile(r"^\d{1,2}[.、][ \t]*\S")
# 整行只有选项标号却没有任何内容，如 "A. B. C. D. E." —— 跨页残缺产生的噪声
_EMPTY_OPTION_ROW_RE = re.compile(
    r"^[ \t]*(?:\d{1,2}[.、][ \t]*)?(?:[A-Z][.、][ \t]*){2,}$"
)

# 「识别说明」段落的小标题行。放宽到 0–10 个字的尾巴，是因为模型有时写成
# "识别说明：无"；但仍要求整行基本只有这个标记，正文里顺带提到"识别说明"不算。
_NOTES_HEAD_RE = re.compile(
    r"^[#*\s]*(?:\*\*)?识别说明(?:\*\*)?[ \t]*[：:]?[ \t]*[^\n]{0,10}$"
)


def _strip_fence(text: str) -> str:
    """剥掉包裹全文的代码块外壳。

    模型经常无视"不要用代码块包裹"的要求，把整个结果塞进 ```markdown ... ```。
    只在首行是围栏、末行是围栏时才剥，避免误伤正文本身。
    """
    lines = text.split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _strip_fake_index(text: str) -> str:
    """剥掉模型自造的连续序号。

    续页题号不从 1 开始时（如本页从第 10 题接续），模型常两个都给：
    ``2. 10. 如图…``。前一个 ``2.`` 是它自己的计数器，既把真题号挤到第二位，
    又会连累紧随其后的大题标题（``3. 二、填空题`` 本该是 ``## 二、填空题``）。
    """
    return _FAKE_INDEX_RE.sub(lambda m: m.group("head") or "", text)


def _drop_empty_option_rows(text: str) -> str:
    """丢掉"只有选项标号、没有内容"的整行。

    典型形态 ``1. A. B. C. D. E.``：题目本体在上一页，本页只剩选项骨架，
    模型只好把光秃秃的标号排出来。这行没有任何信息量，留着反而像一道真题目。
    """
    return "\n".join(
        line for line in text.split("\n") if not _EMPTY_OPTION_ROW_RE.match(line)
    )


def _fix_headings(text: str) -> str:
    """修正标题：页脚不输出，题目不是标题，大题一律 ``##``。"""
    kept = []
    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        body = (match.group(2) if match else line).strip()

        # 页脚/页码冒充标题、或独自成行 → 整行丢掉
        if _PAGE_FOOTER_RE.search(body) and len(body) <= _FOOTER_MAX_LEN:
            continue

        if match is None:
            kept.append(line)
        elif _QUESTION_START_RE.match(body):
            # "# 17. 已知…" → "17. 已知…"：题目是正文，不是文档标题
            kept.append(body)
        elif re.match(_SECTION_MARK, body):
            # "# 二、填空题" → "## 二、填空题"：大题层级统一
            kept.append("## " + body)
        else:
            kept.append(line)
    return "\n".join(kept)


def strip_notes(text: str) -> str:
    """剪掉末尾的「识别说明」段落，只留正文。

    拼整份试卷时要先剪：`normalize_markdown` 是**逐页**跑的，一页一份说明，
    直接拼起来 8 页就有 8 份 —— 实测正是如此。剪掉后由 `recognize_paper`
    在整卷末尾统一调 `ensure_notes` 补一份。

    从**最后一个**标记行剪到末尾，这样模型写在正文中间的同名字样不会误伤后半篇。
    找不到标记行就原样返回。
    """
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if _NOTES_HEAD_RE.match(lines[i]):
            return "\n".join(lines[:i]).rstrip()
    return text


def ensure_notes(text: str) -> str:
    """保证输出末尾一定有「识别说明」，让下游可以无条件依赖这个段落。

    模型经常省略它。这里做确定性兜底：没有占位符就写"无"，有占位符就按行号
    列出来，方便人工回查。

    判据是"存在识别说明标记**行**"，而不是"文本里出现过这四个字"：模型把
    "识别说明"写进句子中间时，末尾仍然该有一份真正的说明段落。
    """
    if any(_NOTES_HEAD_RE.match(ln) for ln in text.split("\n")):
        return text

    lines = [i for i, ln in enumerate(text.split("\n"), 1) if _PLACEHOLDER in ln]
    if not lines:
        note = "## 识别说明\n无"
    else:
        where = "、".join("第 %d 行" % i for i in lines)
        note = "## 识别说明\n发现 %d 处无法确认的内容（%s），已用 %s 占位，请人工核对。" % (
            len(lines),
            where,
            _PLACEHOLDER,
        )
    return text.rstrip() + "\n\n" + note


def normalize_markdown(text: str) -> str:
    """清洗模型返回的 Markdown 文本。

    1. 剥掉包裹全文的代码块外壳
    2. 把 LaTeX 的 ``\\(...\\)`` / ``\\[...\\]`` 定界符统一成 ``$...$`` / ``$$...$$``
    3. 剥掉模型自造的连续序号（``2. 10.…`` → ``10.…``）
    4. 丢掉只有选项标号没有内容的空行
    5. 修正标题：页脚不输出、题目不是标题、大题一律 ``##``
    6. 把漏掉层级的大题标题（"一、…" / "第X部分…"）补成 ``##`` 二级标题
    7. 压缩多余空行
    8. 末尾补「识别说明」段落（模型没写的话）

    幂等：对已经规范的文本再调用一次不会有任何变化。

    第 3–6 步针对**续页**：一张图可能只是整卷的其中一页，题号不从 1 开始、
    页面上没有标题。这时模型容易自造序号、把题号升成标题、或抓页脚充当标题。
    这些都是能靠代码稳定判定的形态，不该指望提示词去约束。

    第 6 步是启发式的：只要整行以"一、""二、""第一部分"这类序号开头就当作大题标题。
    对数学、物理、化学等试卷足够可靠；文科学科里若有正文恰好以这类序号起头，会被误判，
    届时把 ``_SECTION_HEAD_RE`` 关掉即可。
    """
    if not text:
        return ""

    out = _strip_fence(text.strip())
    out = _BLOCK_MATH_RE.sub(lambda m: "$$" + m.group(1).strip() + "$$", out)
    out = _INLINE_MATH_RE.sub(lambda m: "$" + m.group(1).strip() + "$", out)
    out = _strip_fake_index(out)
    out = _drop_empty_option_rows(out)
    out = _fix_headings(out)
    out = _SECTION_HEAD_RE.sub("## ", out)
    out = _BLANK_LINES_RE.sub("\n\n", out)
    return ensure_notes(out.strip())
