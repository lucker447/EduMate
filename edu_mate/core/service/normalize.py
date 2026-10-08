"""识别结果归一化。

把模型输出清洗成规范的 Markdown。这里做的变换必须是**确定性**的 ——
凡是能用代码稳定解决的格式问题，就不要指望 OCR 模型去遵守提示词。
"""

import re

__all__ = ["normalize_markdown"]

# \[ ... \]  →  $$ ... $$   （独立公式）
_BLOCK_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
# \( ... \)  →  $ ... $     （行内公式）
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# 三个及以上连续换行 → 一个空行
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# 模型无法确认的字符所用的占位符，与提示词第五节保持一致
_PLACEHOLDER = "〔?〕"
# 大题标题行："一、" 或 "第X部分" 开头且尚未是标题的整行
# 用 lookahead 只匹配插入点，替换时直接补 "## " 前缀即可
_SECTION_HEAD_RE = re.compile(
    r"^(?![ \t]*#)(?=[ \t]*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+部分))",
    re.MULTILINE,
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


def _ensure_notes(text: str) -> str:
    """保证输出末尾一定有「识别说明」，让下游可以无条件依赖这个段落。

    模型经常省略它。这里做确定性兜底：没有占位符就写"无"，有占位符就按行号
    列出来，方便人工回查。
    """
    if "识别说明" in text:
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
    3. 把漏掉层级的大题标题（"一、…" / "第X部分…"）补成 ``##`` 二级标题
    4. 压缩多余空行
    5. 末尾补「识别说明」段落（模型没写的话）

    幂等：对已经规范的文本再调用一次不会有任何变化。

    第 3 步是启发式的：只要整行以"一、""二、""第一部分"这类序号开头就当作大题标题。
    对数学、物理、化学等试卷足够可靠；文科学科里若有正文恰好以这类序号起头，会被误判，
    届时把 ``_SECTION_HEAD_RE`` 关掉即可。
    """
    if not text:
        return ""

    out = _strip_fence(text.strip())
    out = _BLOCK_MATH_RE.sub(lambda m: "$$" + m.group(1).strip() + "$$", out)
    out = _INLINE_MATH_RE.sub(lambda m: "$" + m.group(1).strip() + "$", out)
    out = _SECTION_HEAD_RE.sub("## ", out)
    out = _BLANK_LINES_RE.sub("\n\n", out)
    return _ensure_notes(out.strip())
