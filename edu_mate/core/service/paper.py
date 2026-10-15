"""整份试卷识别：多页图片 → 一份完整 Markdown。

分页**独立**识别再拼接，而不是把多页塞进同一次请求 —— 实测多图请求会把分辨率摊薄，
模型反而把整卷题号重排成 1..N、连真实的大题标题也丢掉。
"""

import re
from pathlib import Path

from edu_mate.core.service.normalize import ensure_notes, strip_notes
from edu_mate.core.service.ocr import recognize_exam

__all__ = ["recognize_paper"]

# 大题标题行首的大题序号（"一、" "二、" …），跨页去重要用
_SECTION_NUM_RE = re.compile(r"^##[ \t]*([一二三四五六七八九十]+)、")


def _drop_repeated_sections(text: str, seen: set) -> str:
    """丢掉"本页没印、模型自己补的"大题标题。

    续页上模型经常凭空补出本页根本没有的大题标题，实测必现，而且每次编的分值都不一样。
    一份卷子里真实的大题序号是唯一的，所以同一序号出现第二次，要么是模型补的，
    要么是该大题跨页重印 —— 两种情况都不该再输出一行标题。

    ``seen`` 跨页累积，就地更新。
    """
    kept = []
    for line in text.split("\n"):
        match = _SECTION_NUM_RE.match(line.lstrip())
        if match:
            num = match.group(1)
            if num in seen:
                continue
            seen.add(num)
        kept.append(line)
    return "\n".join(kept)


def recognize_paper(page_paths, model=None, on_page=None) -> str:
    """按顺序逐页识别，拼成一份完整的 Markdown。

    页面之间用空行分隔，每页内容前插入 ``<!-- 第 <文件名> 页 -->`` 标记：Markdown
    渲染时不可见，但出问题时能直接定位是哪一页识别坏的。标记里用的是文件名的 stem
    而不是序号，所以它指向的是**原始文件**，可回溯。

    单页失败不会中断整份试卷：该页位置留一行 ``<!-- 第 N 页：识别失败 -->``，后续页照常识别。

    各页自己的「识别说明」段落会被剪掉，整卷末尾只留一份：说明是**整卷**级别的，
    逐页拼起来会出现 8 份（实测第一版就是这样），而且模型自写的和兜底生成的格式还不统一。

    Args:
        page_paths: 已排好序的页面图片路径。排序是调用方的职责，本函数不重排。
        model: 覆盖默认模型；``None`` 表示用 ``llm.DEFAULT_MODEL``。
        on_page: 每页结束后的回调 ``on_page(index, total, path, error)``，
            ``index`` 从 1 开始，``error`` 为 ``None`` 表示该页成功。

    Returns:
        整份试卷的 Markdown；没有页面时返回空串。
    """
    pages = list(page_paths)
    total = len(pages)
    seen_sections = set()
    chunks = []

    for index, raw_path in enumerate(pages, 1):
        path = Path(raw_path)
        error = None
        try:
            text = recognize_exam(str(path), model=model)
        except Exception as exc:  # noqa: BLE001 —— 单页失败不该中断整份试卷
            error = exc
            text = ""

        if on_page is not None:
            on_page(index, total, path, error)

        if error is not None:
            chunks.append(
                "<!-- 第 %s 页：识别失败（%s） -->" % (path.stem, type(error).__name__)
            )
            continue

        body = strip_notes(text)
        body = _drop_repeated_sections(body, seen_sections).strip()
        chunks.append("<!-- 第 %s 页 -->\n\n%s" % (path.stem, body))

    return ensure_notes("\n\n".join(chunks).strip())
