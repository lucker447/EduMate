"""试卷识别业务能力：图片 → 模型识别 → 归一化输出。"""

from edu_mate.core.prompt import EXAM_OCR_PROMPT
from edu_mate.core.service.llm import chat_with_image
from edu_mate.core.service.normalize import normalize_markdown

__all__ = ["recognize_exam"]


def recognize_exam(image_path: str, model: str | None = None) -> str:
    """识别一张试卷图片，返回规范化的 Markdown。

    Args:
        image_path: 本地图片路径。
        model: 覆盖默认模型（默认取环境变量 DASHSCOPE_MODEL，否则 qwen-vl-ocr）。
    """
    raw = chat_with_image(image_path, EXAM_OCR_PROMPT, model=model)
    return normalize_markdown(raw)
