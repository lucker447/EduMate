"""能力层：确定性的、可离线验证的动作。

这一层**不是 agent**：没有多轮、没有决策、没有工具选择。输入给死，输出确定，
所以每一个模块都能在不调 API 的前提下测完（模型调用本身除外）。
"""

from edu_mate.core.service.jsonio import JsonParseError, extract_json_block, parse_into
from edu_mate.core.service.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    achat_text,
    achat_with_images,
    chat_text,
    chat_with_image,
    chat_with_images,
    get_async_client,
    get_client,
    image_to_data_url,
)
from edu_mate.core.service.normalize import normalize_markdown
from edu_mate.core.service.ocr import recognize_exam
from edu_mate.core.service.paper import recognize_paper
from edu_mate.core.service.parse import parse_paper
from edu_mate.core.service.storage import (
    KnowledgeBase,
    NullKnowledgeBase,
    NullPaperStore,
    PaperStore,
)
from edu_mate.core.service.tiler import (
    GRID_COLS,
    GRID_ROWS,
    decide_feed,
    decide_images,
    page_map_from,
    tile_paths,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "GRID_COLS",
    "GRID_ROWS",
    "JsonParseError",
    "KnowledgeBase",
    "NullKnowledgeBase",
    "NullPaperStore",
    "PaperStore",
    "achat_text",
    "achat_with_images",
    "chat_text",
    "chat_with_image",
    "chat_with_images",
    "decide_feed",
    "decide_images",
    "extract_json_block",
    "get_async_client",
    "get_client",
    "image_to_data_url",
    "normalize_markdown",
    "page_map_from",
    "parse_into",
    "parse_paper",
    "recognize_exam",
    "recognize_paper",
    "tile_paths",
]
