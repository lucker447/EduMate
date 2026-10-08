"""模型调用服务。

对上层屏蔽 base_url、鉴权、base64、MIME 这些细节，只暴露"图片 + 提示词 → 文本"一个动作。
"""

import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "chat_with_image", "get_client", "image_to_data_url"]

load_dotenv()

DEFAULT_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1"
)
DEFAULT_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-vl-ocr")

_client = None


def get_client() -> OpenAI:
    """惰性构造客户端。

    放在函数里而不是模块顶层，是为了让缺少 API key 时 import 本模块不会直接抛异常
    —— 只有真正发起调用时才检查。
    """
    global _client
    if _client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY，请在项目根目录的 .env 中配置")
        _client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    return _client


def image_to_data_url(image_path: str) -> str:
    """读本地图片转成 data URL。MIME 按扩展名推断，不要硬编码。"""
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def chat_with_image(image_path: str, prompt: str, model: str | None = None) -> str:
    """把一张图片和一段提示词发给多模态模型，返回文本结果。"""
    resp = get_client().chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )
    return resp.choices[0].message.content
