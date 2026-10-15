"""模型调用服务。

对上层屏蔽 base_url、鉴权、base64、MIME、重试这些细节，只暴露"图片 + 提示词 → 文本"一个动作。

同步和异步两套入口：识别链路是逐页跑的，用同步就够；解答链路每题并行，必须走 ``ainvoke``
下的 ``async def`` 节点（实测同步分支不并行），所以那边用异步客户端。
"""

import asyncio
import base64
import mimetypes
import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "achat_text",
    "achat_with_images",
    "chat_text",
    "chat_with_image",
    "chat_with_images",
    "get_async_client",
    "get_client",
    "image_to_data_url",
]

load_dotenv()

DEFAULT_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://maas.qianwenaiapi.com/compatible-mode/v1"
)
# 不要换回 qwen-vl-ocr：它是 OCR 专用模型，只擅长"把字吐出来"，不按版面规则重组输出。
# 同一提示词下实测（试卷第 2、3 页），它把题号重排成 1..N（同一张图跑 5 次，原题号一次都没保住）、
# 凭空补出本页没有的大题标题、并且从不肯输出 `![图: ...]` 配图占位。
# qwen3-vl-plus 题号照抄准确、真实大题标题一字不差、配图占位也照给。
# 注意 qwen-vl-max-latest 在当前端点是 403 Access denied。
DEFAULT_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3-vl-plus")
# 识别是"照抄"任务，不是创作任务，采样温度必须压到 0。
# 不设的话走端点默认值（通常 1.0），实测同一张图同一提示词会输出 9453 或 1037 字符两种结果，
# 其中一种退化成 96 道重复题的死循环；设成 0 后两次输出逐字节相同。
DEFAULT_TEMPERATURE = float(os.getenv("DASHSCOPE_TEMPERATURE", "0"))

# 单次请求的超时。不设的话一道题卡住能把整卷拖到天亮。
DEFAULT_TIMEOUT = float(os.getenv("DASHSCOPE_TIMEOUT", "120"))

# 传输层重试：连不上、超时、限流、服务端 5xx 才重试。
# **这里数的是网络次数，和"核验打回重解"完全无关** —— 后者是语义重试，由 agent 层的
# 轮次上限管，两套计数不能混。
_RETRY_BACKOFF = (1.0, 2.0, 4.0)

_client = None
_async_client = None


def _require_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY，请在项目根目录的 .env 中配置")
    return api_key


def get_client() -> OpenAI:
    """惰性构造客户端。

    放在函数里而不是模块顶层，是为了让缺少 API key 时 import 本模块不会直接抛异常
    —— 只有真正发起调用时才检查。
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=_require_key(), base_url=DEFAULT_BASE_URL)
    return _client


def get_async_client() -> AsyncOpenAI:
    """惰性构造异步客户端。理由同 ``get_client``。"""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=_require_key(), base_url=DEFAULT_BASE_URL)
    return _async_client


def image_to_data_url(image_path: str) -> str:
    """读本地图片转成 data URL。MIME 按扩展名推断，不要硬编码。"""
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _is_retryable(exc: Exception) -> bool:
    """这次没成、下次可能成的错误。参数写错之类的 4xx 不在其列 —— 重试几次都一样。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _with_retry(call, sleep):
    """指数退避重试。``sleep`` 由调用方给，好让同一套逻辑同时服务同步和异步。"""
    last = None
    for delay in (0.0,) + _RETRY_BACKOFF:
        if delay:
            sleep(delay)
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 —— 判断在 _is_retryable 里
            if not _is_retryable(exc):
                raise
            last = exc
    raise last


def _build_content(image_paths, prompt: str) -> list:
    """拼请求体：图片按顺序排在文本**之前**。

    实测这个顺序下，模型能按提示词里"左上、右上、左下…"的描述逐块对应；
    顺序反过来时对应关系不稳。
    """
    content = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(str(path))}}
        for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    return content


def _require_images(image_paths) -> list:
    paths = [str(path) for path in image_paths]
    if not paths:
        raise ValueError("至少要有一张图；纯文本请求请用 chat_text / achat_text")
    return paths


def _kwargs(model, temperature, timeout, content) -> dict:
    """请求参数。``content`` 可以是字符串（纯文本）也可以是分片列表（带图）。"""
    return {
        "model": model or DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "timeout": DEFAULT_TIMEOUT if timeout is None else timeout,
        "messages": [{"role": "user", "content": content}],
    }


def chat_with_images(
    image_paths,
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> str:
    """把若干张图片和一段提示词发给多模态模型，返回文本结果。

    **不要在一次请求里塞来自不同页面的图。** 实测那样会把分辨率摊薄，模型还会顺手
    把整卷题号重排成 1..N。多图只用于**同一页切出来的块**（见 ``tiler``）。
    """
    paths = _require_images(image_paths)

    def call():
        response = get_client().chat.completions.create(
            **_kwargs(model, temperature, timeout, _build_content(paths, prompt))
        )
        return response.choices[0].message.content

    return _with_retry(call, time.sleep)


async def achat_with_images(
    image_paths,
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> str:
    """``chat_with_images`` 的异步版本。语义完全一致，只是退避时让出事件循环。

    并发不由这里管：限流是编排层的事（每题并行才需要），``service`` 只负责"发一次请求"。
    """
    paths = _require_images(image_paths)

    async def call():
        response = await get_async_client().chat.completions.create(
            **_kwargs(model, temperature, timeout, _build_content(paths, prompt))
        )
        return response.choices[0].message.content

    async def asleep(delay):
        await asyncio.sleep(delay)

    return await _aretry(call, asleep)


def chat_text(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> str:
    """纯文本请求，不带图。"""
    def call():
        response = get_client().chat.completions.create(
            **_kwargs(model, temperature, timeout, prompt)
        )
        return response.choices[0].message.content

    return _with_retry(call, time.sleep)


async def achat_text(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
) -> str:
    """``chat_text`` 的异步版本。

    存在的理由有两个，都是真实会遇到的：
    1. **题目一页都没对上**（md 是手写的，或页码标记和目录里的文件名不一致）时，
       仍然要能作答 —— 没有图是"证据变少"，不是"不能作答"。
    2. **JSON 修复重试**只需要重排格式，把整页图再传一遍是白花钱。
    """
    async def call():
        response = await get_async_client().chat.completions.create(
            **_kwargs(model, temperature, timeout, prompt)
        )
        return response.choices[0].message.content

    async def asleep(delay):
        await asyncio.sleep(delay)

    return await _aretry(call, asleep)


async def _aretry(call, sleep):
    """``_with_retry`` 的异步版。逻辑重复是有意的：同步版用 ``time.sleep``、
    异步版用 ``asyncio.sleep``，混用会把事件循环堵死。"""
    last = None
    for delay in (0.0,) + _RETRY_BACKOFF:
        if delay:
            await sleep(delay)
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc):
                raise
            last = exc
    raise last


def chat_with_image(
    image_path: str,
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """把一张图片和一段提示词发给多模态模型。``chat_with_images`` 的单图别名。"""
    return chat_with_images([image_path], prompt, model=model, temperature=temperature)
