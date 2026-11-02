"""Langfuse 观测层。

两套埋点方式，靠 ``EDUMATE_TRACE_MODE`` 切换（本分支默认 ``callback``）：

- ``manual``（A 方案）：节点里手工 ``with span("第 N 轮 · 解答")``，能标出轮次；
  但同步节点（``finalize``）跑在线程池里，拿不到异步上下文，埋不了点。
- ``callback``（B 方案）：把 ``langfuse.langchain.CallbackHandler`` 挂到 LangGraph 的
  ``config["callbacks"]`` 上，节点 span 由官方 handler 自动生成，入参是进入节点时的
  ``QuestionState``、出参是节点返回的增量，连 ``finalize`` 也有。
  代价是 span 名只有节点名（``solve`` / ``verify``），看不出第几轮 —— 轮次要进去看
  ``rounds`` 字段。

对外入口：``trace``（整卷根 span）、``span``（手工埋点，回调模式下退化为空操作）、
``callback_handler``（B 方案的 handler）、``flush``、``enabled``、``mode``。
没有配置、没装 SDK、或者 ``LANGFUSE_TRACING_ENABLED=false`` 时，全部退化成空操作
—— 调用点**不需要**判断开关，也不需要 import langfuse。

两条原则：

1. **观测层的异常永远不能进到业务里。** 建 span 失败就退化成一个空 span；但 ``yield``
   之后业务自己抛的异常原样向外传，不能被这里吞掉（否则节点失败会变成"看起来跑完了"）。
2. **试卷图片不进 trace。** 请求体里的图片是几百 KB 的 base64，这里统一替换成占位文本；
   同时 ``.env`` 里的 ``LANGFUSE_MEDIA_UPLOAD_ENABLED=false`` 会先一步关掉媒体上传
   —— SDK 的处理顺序是「媒体检测/上传 → mask」，光靠 mask 拦不住已经上传的图。
"""

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

__all__ = [
    "callback_handler",
    "enabled",
    "flush",
    "mode",
    "snapshot",
    "span",
    "trace",
]

_log = logging.getLogger(__name__)

# 探测结果缓存：None 表示还没探测过。
_available: bool | None = None

# 回调 handler 是否已经成功建起来：None 表示还没试过。
# 只有它明确为 True 才关掉手工 span —— handler 建不起来（没装 langchain、
# import 失败）时必须退回手工埋点，否则整棵 trace 会空掉。
_callback_ready: bool | None = None

_MODE_ENV = "EDUMATE_TRACE_MODE"

# data URL 前缀，例如 ``data:image/jpeg;base64,``。
_DATA_URI_RE = re.compile(r"^data:([\w/+.-]+);base64,")


class _NullSpan:
    """空 span。观测关闭时调用点仍然可以照常 ``.update(...)``。"""

    def update(self, **_kwargs) -> None:
        pass


_NULL_SPAN = _NullSpan()


def snapshot(value: Any) -> Any:
    """把 state（或任意 Pydantic 对象）递归转成可 JSON 序列化的结构。

    节点的 ``input`` 是**进入该节点时的完整状态**，``output`` 是节点返回的增量；
    两份合起来就是 LangGraph 那一步的状态变化。不这么做的话 trace 里只有手挑的几个
    字段，看不出 ``QuestionState`` 本身（尤其是上一轮的 ``verification``／``prior_issues``
    是怎么带进下一轮的）。

    失败一律降级成 ``repr`` —— 观测层不能因为一个序列化不了的对象把节点搞挂。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001
            return repr(value)
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [snapshot(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        # 状态里本来只放图片路径，不放二进制；真出现了也不能把它整个传上去。
        return "[%d 字节二进制已省略]" % len(value)
    return str(value)


def _mask(*, data: Any) -> Any:
    """把 base64 图片替换成占位文本。

    SDK 要求关键字参数 ``data=``（``MaskFunction`` 协议），并且返回值必须能 JSON 序列化。
    递归处理 dict / list，因为 OpenAI 请求体里的图片藏在
    ``{"type": "image_url", "image_url": {"url": "data:..."}}`` 这种嵌套结构里。
    """
    if isinstance(data, str):
        match = _DATA_URI_RE.match(data)
        if match:
            return "[图片已省略 mime=%s 约 %d KB]" % (match.group(1), len(data) // 1024)
        # 兜底：不以上面的前缀开头、但明显是内嵌的数据 URI。
        if len(data) > 1000 and ";base64," in data:
            return "[图片已省略 约 %d KB]" % (len(data) // 1024)
        return data
    if isinstance(data, dict):
        return {key: _mask(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(data=value) for value in data]
    return data


def _load() -> bool:
    """探测一次 Langfuse 是否可用，结果缓存。"""
    global _available
    if _available is not None:
        return _available

    _available = False
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return _available
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in ("false", "0"):
        return _available

    try:
        # mask 只能通过构造函数注册，环境变量里没有对应项。
        # 先构造一次，后续 get_client() 会沿用这份配置（含 mask）。
        from langfuse import Langfuse  # noqa: PLC0415

        Langfuse(mask=_mask, base_url=os.getenv("LANGFUSE_BASE_URL") or None)
    except Exception:  # noqa: BLE001 —— 观测不可用只是少一份数据，不能影响主流程
        _log.warning("Langfuse 初始化失败，观测已关闭", exc_info=True)
        return _available

    _available = True
    return _available


def enabled() -> bool:
    """当前是否真的在上报。调试 / 测试用，业务代码不需要调。"""
    return _load()


def mode() -> str:
    """当前的埋点方式：``manual``（A）或 ``callback``（B）。

    本分支默认 ``callback``；临时对比两套形状时设
    ``EDUMATE_TRACE_MODE=manual`` 即可切回手工 span。
    """
    return os.getenv(_MODE_ENV, "callback").strip().lower()


def _manual_disabled() -> bool:
    """回调 handler 接管时，手工 span 让位，避免同一节点出现两条重复的树。"""
    return mode() == "callback" and _callback_ready is True


def callback_handler():
    """建一个 Langfuse 的 LangChain/LangGraph 回调；不可用时返回 ``None``。

    返回 ``None`` 时会把 ``_callback_ready`` 置成 ``False``，``span()`` 继续走手工埋点
    —— 少一份自动状态，总比整棵树没有强。
    """
    global _callback_ready
    if mode() != "callback" or not _load():
        _callback_ready = False
        return None

    try:
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415
    except Exception:  # noqa: BLE001 —— 没装 langchain 元包时就是这里
        _log.warning("langfuse.langchain 不可用，退回手工埋点", exc_info=True)
        _callback_ready = False
        return None

    try:
        handler = CallbackHandler()
    except Exception:  # noqa: BLE001
        _log.warning("Langfuse CallbackHandler 构造失败，退回手工埋点", exc_info=True)
        _callback_ready = False
        return None

    _callback_ready = True
    return handler


def _open(kind: str, name: str, *, input=None, metadata=None, model=None):
    """建一个 context manager；建不出来返回 None。"""
    if not _load():
        return None
    try:
        from langfuse import get_client  # noqa: PLC0415

        kwargs = {
            "as_type": kind,
            "name": name,
            "input": input,
            "metadata": metadata,
        }
        if model:
            kwargs["model"] = model
        return get_client().start_as_current_observation(**kwargs)
    except Exception:  # noqa: BLE001
        _log.warning("创建 Langfuse span 失败，本条不记录", exc_info=True)
        return None


@contextmanager
def _enter(cm) -> Iterator[Any]:
    """手动进出 context manager：``__enter__`` 失败就退化成空 span，
    ``yield`` 之后业务抛的异常原样向外传。"""
    if cm is None:
        yield _NULL_SPAN
        return

    try:
        observation = cm.__enter__()
    except Exception:  # noqa: BLE001
        _log.warning("进入 Langfuse span 失败，本条不记录", exc_info=True)
        yield _NULL_SPAN
        return

    try:
        yield observation
    except BaseException as exc:  # noqa: BLE001 —— 只负责收尾，异常继续抛
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:  # noqa: BLE001
            _log.warning("关闭 Langfuse span 失败", exc_info=True)
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            _log.warning("关闭 Langfuse span 失败", exc_info=True)


@contextmanager
def span(name: str, *, input=None, metadata=None, _force: bool = False):
    """普通 span。没有观测时 yield 一个空 span。

    ``_force`` 只给 ``trace()`` 用：回调模式下根 span 仍然必须手工建，因为
    ``langfuse.openai`` 的 generation 靠 OTel 当前 span 找父节点，而实测在
    LangGraph 的异步节点里，handler 的 span 并不是模型调用时的当前 span；没有这层
    根 span，每次模型调用都会变成一条独立的 orphan trace。
    """
    if not _force and _manual_disabled():
        yield _NULL_SPAN
        return
    with _enter(_open("span", name, input=input, metadata=metadata)) as observation:
        yield observation


@contextmanager
def trace(
    name: str,
    *,
    session_id=None,
    tags=None,
    metadata=None,
    input=None,
):
    """整卷的根 span，同时设置 trace 级属性（名字 / session / 标签 / 元数据）。

    ``propagate_attributes`` 要在根 span **之前**进入，属性才会落到根 span 上。
    """
    attributes = None
    if _load():
        try:
            from langfuse import propagate_attributes  # noqa: PLC0415

            attributes = propagate_attributes(
                trace_name=name,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
            )
            attributes.__enter__()
        except Exception:  # noqa: BLE001
            _log.warning("设置 Langfuse trace 属性失败", exc_info=True)
            attributes = None

    try:
        with span(name, input=input, metadata=metadata, _force=True) as root:
            yield root
    finally:
        if attributes is not None:
            try:
                attributes.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                _log.warning("退出 Langfuse trace 属性失败", exc_info=True)


def flush() -> None:
    """进程结束前把缓冲里的数据推出去。短生命周期程序必须调，否则丢最后一批。"""
    if not _load():
        return
    try:
        from langfuse import get_client  # noqa: PLC0415

        get_client().flush()
    except Exception:  # noqa: BLE001
        _log.warning("Langfuse flush 失败", exc_info=True)
