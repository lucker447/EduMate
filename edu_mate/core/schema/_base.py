"""schema 层的公共底座，以及若干「脏输入」强制转换。

模型返回的 JSON 不是我们自己生成的，字段类型经常飘：``passed`` 可能是
``true`` / ``"true"`` / ``"是"`` / ``"通过"``，``confidence`` 可能是 ``0.8`` 也可能是
``"high"``。这里做的转换必须是**确定性**的 —— 跟 ``service/normalize.py`` 一个思路：
能用代码稳定收拾的脏数据，就不要指望提示词去约束模型。

转换函数认不出来时返回 ``None``，由各字段自己的校验器决定是报错还是取默认 ——
把"认不出来"和"该取什么默认值"分开，是为了让默认值这件事在字段定义处一眼可见。
"""

from pydantic import BaseModel, ConfigDict

__all__ = ["Base", "to_bool", "to_float"]

_TRUE_WORDS = {"true", "yes", "y", "1", "是", "通过", "正确", "符合", "合格"}
_FALSE_WORDS = {"false", "no", "n", "0", "否", "不通过", "错误", "不符合", "不合格"}
_CONFIDENCE_WORDS = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
    "高": 0.9,
    "中": 0.6,
    "低": 0.3,
}


class Base(BaseModel):
    """schema 内所有模型的公共底座：多出来的字段一律忽略。

    下游要解析的是**模型输出**，它顺带多回一个 ``"note": "…"`` 不该让整道题判为失败。
    真正的结构性问题靠必填字段缺失来暴露，而不是靠多余字段。
    """

    model_config = ConfigDict(extra="ignore")


def to_bool(value):
    """把模型的各种"是/否"归一到 bool；认不出来返回 ``None``。"""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
    return None


def to_float(value):
    """把 ``0.8`` / ``"0.8"`` / ``"80%"`` / ``"high"`` 归一到 float；认不出来返回 ``None``。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[text]
        percent = text.endswith("%")
        if percent:
            text = text[:-1].strip()
        try:
            value = float(text)
        except ValueError:
            return None
        # "80%" 是 0.8 不是 80：这个字段整体是 0~1 的刻度，剥了百分号不除就是量纲错。
        return value / 100.0 if percent else value
    return None
