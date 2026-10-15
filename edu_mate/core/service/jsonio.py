"""从模型输出里抠出 JSON，并校验成 Pydantic 对象。

不用 ``response_format`` / function calling 的原因：端点是第三方代理，对这两者的支持度
未知，把可靠性押在上面不划算。这里走「提示词约定 + 确定性提取 + 校验」，每一层都能离线测。

**LaTeX 反斜杠是这个模块存在的主要理由。** 模型把 ``$\\frac{1}{2}$`` 写进 JSON 字符串时
通常只写一个反斜杠，而 ``\\f`` 在 JSON 里是合法转义（换页符）—— 于是 ``json.loads``
**不报错**，安安静静地把 ``\\frac`` 变成 ``\\x0c`` + ``rac``。公式就此损坏，而且没有任何
错误可以追查。

对付它的办法是**按领域语义重新划分**哪些转义可以信：``\\b`` ``\\f`` ``\\t`` 分别对应
退格、换页、制表符，这三种字符在试卷文本里永远不可能是本意，却正好撞上 ``\\beta``
``\\frac`` ``\\times`` ``\\theta`` 这几个高频命令 —— 所以一律当 LaTeX 处理，补成双反斜杠。
其余的 ``\\X`` 本来就会让 ``json.loads`` 报错，一并当 LaTeX 处理。

只有 ``\\n`` 必须保留：JSON 字符串里的真换行很常见。于是再拿一份**完整命令名**清单
把 ``\\nu`` ``\\neq`` ``\\nabla`` 从换行里挑出来 —— 只有整个命令名完全吻合才算数，
所以 ``\\nuclear`` ``\\nThe`` 依然是换行。残余风险记在 ``_N_COLLISIONS`` 旁边。

最后一关是控制字符检查：它不是主力，只是兜底 —— 万一模型真在字符串里吐出一个裸控制字符，
或者将来有人改动上面的规则漏了什么，这里会拦住，而不是让损坏悄悄流进答案。
"""

import json
import re

from pydantic import ValidationError

__all__ = ["JsonParseError", "extract_json_block", "parse_into"]

_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = {"}": "{", "]": "["}

# 认作"模型本意就是 JSON 转义"的字符。b/f/t 被有意排除（见模块说明），
# u 要单独看后面跟不跟 4 位十六进制，所以也不在这里。
_JSON_ESCAPES = '"\\/n'
_HEX4_RE = re.compile(r"[0-9a-fA-F]{4}")

# 和 \n 撞车的 LaTeX 命令。判据是**整个命令名完全吻合**，所以 "\nu" 是 \nu，
# 而 "\nuclear"、"The next line\n" 之后跟的任何英文词都仍然是换行。
#
# 已知残余：`\not` 故意不在表里 —— 英文里"换行 + not"太常见，为它牺牲真换行不划算。
# 漏掉的后果是文本里出现一个多余的换行加 "ot"，肉眼可辨，不是静默损坏。
_N_COLLISIONS = (
    "nabla",
    "ncong",
    "neq",
    "newline",
    "ngeq",
    "nleq",
    "nmid",
    "nolimits",
    "nonumber",
    "notin",
    "nparallel",
    "nsubseteq",
    "nu",
)


def _is_n_collision(raw: str, pos: int) -> bool:
    """``raw[pos]`` 是 ``n``。判断它到底开头的是一条 LaTeX 命令，还是纯粹的换行。"""
    word = re.match(r"[A-Za-z]+", raw[pos:pos + 12])
    return bool(word) and word.group(0) in _N_COLLISIONS

# 除换行 (\x0a) 外，所有 ASCII 控制字符。试卷文本里它们只可能是 JSON 转义出了岔子。
_SUSPICIOUS_RE = re.compile(r"[\x00-\x09\x0b-\x1f]")


class JsonParseError(ValueError):
    """模型输出变不成合法的 Pydantic 对象。

    ``hint`` 是写给**模型**看的一句话，供"一次修复重试"时告诉它错在哪。
    """

    def __init__(self, message: str, raw: str = "", hint: str = ""):
        super().__init__(message)
        self.raw = raw
        self.hint = hint or message


def _strip_outer_fence(text: str) -> str:
    """剥掉包裹全文的三反引号。只认首尾都成对的情况，避免误伤正文。"""
    lines = text.strip().split("\n")
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def extract_json_block(text: str) -> str:
    """从夹叙夹议的输出里抠出第一个完整的 JSON 对象/数组。

    按括号配对扫描，并且**认得字符串内部**：像 ``{"a": "}"}`` 这种值里带右花括号的，
    不这样扫就会提前截断。
    """
    if not text or not text.strip():
        raise JsonParseError("模型返回了空内容", text or "",
                             "你的上一条回复是空的。请只输出一个 JSON 对象。")

    body = _strip_outer_fence(text)
    start = body.find("{")
    if start < 0:
        start = body.find("[")
    if start < 0:
        raise JsonParseError("输出里找不到 JSON（没有 { 也没有 [）", text,
                             "你的上一条回复里没有 JSON。请只输出一个 JSON 对象，"
                             "不要有任何解释文字。")

    stack = []
    in_string = False
    escaped = False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(char)
        elif char in _CLOSERS:
            if not stack or stack[-1] != _CLOSERS[char]:
                break
            stack.pop()
            if not stack:
                return body[start:index + 1]

    raise JsonParseError("JSON 括号没有闭合", text,
                         "你的上一条回复里的 JSON 括号没有闭合。请重新输出完整、"
                         "闭合的 JSON 对象。")


def _escape_invalid_backslashes(raw: str) -> str:
    """把不是合法 JSON 转义的 ``\\X`` 补成 ``\\\\X``。

    只碰"本来就会让 ``json.loads`` 报错"的位置，加上 ``\\b`` ``\\f`` ``\\t`` 这三个
    "在试卷文本里不可能是本意"的位置；其余按 JSON 规则合法、且语义上确实需要的转义
    （``\\n`` ``\\"`` ``\\\\`` ``\\/`` ``\\uXXXX``）原样保留。
    """
    out = []
    index = 0
    end = len(raw)
    while index < end:
        char = raw[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue

        nxt = raw[index + 1] if index + 1 < end else ""
        if nxt == "u" and _HEX4_RE.fullmatch(raw[index + 2:index + 6] or ""):
            out.append(raw[index:index + 6])
            index += 6
            continue
        if nxt in _JSON_ESCAPES and not (nxt == "n" and _is_n_collision(raw, index + 1)):
            out.append(char + nxt)
            index += 2
            continue

        # 剩下的都是 LaTeX 命令：\sqrt \alpha \times \frac \beta … 补一个反斜杠
        out.append("\\\\")
        index += 1
    return "".join(out)


def _drop_trailing_commas(raw: str) -> str:
    """删掉对象/数组末尾多余的逗号。

    只删"逗号之后只剩空白、再跟 ``}`` 或 ``]``"的那些，并且跳过字符串内部 —— 模型写高亮
    引号或正文里带出 ``,}`` 这种字符组合不是不可能，正则一刀切会误伤。
    """
    out = []
    index = 0
    end = len(raw)
    in_string = False
    escaped = False
    while index < end:
        char = raw[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == ",":
            look = index + 1
            while look < end and raw[look] in " \t\r\n":
                look += 1
            if look < end and raw[look] in "}]":
                index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _first_control_char(value) -> str | None:
    """在解析结果里找第一个可疑控制字符；没有就返回 ``None``。"""
    if isinstance(value, str):
        match = _SUSPICIOUS_RE.search(value)
        return match.group(0) if match else None
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_control_char(key) or _first_control_char(item)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_control_char(item)
            if found:
                return found
    return None


def _brief(exc: ValidationError) -> str:
    """把 Pydantic 的报错压成一行，够定位就行。"""
    parts = []
    for error in exc.errors()[:3]:
        where = ".".join(str(p) for p in error["loc"]) or "(根)"
        parts.append("%s: %s" % (where, error["msg"]))
    return "；".join(parts)


def parse_into(model_cls, text: str):
    """把模型输出解析成 ``model_cls`` 实例。

    Raises:
        JsonParseError: 抠不出 JSON、语法错、含可疑控制字符，或字段校验不过。
            异常的 ``hint`` 可以直接塞进"修复重试"的提示词里。
    """
    block = extract_json_block(text)
    try:
        data = json.loads(_drop_trailing_commas(_escape_invalid_backslashes(block)))
    except json.JSONDecodeError as exc:
        raise JsonParseError(
            "JSON 语法错误：%s（第 %d 行第 %d 列）" % (exc.msg, exc.lineno, exc.colno),
            text,
            "你的上一条回复不是合法 JSON：%s。请只输出一个 JSON 对象，"
            "字符串里的反斜杠要写成两个，例如 LaTeX 的 \\frac 要写成 \\\\frac。" % exc.msg,
        ) from exc

    bad = _first_control_char(data)
    if bad is not None:
        raise JsonParseError(
            "输出里含未转义的 LaTeX 反斜杠（被当成 JSON 转义，变成了控制字符 U+%04X）"
            % ord(bad),
            text,
            "你的上一条回复里有未转义的反斜杠：JSON 把 \\f \\t \\b 这类当成转义字符吃掉了。"
            "请把字符串里所有 LaTeX 反斜杠都写成两个，例如 \\frac 写成 \\\\frac、"
            "\\times 写成 \\\\times。",
        )

    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise JsonParseError(
            "字段校验失败：%s" % _brief(exc), text,
            "你的上一条回复字段不合要求：%s。请按约定的字段名重新输出。" % _brief(exc),
        ) from exc
