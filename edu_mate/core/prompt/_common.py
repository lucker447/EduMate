"""提示词的公共片段，以及渲染工具。

**模板一律用哨兵 ``.replace()`` 渲染，绝不用 ``str.format`` / f-string / ``string.Template``。**
题目里全是 LaTeX，``{}`` 和 ``$`` 遍地都是：``.format`` 会把 ``\\frac{1}{2}`` 里的花括号
当成占位符直接报 KeyError，``Template`` 会把 ``$`` 当变量前缀。这是一个已经踩过的坑，
不是风格偏好。
"""

import re

__all__ = [
    "ANSWER_JSON_EXAMPLE",
    "HONESTY_RULES",
    "JSON_RULES",
    "REPAIR_PROMPT",
    "VERIFY_JSON_EXAMPLE",
    "image_note",
    "options_block",
    "question_block",
    "render",
    "render_json_rules",
    "render_repair",
    "sentinel",
]

# 占位符写成 {{名字}}，形如 {{QUESTION}}。用双花括号是为了和 LaTeX 的单花括号区分开，
# 万一哪天有人不小心对模板跑了 .format，双花括号会被原样保留而不报错 —— 出问题也出得温和。
_SENTINEL_RE = re.compile(r"\{\{([A-Z_]+)\}\}")


def sentinel(name: str) -> str:
    """``sentinel("QUESTION")`` → ``"{{QUESTION}}"``。"""
    return "{{%s}}" % name


def render(template: str, **values) -> str:
    """把 ``{{NAME}}`` 占位符替换掉。

    没给值的占位符会**原样留着**，而不是替换成空串 —— 宁可让提示词里露出一行
    ``{{CHOICE}}`` 被人看见，也不要静默地少喂一段要求给模型。
    """
    def swap(match):
        key = match.group(1)
        if key in values:
            return "" if values[key] is None else str(values[key])
        return match.group(0)

    return _SENTINEL_RE.sub(swap, template)


# 所有提示词共用。这段是本轮实测结论的直接产物，改它之前先想清楚为什么要改。
HONESTY_RULES = r"""## 关于"看不见的东西"，这是硬要求

附带的图片是**底稿**，题目文字是**转写**。转写会出错，图片不会。两者冲突时以图片为准。

- **图文冲突以图为准。** 题号、数字、选项内容都可能被认错。发现冲突要明确指出冲突在哪，
  并按图片作答。
- **看不清就说看不清。** 图太小、太糊、被裁掉一半时，明确写"无法确认"，并说清是哪一部分。
- **禁止猜测填充。** 绝不要写"虽然我们看不到图，但根据题意……"然后继续作答。
  实测里模型正是这样凭先验猜出了一个答案，而且**猜的时候和真读到图时一样自信** ——
  这是最危险的一种错。看不到就说看不到，触发人工复核，比给一个自信的错答案好得多。
- **不确定的边界情况要摆出来。** 比如"若 $x>0$ 则…，若 $x<0$ 则…"，宁可两条都写。"""


JSON_RULES = r"""## 输出格式

只输出**一个** ```json 代码块，块内是一个 JSON 对象。不要有任何解释文字、开场白或结束语。

- 字段名必须和上面要求的完全一致，不要改名、不要加字段。**上面列出的字段一个都不能省**，
  包括你认为"没什么可写"的那些 —— 省掉的字段在程序看来就是"你没做这一步"。
- 字符串里的换行写成 `\n`。
- **反斜杠必须写成两个** —— 这是最容易出错的一条。JSON 会把 `\f` `\t` `\b` 当转义字符吃掉，
  所以 LaTeX 命令一律写成 `\\frac`、`\\times`、`\\sqrt` 这种形式。少写一个反斜杠，
  你写的公式就会变成乱码。
- 数学公式保持 LaTeX。行内用 `$...$`。

正确的样子（注意 `\\frac` 前面是两个反斜杠）：

```json
{{JSON_EXAMPLE}}
```"""

# 示例必须**按调用方的形状**给。以前两边共用一份 Answer 形状的示例，结果是核验环节
# 照着它输出一个只有 `passed` 的对象 —— 连 `independent_answer` 都没写，抗锚定的证据
# 全空，而核验却报了"通过"。示例的形状比"记得填哪些字段"这句话管用得多。
ANSWER_JSON_EXAMPLE = (
    '{"choice": "A", "steps": ["配方得 $y=(x-1)^2-4$", '
    '"故最小值是 $\\\\frac{-4}{1}=-4$"], '
    '"answer": "顶点在 $(1,-4)$、开口向 A 图，故选 A", '
    '"reasoning": "对照四个选项的顶点位置即可排除其余三项", '
    '"confidence": 0.9}'
)

# 这个例子**必须示范"答案对、只是有瑕疵"**：`passed: true` 配一条 `minor`。
#
# 换过两版，两版都被实测打回来，所以别再凭直觉改：
# - 第一版和解答共用一份 Answer 形状的示例 → 核验照着它输出一个只有 passed 的对象，
#   `independent_*` 全空，抗锚定的证据没了，它却报了"通过"。
# - 第二版故意用 `passed: false` 的例子 → 矫枉过正：核验开始拿阅卷老师的标准挑毛病
#   （步骤没展开、图的比例不对），两道答案本来正确的题被连打回三轮，答案一字未改。
#
# 示例的形状比"记得填哪些字段"这句话管用得多 —— 换的例子直接改变了模型的行为，
# 两次都是。所以这里选的是**判据本身**的样子：列了问题，但问题不影响结论，所以通过。
VERIFY_JSON_EXAMPLE = (
    '{"passed": true, "independent_answer": "B", '
    '"independent_reasoning": "我自己代 $x=1$ 算得 $y=-1$，顶点在 $x=1$，与 B 的图一致；'
    'A 的顶点在 $x<0$，排除", '
    '"issues": [{"kind": "步骤缺失", "severity": "minor", '
    '"detail": "第 3 步只说了可用向量点积证明，没有把点积算出来", '
    '"suggestion": "补一句 $\\\\vec{EF}\\\\cdot\\\\vec{EH}=0$ 即可"}]}'
)


def render_json_rules(example: str) -> str:
    """把输出格式规则和**该环节的**示例拼起来。

    必须单独渲染一次，不能把 ``JSON_RULES`` 直接当值传进外层 ``render`` ——
    ``re.sub`` 不会回头再扫替换进去的文本，值里的 ``{{JSON_EXAMPLE}}`` 会原样漏进提示词。
    """
    return render(JSON_RULES, JSON_EXAMPLE=example)


# "一次修复重试"用的提示词。之所以敢不带图重发（见 ``llm.achat_text`` 的说明），
# 是因为这一步要修的是**格式**：模型手上有自己刚写的那段文字，重排成合法 JSON 不需要
# 再看题。带图重发既贵，又多引入一次"顺手改答案"的机会。
REPAIR_PROMPT = r"""你上一次的输出不是合法的 JSON，程序解析不了。

错误：{{ERROR}}

你上一次的输出：

--- 开始 ---
{{RAW}}
--- 结束 ---

请**只修格式**：把上面的内容重新输出成一个合法的 JSON 对象，字段名和取值一个都不要改。
不要重新解题，不要改答案，不要加任何解释文字。字符串里的反斜杠要写成两个
（`\frac` 要写成 `\\frac`、`\times` 要写成 `\\times`）。
"""


def render_repair(raw: str, hint: str) -> str:
    """渲染修复重试的提示词。``hint`` 来自 ``JsonParseError``，是写给模型看的一句话。"""
    return render(REPAIR_PROMPT, RAW=raw, ERROR=hint)


def options_block(question) -> str:
    """选择题的选项，**从结构化字段渲染**，不从 ``source_md`` 里捞。

    这一步不能省。``source_md`` 是**抽出选项之后**的题干切片，选项内容不在里面 ——
    全卷实测 10 道带选项的题里有 9 道如此。以前把 ``source_md`` 直接当"题目原文"喂过去，
    等于让模型在**看不到选项说什么**的情况下选 A/B/C/D，它只能去猜题图里的选项，
    而那恰好是实测里最容易翻车的形态。所以这里补上。

    选项内容是图时（``text`` 为空）显式说明，而不是留一行空白的 ``A.``：
    这样模型至少知道"一共四个选项，内容在附件里"，不会以为选项丢了而去自由发挥。
    """
    if not question.options:
        return ""
    lines = [
        "%s. %s" % (option.label, option.text.strip() or "（本选项的内容是图，见附件）")
        for option in question.options
    ]
    return "选项：\n\n%s\n\n" % "\n".join(lines)


def question_block(question) -> str:
    """题目的公共片段。

    「解答」和「核验」两个环节**必须用同一份**：两边看到的题目一旦不同，核验就没有意义
    —— 它在核对的不是同一道题。
    """
    return (
        "## 题目\n\n"
        "- 题号：第 %s 题\n"
        "- 题型：%s\n"
        "- 所属大题：%s\n"
        "- 原卷页码：%s\n\n"
        "题目原文（来自试卷转写）：\n\n"
        "--- 题目开始 ---\n%s\n--- 题目结束 ---\n\n"
        "%s"
        "文中 `<!-- 第 N 页 -->` 是页码标记，说明此处换页；`![图: …]` 表示此处有配图，"
        "方括号里是识别时的转述，**只用于定位，一切以附件图片为准**。"
        % (
            question.number,
            question.qtype.label,
            question.section or "（无）",
            "、".join(question.page_stems) or "未知",
            question.source_md,
            options_block(question),
        )
    )


def image_note(count: int, tiled: bool = False) -> str:
    """描述附件是什么。

    措辞会直接影响模型对不对得上号，所以"第几张是哪一块"必须说死，不能只写"若干张图"。
    顺序用"第 1 张是左上角"来表达，而不是罗列"左上、右上、中左…" —— 后者写死了网格形状，
    网格一改提示词就错。
    """
    if count <= 0:
        return "## 附件\n\n这道题没有配图，请只依据上面的题目文字作答。"
    if tiled:
        return (
            "## 附件\n\n"
            "附件是**同一页**扫描件切成的 %d 块，按**从左到右、从上到下**的顺序排列："
            "第 1 张是该页左上角那一块，最后一张是右下角那一块。\n\n"
            "**切开是有原因的**：这一页上的选项图很小，整页缩着看会糊成一片，模型会因此"
            "误以为「看不见图」从而凭先验作答。所以请**逐块真的看清楚**每张图里的图形形状"
            "再下结论，不要略过任何一块。" % count
        )
    return (
        "## 附件\n\n附件是原卷的整页扫描图，共 %d 张，按页码顺序排列。"
        "题干或配图看不清时，以图片为准。" % count
    )
