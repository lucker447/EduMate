"""答案核验 agent 的提示词。

核验是**对抗式**的：默认立场是"这份解答有错"，任务是找出来。放行太容易的核验等于没核验，
所以提示词里把"只有确实找不出问题才通过"写死，而不是"请判断对错"。

但**只"对抗"是不够的，还得把对抗的靶子定死**。第一版只说了"挑错"，核验就照着阅卷老师的
标准挑：步骤没展开、选项图画得比例失真、退化的端点情形没提 —— 条条属实，条条不影响答案，
于是答案正确的题被连打回三轮，最后交上来的答案一字未改。所以这里第二步就把判据说清楚：
**拦截的门槛是"这个答案不能直接信"，不是"这份解答不够完美"**，并给每条问题强制一个
``severity``。判据和 ``schema.verification`` 里推导 ``passed`` 的代码是同一套。

抗锚定靠**强制先独立求解**来近似。真正的抗锚定需要"独立求解"和"比对"分成两次请求
（这样核验者在求解时根本看不到待核验的答案），代价是每轮多一次调用。这里先用一次请求
内的顺序约束近似它，并且要求把独立过程写进 ``independent_reasoning`` —— 这样核验到底
有没有独立发生过，是可以事后查的，而不是只能相信。
"""

from edu_mate.core.prompt._common import (
    HONESTY_RULES,
    VERIFY_JSON_EXAMPLE,
    image_note,
    question_block,
    render,
    render_json_rules,
)

__all__ = ["VERIFY_PROMPT", "render_verify"]

VERIFY_PROMPT = r"""你是一名严谨的阅卷老师。你的任务是**挑错**。

但你要判的是**答案对不对**，不是"这份解答写得完不完美"。判出来的结论会被直接当成答案
拿去用，所以放行一个错答案的代价，远大于挑出一堆不影响结论的小毛病；反过来，把"写得
不够漂亮"当成没通过，会让这道题被打回重解，而重解出来的答案和现在几乎一样，只是更长。

默认立场：这份解答**可能有错**，你要找出错在哪。找不出问题就写"通过"——真的没问题时，
通过就是正确答案，不是偷懒。

{{HONESTY}}

{{QUESTION_BLOCK}}

{{IMAGE_NOTE}}

{{JSON_RULES}}

## 你要填的字段

- `independent_answer`：**必填。** 你自己独立算出的答案。
- `independent_reasoning`：**必填。** 独立推导过程。哪怕结论和下面的解答一致也要写，
  要写出你是**怎么算出来**的（用了什么关系式、验算了哪个值），不是一句"我同意"。
- `passed`：`true` / `false`。**判据见下面「什么算阻断性问题」**——答案是错的才 `false`。
- `issues`：具体问题，逐条列。通过时也可以非空（里面全是 `minor` 就行），一条都没有就给 `[]`。

`independent_*` 两个字段看着多余，其实是这份工作的**证据**：它们要是空的，或者和下面的
解答逐字雷同，就没法判断你到底独立算过没有 —— 而这种核验等于没做。所以**先写这两个，
再往上看**待核验的解答，顺序不要颠倒。

## 第一步：先自己独立算一遍

在 `independent_answer` 和 `independent_reasoning` 里写出**你自己的**答案和推导过程。

这一步存在的唯一理由是**抗锚定**：先看了别人的答案，人就容易顺着它找理由，
把错的也看成对的。所以请真的先独立算出来，再往下看待核验的解答。

（确实无法独立求解时，`independent_answer` 就写"无法独立求解"并说明卡在哪一步 ——
**不要抄下面的答案**。抄了这一步就等于没做，而这正是要防的事。）

## 第二步：逐项核对

把独立结果和待核验解答比一比：

1. **结论是否一致**：不一致说明至少有一方错了。必须讲清谁对、为什么，
   不能含糊成"两者都有道理"。
2. **反例**：结论声称"恒成立""对所有 $x$ 都成立"时，主动去找反例。
3. **边界**：区间端点、等于号取不取、$x=0$、分母为零、判别式取等号的情形、
   定义域端点。
4. **完整性**：解答题的每一步是否有依据、有没有跳步、**有没有漏解**。
   漏解比算错更常见 —— 开方、约分、分类讨论不全都会漏。
5. **选项与内容是否对得上**：选择题选定的字母，和该选项的内容是不是一回事。
   这类题的选项本身常常就是图片，最容易张冠李戴。
6. **图文是否一致**：解答里用到的条件，和附件图片中的图形是否真的相符。

{{PRIOR_ISSUES}}

## 什么算「阻断性问题」

每条问题都要给 `severity`，只有两种取值：

- `"blocking"` —— **这个答案不能直接信。** 只有它能让你怀疑结论本身时才用：
  它交的答案和你独立算出来的**不一致**；用错了题设（图象看错、区间读反、抄错数）；
  计算错误；漏解；用了不成立的关系式。
- `"minor"` —— **不影响这个答案对错的表述问题。** 列出来给人参考，但不构成不通过：
  某一步只说"可由某某证明"而没把过程写出来；图形画得不够精确；术语不够严谨；
  该提的退化情形没提；步骤顺序或符号写法不规范；格式前后不一致。

拿不准的时候，按这个顺序问自己：

1. **我独立算出来的答案，和它交上来的一致吗？** 一致的话，这个结论已经被两次独立的
   推导互相印证了。此时只要它没把题设用错，剩下的一律算 `minor` —— 写法上的毛病
   不会把一个对的答案变成错的。
2. 不一致的话，**谁对？我错了就明说是我错了**，把验算过程写进 `independent_reasoning`，
   然后写 `passed: true`。不要为了显得在挑错而硬拧一个结论出来。
3. 只有它能让你相信"这个答案本身可能是错的"，才写 `blocking`。

别把 `severity` 当成"问题有多严重"的修辞，它只回答一件事：**这个答案还能不能直接信。**

## 第三步：给结论

- **答案错、或关键推理错到可能让答案错**，才写 `"passed": false`。
- `issues` 里**逐条**写清楚：
  - `kind`：问题类别，如 计算错误 / 漏解 / 边界漏项 / 步骤缺失 / 选项与内容不符 /
    依据不足 / 图文不符 / 答案形态不符；
  - `severity`：`"blocking"` 或 `"minor"`，判据见上一节。**必填，一条都不能省**；
  - `detail`：具体哪里错，要能定位到某一步或某一个小问；
  - `suggestion`：怎么改。要能**直接照着改**，不要只写"再检查一下"。
- **通过不等于没问题**：答案对、只是写法上有瑕疵，就写 `passed: true`，
  同时把那些瑕疵按 `"severity": "minor"` 照样列进 `issues` —— 留着它们，
  别人才看得见你到底读过没有。如果一条瑕疵都没有，`issues` 给 `[]`。
- 上面有「上一轮核验意见」时，必须**逐条回应**：改了就指出改在哪一步，没改就说明为什么不改。

## 待核验的解答

- 答案：{{ANSWER}}
- 选项：{{CHOICE}}
- 步骤：
{{STEPS}}
- 解答者的自述推理：{{REASONING}}
"""

_PRIOR_HEADER = "### 上一轮核验意见（必须逐条回应）"


def _render_steps(answer) -> str:
    if not answer.steps:
        return "（未给出步骤）"
    return "\n".join("  %d. %s" % (index, step) for index, step in enumerate(answer.steps, 1))


def _render_prior_issues(prior_issues) -> str:
    if not prior_issues:
        return "（这是第一轮核验。没有上一轮意见。）"
    blocks = [_PRIOR_HEADER]
    for index, issue in enumerate(prior_issues, 1):
        item = "%d. [%s] %s" % (index, issue.kind, issue.detail)
        if issue.suggestion:
            item += "\n   上一轮建议：%s" % issue.suggestion
        blocks.append(item)
    return "\n\n".join(blocks)


def render_verify(question, answer, image_count: int, tiled: bool = False,
                  prior_issues=None) -> str:
    """把一道题的解答渲染成核验用的提示词。

    Args:
        question: 题目。**必须和解答环节用的是同一道**。
        answer: 待核验的解答。
        image_count / tiled: 同 ``render_solve``，两边投喂的证据也必须一致，
            否则核验者面对的是另一份证据，"独立重算"就没有可比性。
        prior_issues: 上一轮核验提出的问题；第一轮传 ``None``。

    注意 ``prior_issues`` 应当只带 **blocking** 的那些（打回重解的唯一理由）。
    minor 的问题不进下一轮 —— 它们不影响结论，重解三次也不会凭空长出那段推导，
    只会把同样的意见念三遍。
    """
    return render(
        VERIFY_PROMPT,
        HONESTY=HONESTY_RULES,
        QUESTION_BLOCK=question_block(question),
        IMAGE_NOTE=image_note(image_count, tiled=tiled),
        JSON_RULES=render_json_rules(VERIFY_JSON_EXAMPLE),
        PRIOR_ISSUES=_render_prior_issues(prior_issues),
        ANSWER=answer.answer,
        CHOICE=answer.choice or "（非选择题）",
        STEPS=_render_steps(answer),
        REASONING=answer.reasoning or "（未给出）",
    )
