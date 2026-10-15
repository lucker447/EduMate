"""智能体编排层：「解答→核验」循环。

```
run_paper(...)  →  runner.py   对外唯一入口：识别（可跳过）→ 解析 → 跑图 → 汇总
                   graph.py    图的装配：外层 Send 扇出 + 每题子图
                   nodes.py    图上的节点，依赖从 Deps 注入
                   state.py    两层图各自的状态
```

题目在子图里跑：``solve → verify``，不通过就打回重解，最多 ``max_rounds`` 轮；用满仍不通过
就**照常输出并标注**，不静默丢弃。

模型调用全部走 ``Deps.call_llm`` 这一个注入点，所以整套循环能在不调一次 API 的前提下
跑完整 —— 假 LLM 可以按脚本吐错答案、坏 JSON、超时，把所有分支都过一遍。
"""

from edu_mate.core.agent.graph import build_graph, build_question_graph
from edu_mate.core.agent.nodes import Deps, make_call_llm
from edu_mate.core.agent.runner import NoQuestionsError, arun_paper, run_paper

__all__ = [
    "Deps",
    "NoQuestionsError",
    "arun_paper",
    "build_graph",
    "build_question_graph",
    "make_call_llm",
    "run_paper",
]
