"""数据存储接口的**预留**：只定义形状，不接入任何实现。

需求文档第 5 条要求"预留之后开发需要的知识库和数据库存储接口，此次先略过数据存储"。
所以这个模块里**没有一行 IO** —— 只有 Protocol 和两个什么都不做的实现。它的价值在于把
"以后要存什么"这件事**现在**定下来：等真有数据库时，`PaperResult` 的形状已经稳定，
接进去只是写一个满足 Protocol 的类，不用回头改上层。

**用 Protocol 而不是 ABC。** 继承 ABC 意味着将来的 `SqlitePaperStore` 必须 import 这个
模块才能被认作存储 —— 而"能不能当存储用"本来就只该由**方法签名**决定。Protocol 是结构化
的：一个恰好有那些方法的类，不用知道这里存在过。这样新增后端（SQLite、Postgres、向量库）
不必碰这个文件，也不必碰上层。

**接口是 async 的。** 整条链路跑在 ``ainvoke`` 下、靠并发换速度，一个同步的远程调用会把
事件循环堵住，把并发全部抵消。将来若接的是本地 SQLite，实现里用 ``asyncio.to_thread``
包一层即可 —— 让实现去迁就接口，比让接口迁就某一种实现好。

**``Null*`` 是"没接存储"，不是"内存存储"。** 它**丢弃**写入而不是攒在内存里：攒着会让人
以为接好了，直到进程退出才发现什么都没剩下。丢弃配合一个安全的默认值，语义上是诚实的
——"现在还没有地方放"。
"""

from typing import Protocol, runtime_checkable

from edu_mate.core.schema import PaperResult

__all__ = [
    "KnowledgeBase",
    "NullKnowledgeBase",
    "NullPaperStore",
    "PaperStore",
]


@runtime_checkable
class PaperStore(Protocol):
    """整卷结果的持久化。落盘的那份 ``PaperResult`` JSON 就是这里的自然单位。

    ``key`` 由调用方给（一般用试卷名），**不由实现生成**：同一份卷子重跑时应当覆盖
    而不是堆积副本，所以"这份卷子叫什么"必须由上层决定，不能交给存储层编号。
    """

    async def save(self, key: str, result: PaperResult) -> None:
        """写入/覆盖一份整卷结果。"""
        ...

    async def load(self, key: str) -> PaperResult | None:
        """读回一份整卷结果；没有则 ``None``（**不抛异常** —— 查不到是正常情况）。"""
        ...

    async def keys(self) -> list:
        """已存的所有 key，用于列卷子。顺序不做保证。"""
        ...


@runtime_checkable
class KnowledgeBase(Protocol):
    """知识库检索。给**讲解 agent**用的：讲一道题时，去捞该知识点的教材原文、
    同类例题、常见错因，让讲解能"串联起相关的不同知识点"（需求文档对讲解 agent 的要求）。

    这里只有"存进去 / 查出来"两个动作，故意不做图查询、不做知识图谱 —— 那些是**实现**
    的选择，不是上层的需求。上层只需要知道"给我一句查询，还我几条相关文档"。
    """

    async def add(self, documents, *, source: str | None = None) -> int:
        """批量加入文档，返回加入的条数。``source`` 用于回溯这批文档从哪来。"""
        ...

    async def search(self, query: str, *, limit: int = 5) -> list:
        """按语义查相关文档，最相关的在前面。空知识库返回 ``[]``，不抛异常。"""
        ...


class NullPaperStore:
    """什么都不存的 ``PaperStore``：写入即丢弃，读回永远是空。

    给上层一个**可以无条件调用的默认值**，这样在存储接好之前，调用处不必写
    ``if store is not None:`` —— 那种判空会散落在每个调用点，等真接上时又得一个个删。
    """

    async def save(self, key: str, result: PaperResult) -> None:
        return None

    async def load(self, key: str) -> PaperResult | None:
        return None

    async def keys(self) -> list:
        return []


class NullKnowledgeBase:
    """什么都不查的 ``KnowledgeBase``：查不到任何东西。

    对讲解 agent 来说，"没有知识库"是有意义的正常状态（退化成只靠模型自己的知识），
    所以这里返回空列表而不是报错 —— 讲解照样跑得出来，只是少了外部依据。
    """

    async def add(self, documents, *, source: str | None = None) -> int:
        return 0

    async def search(self, query: str, *, limit: int = 5) -> list:
        return []
