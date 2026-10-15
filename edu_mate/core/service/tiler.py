"""按固定网格切块，供"小图密集"的页面使用，以及决定一道题该喂哪些图。

**为什么需要它**（第 9 题的实测）：第 2 页宽 1280 px，四个函数图象选项横排，
每个选项约占 240 px；整页投喂时模型还会再缩到约 1024 px 宽，每个选项只剩约 190 px。
这个尺度下模型会说"虽然我们看不到图"，然后**凭先验猜一个答案** —— 而且猜的时候
和真读到图时一样自信。把同一页切成 6 块、几块放在同一次请求里发出去，模型就能逐条
读出"B 的顶点压在 x 轴上""D 对称于 y 轴"这类只有真看到图才说得出的细节。

**为什么是固定网格**，而不是按内容定位：定位要么得问模型、要么得引入版面分析，
两者都可能失败；网格只依赖"页面尺寸"这一个确定的事实，失手了也只会是多切几块。

缓存落在 ``<out_dir>/.tiles/``。文件名里带源图的 mtime 和字节数，所以源图被重新导出时
会自然产生新的块，不会拿着旧缓存算出错误答案。代价是残留文件要手动清，对开发工具够用。
"""

import hashlib
import tempfile
from pathlib import Path

from PIL import Image

__all__ = [
    "GRID_COLS",
    "GRID_ROWS",
    "decide_feed",
    "decide_images",
    "default_tile_dir",
    "page_map_from",
    "tile_paths",
]

# 面向 1280×1700 左右的扫描页：2×3 得到约 640×567 的块，远高于上面那个失败区间。
GRID_COLS = 2
GRID_ROWS = 3

# 块宽度低于这个值才放大。定在 560 而不是更高，是因为**正常扫描页切出来的块就是约 640
# 宽**，阈值一旦高过它，每一页都会被无谓放大一倍，token 直接翻番。
# 触发它说明源页本身异常小（扫描档被缩过），那种情况下块才会成为新的瓶颈。
MIN_TILE_WIDTH = 560
MAX_UPSCALE = 4

TILE_DIR_NAME = ".tiles"


def page_map_from(page_paths) -> dict:
    """``[路径, …]`` → ``{文件名主干: 路径}``。

    键用主干而不是序号，因为它和 md 里的 ``<!-- 第 <主干> 页 -->`` 标记指向同一个东西，
    也就是**原始文件**，可以回溯。
    """
    return {Path(path).stem: str(path) for path in page_paths}


def default_tile_dir() -> Path:
    """不给 ``out_dir`` 时，切块缓存放哪儿：系统临时目录下的固定名。

    放在临时目录而不是工作目录，是因为**工作目录是用户的，不是我们的**。这里踩过一次：
    默认值原本是 ``Path.cwd()``，于是 ``solve --dry-run``（一个号称"不调 API、不改任何
    东西"的命令）在仓库根目录里丢下了一个 ``.tiles/``。给了 ``out_dir`` 就放输出目录
    里，清输出时缓存跟着走；没给就放这里，那本来就该被清。

    固定名（不带随机后缀）是为了跨次运行还能命中缓存，省下重复切块。文件名里带源图
    指纹，所以多个题共用一个目录也不会串。
    """
    return Path(tempfile.gettempdir()) / "edumate_tiles"


def _tile_dir(out_dir) -> Path:
    return Path(out_dir) / TILE_DIR_NAME if out_dir else default_tile_dir()


def _stamp(page: Path) -> str:
    """源图的指纹。改了源图，块的文件名就变，旧块自然失效。

    用 ``st_mtime_ns`` 而不是 ``st_mtime``：秒级精度下，同一秒内重新导出的页面会算出
    同一个指纹，于是拿着旧块去解新图 —— 而"改完立刻重跑"恰恰是最常发生的情形。
    再取哈希，是因为纳秒时间戳有 19 位，直接进文件名太长，不利于人肉排查。
    """
    stat = page.stat()
    raw = "%d_%d" % (stat.st_mtime_ns, stat.st_size)
    return hashlib.blake2b(raw.encode("ascii"), digest_size=4).hexdigest()


def _maybe_upscale(tile: Image.Image) -> Image.Image:
    width, height = tile.size
    if width >= MIN_TILE_WIDTH or width <= 0:
        return tile
    # 向上取整：MIN_TILE_WIDTH // width 会把"几乎够宽"的块留在阈值以下，
    # 放大完还是不够宽，白做一次。
    factor = min(MAX_UPSCALE, -(-MIN_TILE_WIDTH // width))
    if factor <= 1:
        return tile
    return tile.resize((width * factor, height * factor), Image.LANCZOS)


def tile_paths(page_path, out_dir=None, cols: int = GRID_COLS, rows: int = GRID_ROWS) -> list:
    """把一页切成 ``cols × rows`` 块，落盘缓存，返回**行优先**排列的路径。

    顺序很重要：调用方按这个顺序把块铺进请求，提示词里也按"左上、右上、左下…"描述，
    两边必须一致，否则模型会把块对错位置。
    """
    page = Path(page_path)
    target = _tile_dir(out_dir)
    stamp = _stamp(page)

    outputs = [
        target / ("%s__%s__r%dc%d.png" % (page.stem, stamp, row, col))
        for row in range(rows)
        for col in range(cols)
    ]
    if all(path.exists() for path in outputs):
        return [str(path) for path in outputs]

    target.mkdir(parents=True, exist_ok=True)
    with Image.open(page) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        for index, out in enumerate(outputs):
            if out.exists():
                continue
            row, col = divmod(index, cols)
            box = (
                col * width // cols,
                row * height // rows,
                (col + 1) * width // cols,
                (row + 1) * height // rows,
            )
            _maybe_upscale(image.crop(box)).save(out)

    return [str(path) for path in outputs]


def decide_feed(question, page_map: dict, out_dir=None):
    """决定这道题要喂哪些图，返回 ``(路径列表, tiled)``。

    ``tiled=True`` 表示这些图是**同一页切出来的块**。调用方必须把这个标志一路传到提示词里：
    说"附件是整页图"却其实传了块（或反过来），模型会按错误的预设去理解版面，比不告诉它还糟。

    - **选项是图**（``has_image_options``）→ 只发选项所在那页的切块。
      不发题干页：题干已经有文字了，再压一张整页图进去只会稀释注意力。
    - **其余** → 发题目涉及的每一页原图。整页投喂对"大图"是够用的（第 20 题那类
      占页面 1/4 以上的图，实测整页投喂全部正确），而且页图是**底稿**：
      Markdown 是转写，转写会错，图不会。

    页面没能对上 ``page_map``（比如 md 里的页码标记和目录里的文件名不一致）时跳过该页，
    由调用方去报告缺口 —— 这里不猜。这时**返回的 ``tiled`` 是 False**：退化成整页投喂
    时它就是整页，标志必须跟着实际投喂的内容走，不能跟着意图走。
    """
    if question.has_image_options:
        stem = question.option_page or (question.page_stems[-1] if question.page_stems else "")
        path = page_map.get(stem)
        if path:
            return tile_paths(path, out_dir=out_dir), True

    images = []
    for stem in question.page_stems:
        path = page_map.get(stem)
        if path and path not in images:
            images.append(path)
    return images, False


def decide_images(question, page_map: dict, out_dir=None) -> list:
    """``decide_feed`` 的只要路径版本。"""
    return decide_feed(question, page_map, out_dir=out_dir)[0]
