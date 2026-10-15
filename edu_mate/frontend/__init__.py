"""面向用户的入口层。

当前只提供命令行入口 ``edu_mate.frontend.cli``；以后加 Web 界面时，在同一层新增模块即可。

注意：这里**不要** re-export ``cli`` 里的符号。``python -m edu_mate.frontend.cli`` 会先导入
本包，若本包已把 ``cli`` 导入过，runpy 执行它会触发 RuntimeWarning。需要 ``main`` 请直接从
``edu_mate.frontend.cli`` 导入。
"""
