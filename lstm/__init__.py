"""LSTM 模块：包含训练与推理可视化入口。

用法：
- 训练：python -m lstm.train
- 推理可视化：python -m lstm.infer_plot
"""
# 注意：避免在包初始化时导入子模块（train/infer_plot），
# 以免使用 `python -m lstm.train` 时出现 runpy RuntimeWarning。
__all__ = []