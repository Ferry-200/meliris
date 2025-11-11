"""LSTM 模块：包含训练与推理可视化入口。

用法：
- 训练：python -m lstm.train
- 推理可视化：python -m lstm.infer_plot
"""

from .train import main as train_main  # noqa: F401
from .infer_plot import main as visualize_main  # noqa: F401