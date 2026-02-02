import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_performance_mode(deterministic: bool = False, benchmark: bool = True):
    """
    Configures PyTorch CuDNN settings for performance optimization.

    Disabling strict determinism and enabling benchmarking allows CuDNN to
    select the most efficient algorithms for the hardware, which is crucial
    for maximizing throughput on A100 GPUs.

    Args:
        deterministic (bool): If True, ensures reproducibility but may reduce performance.
                              If False, allows non-deterministic algorithms for speed.
                              Defaults to False.
        benchmark (bool): If True, enables the CuDNN auto-tuner to find the best
                          algorithm for the hardware configuration. Defaults to True.
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
