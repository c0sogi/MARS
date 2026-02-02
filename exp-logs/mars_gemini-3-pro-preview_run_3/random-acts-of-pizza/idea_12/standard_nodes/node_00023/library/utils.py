import os
import random
import time
import numpy as np
import torch
from contextlib import contextmanager
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timer(name):
    """
    Context manager to measure and print execution time of a block.

    Args:
        name (str): A description of the code block being measured.
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"[{name}] done in {elapsed} s")
