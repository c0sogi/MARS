import os
import time
import random
import numpy as np
import torch
import warnings
from library.config import Config


def set_seed(seed: int = Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, PyTorch, and TensorFlow.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_STATE.
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment for hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TensorFlow (safely import in case it's not being used, though installed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Prevent TF from allocating all GPU memory
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                pass
    except ImportError:
        pass


class Timer:
    """
    A context manager to measure and print the execution time of a code block.

    Usage:
        with Timer("Data Loading"):
            load_data()
    """

    def __init__(self, name: str = "Task"):
        """
        Args:
            name (str): A description of the task being timed.
        """
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[Timer] Starting: {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        # Printing with high precision as requested for metrics/timing
        print(f"[Timer] {self.name} finished in {elapsed_time:.6f}s")
