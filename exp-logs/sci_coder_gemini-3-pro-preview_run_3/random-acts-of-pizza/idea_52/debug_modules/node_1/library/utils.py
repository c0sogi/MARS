import os
import sys
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, PyTorch, and TensorFlow to ensure
    reproducibility across runs.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash seed (for dictionary iteration order, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in PyTorch
    # Note: This may impact performance but is necessary for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TensorFlow (if installed and used)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        # Prevent TensorFlow from allocating all GPU memory upfront
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before GPUs have been initialized
                pass
    except ImportError:
        pass


def get_device() -> torch.device:
    """
    Checks for GPU availability and returns the appropriate PyTorch device.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        msg = f"Device selected: CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        msg = "Device selected: CPU"

    # Log the device selection if logging is configured, else print
    if logging.getLogger().hasHandlers():
        logging.info(msg)
    else:
        print(msg)

    return device


def setup_logging(log_path: str = None, level=logging.INFO):
    """
    Configures the root logger to print to stdout and optionally to a file.

    Args:
        log_path (str, optional): Path to the log file. If None, logs only to stdout.
        level (int, optional): Logging level. Defaults to logging.INFO.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_path:
        # Ensure directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    # Reset existing handlers to prevent duplicate logs if called multiple times
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def log_metric(name: str, value: float):
    """
    Logs a metric with full precision as required by the task.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    logging.info(f"{name}: {value}")
