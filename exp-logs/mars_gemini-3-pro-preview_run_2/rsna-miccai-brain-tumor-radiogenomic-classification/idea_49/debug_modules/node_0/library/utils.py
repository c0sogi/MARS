import os
import sys
import logging
from library.config import Config, set_seed


def get_logger(name="ModelPipeline", log_file="execution.log"):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str): The name of the log file to be saved in Config.WORKING_DIR.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    log_path = os.path.join(Config.WORKING_DIR, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_metrics(metrics_dict, logger, prefix="Metrics"):
    """
    Logs a dictionary of metrics with full precision.

    Args:
        metrics_dict (dict): Dictionary containing metric names and values.
        logger (logging.Logger): The logger instance to use.
        prefix (str): A prefix string for the log message.
    """
    parts = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            parts.append(f"{k}: {v}")  # Full precision, no formatting like .4f
        else:
            parts.append(f"{k}: {v}")

    message = f"{prefix}: " + " | ".join(parts)
    logger.info(message)
