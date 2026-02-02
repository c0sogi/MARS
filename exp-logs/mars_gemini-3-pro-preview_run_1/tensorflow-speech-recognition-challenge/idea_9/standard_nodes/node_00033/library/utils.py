import sys
import logging
from library.config import Config, set_seed


def get_logger(name: str = "main"):
    """
    Configures and returns a logger instance that outputs to stdout.
    Ensures that handlers are not duplicated if the logger is requested multiple times.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def map_prediction_to_label(prediction_idx: int) -> str:
    """
    Maps the model's fine-grained prediction index (integer) to the
    competition's required 12-class string label.

    This function bridges the gap between the training objective (fine-grained classes)
    and the submission requirement (12 classes).

    Args:
        prediction_idx (int): The predicted class index from the model.

    Returns:
        str: The final label string ('yes', 'no', ..., 'unknown', 'silence').
    """
    # Ensure the index is a standard Python integer
    idx = int(prediction_idx)

    # Retrieve the mapping from integer ID to fine-grained label string
    # e.g., 0 -> 'bed', 1 -> 'bird', ..., index of 'yes' -> 'yes'
    id2label = Config.get_id2label()

    if idx not in id2label:
        # Fallback for safety; implies index is out of bounds
        return Config.UNKNOWN_LABEL

    fine_grained_label = id2label[idx]

    # Map the fine-grained label to the submission format
    # e.g., 'bed' -> 'unknown', 'yes' -> 'yes', 'silence' -> 'silence'
    submission_label = Config.map_prediction_to_submission(fine_grained_label)

    return submission_label
