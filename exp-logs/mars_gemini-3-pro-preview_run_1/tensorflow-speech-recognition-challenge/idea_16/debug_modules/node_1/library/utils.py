import os
import sys
from typing import Union, List

# Import existing implementations and configuration to avoid duplication
from library.config import Config, ModelEMA, set_seed


def map_fine_grained_to_12_class(label: str) -> str:
    """
    Maps a fine-grained label (31+ classes) to the 12-class submission format.

    The logic follows the competition metric:
    - If the label is one of the 10 target commands, keep it.
    - If the label is 'silence', keep it.
    - All other labels (auxiliary words like 'bed', 'bird', etc.) are mapped to 'unknown'.

    Args:
        label (str): The predicted fine-grained label.

    Returns:
        str: The mapped label ('yes', 'no', ..., 'silence', or 'unknown').
    """
    if label in Config.TARGET_LABELS:
        return label
    elif label == Config.SILENCE_LABEL:
        return Config.SILENCE_LABEL
    else:
        return Config.UNKNOWN_LABEL
