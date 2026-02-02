import logging
import numpy as np
from library.config import Config
from library.utils import setup_logging, seed_everything
from library.data_manager import load_dataset
from library.modeling import train_models, predict_and_submit

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def _slice_data_dict(data_dict, limit):
    """
    Helper function to slice all arrays in the data dictionary for debugging.
    """
    if limit is None or limit <= 0:
        return data_dict

    logger.info(f"Subsampling dataset to {limit} samples for debugging.")
    sliced_dict = {}
    for key, value in data_dict.items():
        if value is not None and isinstance(value, np.ndarray):
            # Ensure we don't try to slice more than available
            current_len = len(value)
            slice_idx = min(limit, current_len)
            sliced_dict[key] = value[:slice_idx]
        else:
            sliced_dict[key] = value
    return sliced_dict


def train_ensemble(load_cached_data=True, load_cached_models=True, debug_limit=None):
    """
    Orchestrates the Stratified K-Fold Cross-Validation training.

    Args:
        load_cached_data (bool): Whether to load features from cache.
        load_cached_models (bool): Whether to load trained models from cache.
        debug_limit (int, optional): Limit the number of training samples for debugging.

    Returns:
        tuple: (pipelines, label_encoder)
            pipelines: List of trained sklearn pipelines.
            label_encoder: Fitted LabelEncoder.
    """
    seed_everything(Config.SEED)
    logger.info("Starting ensemble training pipeline...")

    # 1. Load Training Data
    # We use the 'train' split defined in metadata.
    train_data = load_dataset("train", load_cached_data=load_cached_data)

    # 2. Optional Debug Subsampling
    if debug_limit:
        train_data = _slice_data_dict(train_data, debug_limit)

    # 3. Train Models
    # Delegates to modeling.py which handles the K-Fold loop, OOF scoring, and model saving.
    pipelines, label_encoder, oof_probs, oof_targets = train_models(
        train_data, load_cached_models=load_cached_models
    )

    return pipelines, label_encoder


def predict_ensemble(pipelines, label_encoder, load_cached_data=True, debug_limit=None):
    """
    Orchestrates the inference on the test set and submission generation.

    Args:
        pipelines (list): List of trained sklearn pipelines.
        label_encoder (LabelEncoder): Fitted LabelEncoder.
        load_cached_data (bool): Whether to load features from cache.
        debug_limit (int, optional): Limit the number of test samples for debugging.
    """
    logger.info("Starting inference pipeline...")

    # 1. Load Test Data
    test_data = load_dataset("test", load_cached_data=load_cached_data)

    # 2. Optional Debug Subsampling
    if debug_limit:
        test_data = _slice_data_dict(test_data, debug_limit)

    # 3. Generate Predictions and Submit
    # Delegates to modeling.py which handles averaging predictions and saving to CSV.
    predict_and_submit(test_data, pipelines, label_encoder)
