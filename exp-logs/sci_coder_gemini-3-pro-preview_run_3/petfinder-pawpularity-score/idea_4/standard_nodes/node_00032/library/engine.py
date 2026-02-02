import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import get_score
from library.models import AdaptiveBackbone
from library.ensemble import extract_features_and_cache


def extract_features(model, dataloader, mode, tta=False, load_cached_data=True):
    """
    Extracts features using the fine-tuned model.
    Wraps the caching and TTA logic provided in library.ensemble.

    Args:
        model: The fine-tuned AdaptiveBackbone model.
        dataloader: DataLoader for the dataset.
        mode: 'train', 'valid', or 'test'.
        tta: Whether to apply Test-Time Augmentation.
        load_cached_data: Whether to try loading from cache first.

    Returns:
        tuple: (features, targets, ids)
    """
    return extract_features_and_cache(
        model=model,
        dataloader=dataloader,
        device=Config.DEVICE,
        mode=mode,
        tta=tta,
        load_cached_data=load_cached_data,
    )
