import os
import sys
import torch
import numpy as np

# Import configuration and utilities from the provided library
from library.config import SEED
from library.utils import seed_everything

# Import model-related functions from the provided library
# predict_tta is imported to expose it as part of this module's interface
from library.model import get_data, generate_submission, predict_tta


def run_inference(num_samples=None, debug=False):
    """
    Executes the inference pipeline: loads data, (optionally) subsets it,
    and generates the submission file using the trained model ensemble.

    Args:
        num_samples (int, optional): The number of test samples to process.
                                     If None, processes the entire test set.
        debug (bool): If True and num_samples is None, defaults to a small subset (e.g., 10)
                      for quick debugging.
    """
    # Ensure reproducibility
    seed_everything(SEED)

    # Load test data using the library's caching mechanism
    # get_data returns: (train, val, test)
    _, _, (test_ids, test_noisy) = get_data(load_cached_data=True)

    # Handle dataset subsetting for debugging or quick validation
    if debug and num_samples is None:
        num_samples = 10

    if num_samples is not None:
        # Clamp num_samples to the actual size of the dataset
        num_samples = min(num_samples, len(test_ids))
        test_ids = test_ids[:num_samples]
        test_noisy = test_noisy[:num_samples]
        print(f"Running inference on a subset of {num_samples} samples.")

    # Generate submission using the library function
    # This function handles:
    # 1. Loading the 5 trained fold models
    # 2. Applying Test-Time Augmentation (TTA)
    # 3. Ensemble averaging
    # 4. Formatting and saving to ./submission/submission.csv
    generate_submission(test_ids, test_noisy)
