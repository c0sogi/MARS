import numpy as np
import pandas as pd
from library.utils import seed_everything
from library.model_resnet import run_resnet_cv as _lib_run_resnet_cv
from library.model_resnet import TabularDataset

# Implement ForestDataset as requested by aliasing the library's TabularDataset
# This dataset handles the specific needs of the ResNet (categorical indices + continuous features)
ForestDataset = TabularDataset


def run_resnet_cv(
    load_cached_data=True, n_splits=5, seed=42, epochs=50, batch_size=2048, patience=10
):
    """
    Manages the training lifecycle of the ResNet model using Stratified 5-Fold CV.

    This function wraps the library implementation to execute the training loop,
    which includes:
    - Data loading via DataProcessor (with caching)
    - Model initialization (TabularResNet)
    - Training with AdamW and OneCycleLR
    - Early Stopping based on validation Cross-Entropy Loss

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        n_splits (int): Number of cross-validation folds.
        seed (int): Random seed for reproducibility.
        epochs (int): Maximum number of training epochs per fold.
        batch_size (int): Batch size for training and inference.
        patience (int): Early stopping patience rounds.

    Returns:
        oof_preds (np.ndarray): Out-of-fold probability predictions (n_train, n_classes).
        test_preds (np.ndarray): Averaged test set probability predictions (n_test, n_classes).
        le (LabelEncoder): The label encoder used for the target variable.
        y_full (np.ndarray): The full array of target labels (aligned with oof_preds).
    """
    # Set seed for reproducibility
    seed_everything(seed)

    print(
        f"Starting ResNet CV Engine (Epochs={epochs}, Batch={batch_size}, Patience={patience})..."
    )

    # Delegate to the library implementation which contains the full logic
    # for DataProcessor, TabularResNet, and the training loop.
    oof_preds, test_preds, le, y_full = _lib_run_resnet_cv(
        load_cached_data=load_cached_data,
        n_splits=n_splits,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
    )

    return oof_preds, test_preds, le, y_full
