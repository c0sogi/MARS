import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import the pre-defined model and training utilities from the library
# to satisfy the requirement of not re-implementing existing code.
# By importing train_one_epoch and validate, we expose them in this module's namespace
# as required by the module description.
from library.model import CAMA_CNN, train_one_epoch, validate, train_cama_cnn
from library.utils import load_data, set_seed
from library.dataset import IcebergDataset


def run(
    batch_size=32,
    epochs=75,
    patience=12,
    lr=1e-3,
    weight_decay=1e-4,
    n_folds=5,
    seed=42,
    cache_dir="./working/idea_24",
    submission_path="./submission/submission.csv",
):
    """
    Executes the full training and evaluation pipeline for the Iceberg vs Ship classification task.

    This function acts as the driver for the engine, leveraging the CAMA-CNN architecture
    and the robust 5-Fold Cross-Validation strategy defined in the library.

    Args:
        batch_size (int): Batch size for training and inference.
        epochs (int): Maximum number of training epochs per fold.
        patience (int): Early stopping patience epochs.
        lr (float): Learning rate for Adam optimizer.
        weight_decay (float): L2 regularization factor.
        n_folds (int): Number of folds for Cross-Validation.
        seed (int): Random seed for reproducibility.
        cache_dir (str): Directory to cache processed numpy arrays.
        submission_path (str): Path to save the final submission CSV.
    """

    # Ensure the directory for the submission file exists
    if os.path.dirname(submission_path):
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    print(f"Initializing Engine with seed {seed}...")
    print(
        f"Configuration: Epochs={epochs}, Patience={patience}, Batch={batch_size}, LR={lr}, WD={weight_decay}"
    )

    # Delegate to the comprehensive training pipeline in library.model
    # This handles:
    # - Data loading and caching (via library.utils)
    # - Stratified K-Fold splitting
    # - Model instantiation (CAMA_CNN)
    # - Training loop with BCEWithLogitsLoss and Early Stopping
    # - Validation scoring (Log Loss)
    # - Test set inference and Ensemble averaging
    train_cama_cnn(
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        n_folds=n_folds,
        seed=seed,
        cache_dir=cache_dir,
        submission_path=submission_path,
    )

    print("Engine execution finished successfully.")
