import os
import numpy as np
import torch
from library.config import Config
from library.feature_engineering import FeaturePipeline
from library.model_rf import train_rf_model
from library.model_mlp import train_mlp_model


def run_training(load_cached_data: bool = True):
    """
    Orchestrates the full training pipeline for the Hybrid Ensemble solution.

    Steps:
    1. Executes Feature Engineering to generate specific inputs for RF and MLP streams.
    2. Trains the Interaction-Enhanced Consistency Random Forest.
    3. Trains the Orthogonal Skip-Gated MLP.

    Args:
        load_cached_data (bool): If True, attempts to load features from cache (parquet/npy).
                                 If False or cache missing, re-computes features.

    Returns:
        tuple: Contains (rf_model, mlp_model, X_rf_test, X_mlp_test)
               - rf_model: Trained InteractionRandomForest instance.
               - mlp_model: Trained SkipGatedMLP (nn.Module).
               - X_rf_test: Numpy array of RF features for the test set.
               - X_mlp_test: Dictionary of tensors for MLP features for the test set.
    """
    # ==========================================
    # 1. Feature Engineering
    # ==========================================
    print("Initializing Feature Engineering Pipeline...")
    pipeline = FeaturePipeline()

    # Retrieve all data splits for both models
    # process_data handles caching internally based on the flag
    data = pipeline.process_data(load_cached_data=load_cached_data)

    (
        X_rf_train,
        X_rf_val,
        X_rf_test,
        X_mlp_train,
        X_mlp_val,
        X_mlp_test,
        y_train,
        y_val,
    ) = data

    # ==========================================
    # 2. Train Random Forest Stream
    # ==========================================
    print("\n" + "=" * 40)
    print("Starting Random Forest Training Stream")
    print("=" * 40)

    # Train and validate RF
    # This function handles initialization, fitting, and printing validation AUC
    rf_model = train_rf_model(X_rf_train, y_train, X_rf_val, y_val)

    # ==========================================
    # 3. Train MLP Stream
    # ==========================================
    print("\n" + "=" * 40)
    print("Starting MLP Training Stream")
    print("=" * 40)

    # Train and validate MLP
    # This function handles DataLoader creation, training loop, and early stopping
    # Returns (best_model, trainer_instance)
    mlp_model, mlp_trainer = train_mlp_model(X_mlp_train, y_train, X_mlp_val, y_val)

    print("\n" + "=" * 40)
    print("Training Pipeline Complete")
    print("=" * 40)

    return rf_model, mlp_model, X_rf_test, X_mlp_test
