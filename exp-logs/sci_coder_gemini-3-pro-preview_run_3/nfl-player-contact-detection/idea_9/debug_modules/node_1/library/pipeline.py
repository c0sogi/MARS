import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

# Import from provided library files
from library.utils import seed_everything
from library.features import FeatureEngineer
from library.model_wrapper import DualStreamModel
import library.config as config  # Import config module to allow runtime parameter patching


def run_training(
    load_cached_data: bool = True,
    debug: bool = False,
    n_estimators: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> DualStreamModel:
    """
    Orchestrates the training pipeline for the Dual-Stream GBDT.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from disk.
        debug (bool): If True, drastically reduces dataset size for debugging.
        n_estimators (int, optional): Override for XGBoost n_estimators.
        learning_rate (float, optional): Override for XGBoost learning_rate.

    Returns:
        DualStreamModel: The trained model wrapper.
    """
    seed_everything(config.SEED)

    # --- 1. Hyperparameter Overrides ---
    if n_estimators is not None:
        config.XGB_PARAMS["n_estimators"] = n_estimators
    if learning_rate is not None:
        config.XGB_PARAMS["learning_rate"] = learning_rate

    print(f"Starting Training Pipeline (Debug={debug})...")

    # --- 2. Feature Engineering ---
    print("Generating Training Features...")
    fe_train = FeatureEngineer(mode="train")
    train_data = fe_train.generate_features(load_cached_data=load_cached_data)

    print("Generating Validation Features...")
    fe_val = FeatureEngineer(mode="validation")
    val_data = fe_val.generate_features(load_cached_data=load_cached_data)

    # --- 3. Debug Slicing ---
    if debug:
        print("Debug mode active: Slicing datasets to 1000 samples.")
        debug_size = 1000

        def slice_stream_data(data_dict):
            sliced_dict = {}
            for stream, (X, y, ids) in data_dict.items():
                limit = min(len(X), debug_size)
                sliced_dict[stream] = (
                    X.iloc[:limit].copy(),
                    y[:limit].copy(),
                    ids[:limit].copy(),
                )
            return sliced_dict

        train_data = slice_stream_data(train_data)
        val_data = slice_stream_data(val_data)

    # --- 4. Model Training ---
    print("Initializing Dual-Stream Model...")
    model = DualStreamModel()

    # Train both streams (A: Player-Player, B: Player-Ground)
    # The wrapper handles undersampling and threshold optimization internally.
    model.train(train_data, val_data)

    print("Training Pipeline Completed.")
    return model


def run_inference(
    model: Optional[DualStreamModel] = None, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Orchestrates the inference pipeline.

    Args:
        model (DualStreamModel, optional): Trained model instance. If None, loads from disk.
        load_cached_data (bool): Whether to load pre-computed features from disk.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    seed_everything(config.SEED)
    print("Starting Inference Pipeline...")

    # --- 1. Feature Engineering ---
    print("Generating Test Features...")
    fe_test = FeatureEngineer(mode="test")
    test_data = fe_test.generate_features(load_cached_data=load_cached_data)

    # --- 2. Model Loading ---
    if model is None:
        print("No model instance provided. Loading models from disk...")
        model = DualStreamModel()
        model.load_models()

    # --- 3. Prediction ---
    # The predict method handles routing samples to Stream A or B based on contact_id,
    # applying specific thresholds, and saving the submission file.
    submission_df = model.predict(test_data)

    print(f"Inference Completed. Submission shape: {submission_df.shape}")
    return submission_df
