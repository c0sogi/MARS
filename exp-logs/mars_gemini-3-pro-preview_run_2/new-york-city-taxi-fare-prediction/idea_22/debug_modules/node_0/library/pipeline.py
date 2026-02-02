import os
import random
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import create_dual_hygiene_sets
from library.feature_engineering import process_and_cache_data
from library.model import TaxiFareXGBoost


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and environment variables.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_training_pipeline(
    load_cached_data=True,
    learner_sample_size=None,
    n_estimators=None,
    early_stopping_rounds=None,
):
    """
    Orchestrates the end-to-end training workflow:
    1. Sets up the environment and seeds.
    2. Loads and sanitizes data using the Dual-Hygiene strategy.
    3. Generates features using Hierarchical Distributional Fingerprinting (Wisdom)
       and Intersection-Filtered Vectorized Subtraction (Learner).
    4. Trains the XGBoost model with early stopping.

    Args:
        load_cached_data (bool): If True, attempts to load intermediate Parquet files from ./working.
        learner_sample_size (int, optional): Override for Config.LEARNER_SAMPLE_SIZE.
        n_estimators (int, optional): Override for Config.XGB_PARAMS['n_estimators'].
        early_stopping_rounds (int, optional): Override for Config.EARLY_STOPPING_ROUNDS.

    Returns:
        tuple: (trained_model, test_feat_df)
    """
    # 1. Apply Configuration Overrides
    if learner_sample_size is not None:
        print(f"Overriding LEARNER_SAMPLE_SIZE: {learner_sample_size}")
        Config.LEARNER_SAMPLE_SIZE = learner_sample_size

    if n_estimators is not None:
        print(f"Overriding XGB_PARAMS['n_estimators']: {n_estimators}")
        Config.XGB_PARAMS["n_estimators"] = n_estimators

    if early_stopping_rounds is not None:
        print(f"Overriding EARLY_STOPPING_ROUNDS: {early_stopping_rounds}")
        Config.EARLY_STOPPING_ROUNDS = early_stopping_rounds

    # 2. Setup and Reproducibility
    Config.setup()
    set_seed(Config.SEED)

    # 3. Data Loading (Dual-Hygiene Split)
    print("--- Starting Data Loading ---")
    wisdom_df, learner_df, val_df, test_df = create_dual_hygiene_sets(
        load_cached_data=load_cached_data
    )

    # 4. Feature Engineering
    print("--- Starting Feature Engineering ---")
    # This step fits the encoder on Wisdom and transforms Learner/Val/Test.
    # It handles the leakage prevention logic for the Learner set.
    learner_feat, val_feat, test_feat = process_and_cache_data(
        wisdom_df, learner_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 5. Model Training
    print("--- Starting Model Training ---")
    model = TaxiFareXGBoost()
    model.train(learner_feat, val_feat)

    return model, test_feat


def run_inference_pipeline(model, test_feat):
    """
    Orchestrates the inference workflow:
    1. Generates predictions for the test set using the trained model.
    2. Post-processes predictions (min fare floor).
    3. Saves the final submission file.

    Args:
        model (TaxiFareXGBoost): The trained model object.
        test_feat (pd.DataFrame): The featurized test dataset containing 'key'.
    """
    print("--- Starting Inference ---")

    # Generate predictions
    predictions = model.predict(test_feat)

    # Save submission
    # test_feat retains the 'key' column from the original test set, which is required for submission.
    model.save_submission(test_feat, predictions)

    print("Inference pipeline completed.")
