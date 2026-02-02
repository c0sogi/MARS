import os
import pandas as pd
from library.config import Config
from library.utils import set_seed, setup_logger
from library.feature_engineering import FeatureEngineer
from library.model_handler import DualStreamModel


def run_training_pipeline(load_cached_data=True, force_retrain=False):
    """
    Executes the end-to-end training and evaluation pipeline for the Dual-Stream architecture.

    1. Sets global random seeds for reproducibility.
    2. Generates or loads features for Train, Validation, and Test sets using FeatureEngineer.
       - Stream A: Translational Differentials + Visual Consensus
       - Stream B: Rotational Differentials + Invariant Baselines
    3. Trains the DualStreamModel:
       - Applies Targeted Majority Undersampling (10:1 Negative/Positive ratio).
       - Trains Stream A and Stream B XGBoost models with Early Stopping.
       - Optimizes probability thresholds for each stream to maximize Validation MCC.
    4. Generates predictions on the Test set using the optimized thresholds.
    5. Saves the final submission file.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed features from the cache directory.
                                 If False or cache miss, computes features from scratch.
        force_retrain (bool): If True, ignores existing model files and forces a fresh training run.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # Initialize Logger and Seed
    logger = setup_logger("TrainEval")
    set_seed(Config.SEED)
    logger.info("Initializing Differential-Physics Dual-Stream Pipeline...")

    # --- Feature Engineering Phase ---
    fe = FeatureEngineer()

    # Load Training Data
    logger.info("Step 1/4: Retrieving Training Data...")
    train_data = fe.create_features(mode="train", load_cached_data=load_cached_data)

    # Load Validation Data
    logger.info("Step 2/4: Retrieving Validation Data...")
    val_data = fe.create_features(mode="validation", load_cached_data=load_cached_data)

    # Load Test Data
    logger.info("Step 3/4: Retrieving Test Data...")
    test_data = fe.create_features(mode="test", load_cached_data=load_cached_data)

    # --- Model Training & Optimization Phase ---
    logger.info("Step 4/4: Model Training and Threshold Optimization...")
    model = DualStreamModel()

    # The model handler encapsulates:
    # 1. Undersampling of negatives
    # 2. Training with Early Stopping
    # 3. Linear Search for optimal MCC threshold on Validation set
    model.train(train_data, val_data, force_retrain=force_retrain)

    # --- Inference Phase ---
    logger.info("Generating Test Predictions...")
    submission_df = model.predict(test_data)

    # --- Submission Saving ---
    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Final Submission Shape: {submission_df.shape}")

    # Log basic statistics for sanity check
    if not submission_df.empty:
        contact_rate = submission_df["contact"].mean()
        logger.info(f"Predicted Global Contact Rate: {contact_rate}")

    return submission_df
