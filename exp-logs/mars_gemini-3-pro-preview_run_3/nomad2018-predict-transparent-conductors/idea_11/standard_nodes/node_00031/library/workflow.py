import os
import random
import numpy as np
import torch
import pandas as pd
import warnings

from library.config import SUBMISSION_PATH, DEBUG_SAMPLE_SIZE, RANDOM_SEED
from library.feature_engineering import FeaturePipeline
from library.model_wrapper import DualEnergyPredictor

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Suppress warnings
warnings.filterwarnings("ignore")


def train_and_evaluate(
    sample_size: int = DEBUG_SAMPLE_SIZE, load_cached_data: bool = True
):
    """
    Orchestrates the training and evaluation process.

    Args:
        sample_size (int): Number of samples to use for training/validation (for debugging).
        load_cached_data (bool): Whether to load features from cache if available.

    Returns:
        DualEnergyPredictor: The trained model wrapper.
    """
    print("--- Starting Training and Evaluation Workflow ---")

    # Initialize Feature Pipeline
    pipeline = FeaturePipeline()

    # 1. Process Training Data
    print(f"Processing training data (sample_size={sample_size})...")
    train_df = pipeline.process_split(
        split="train", sample_size=sample_size, load_cached_data=load_cached_data
    )

    # 2. Process Validation Data
    print(f"Processing validation data (sample_size={sample_size})...")
    val_df = pipeline.process_split(
        split="val", sample_size=sample_size, load_cached_data=load_cached_data
    )

    # 3. Initialize and Train Model
    print("Initializing DualEnergyPredictor...")
    predictor = DualEnergyPredictor()

    print("Fitting models...")
    predictor.fit(train_df, val_df)

    # 4. Evaluate Model
    print("Evaluating models on validation set...")
    predictor.evaluate(val_df)

    print("Training and evaluation completed.")
    return predictor


def generate_submission(
    predictor: DualEnergyPredictor,
    sample_size: int = None,
    load_cached_data: bool = True,
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        predictor (DualEnergyPredictor): The trained model wrapper.
        sample_size (int): Number of test samples to process (for debugging).
        load_cached_data (bool): Whether to load features from cache if available.
    """
    print("\n--- Starting Submission Generation Workflow ---")

    # Initialize Feature Pipeline
    pipeline = FeaturePipeline()

    # 1. Process Test Data
    print(f"Processing test data (sample_size={sample_size})...")
    test_df = pipeline.process_split(
        split="test", sample_size=sample_size, load_cached_data=load_cached_data
    )

    # 2. Generate Predictions
    print("Generating predictions for test set...")
    predictions = predictor.predict(test_df)

    # 3. Save Submission
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission to {SUBMISSION_PATH}...")
    predictions.to_csv(SUBMISSION_PATH, index=False)

    print("Submission generated successfully.")
    print(predictions.head())
    return predictions
