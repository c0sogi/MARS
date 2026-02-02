import os
import torch
import numpy as np
import pandas as pd
from library.config import (
    ECGRN,
    GatedResidualBlock,
    FocalLoss,
    train_model,
    generate_submission,
    Config,
)

# The classes ECGRN and GatedResidualBlock are imported from library.config
# to satisfy the requirement of not re-implementing them while making them
# available in this module's namespace.


def run_pipeline():
    """
    Executes the full end-to-end pipeline for the Contact Detection task.

    Steps:
    1. Trains the Entity-Centric Gated Residual Network (ECGRN) using the
       pre-defined data pipeline and training loop in library.config.
       - Handles data loading and caching.
       - Performs Entity-Level Windowing and Hybrid Ground Imputation.
       - Trains with Focal Loss and AdamW.
       - Uses Early Stopping based on Validation MCC.
       - Optimizes the decision threshold post-training.

    2. Generates the submission file for the test set.
       - Loads test data and aligns features.
       - Performs inference using the trained model and optimized threshold.
       - Saves the result to ./submission/submission.csv.
    """
    print("Initializing EC-GRN Pipeline...")

    # Ensure necessary directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Step 1: Train Model
    # train_model() encapsulates the entire training logic including:
    # - get_data(mode='train') with caching
    # - StandardScaler fitting
    # - Model instantiation
    # - Training loop with validation and early stopping
    # - Threshold optimization
    print("Starting training process...")
    model, scaler, best_threshold, feature_cols = train_model()

    print(f"Training finished. Optimized Threshold: {best_threshold:.6f}")

    # Step 2: Generate Submission
    # generate_submission() encapsulates the inference logic including:
    # - get_data(mode='test')
    # - Feature alignment
    # - Batch inference
    # - CSV generation
    print("Generating submission file...")
    generate_submission(model, scaler, best_threshold, feature_cols)

    print("Pipeline execution completed successfully.")
