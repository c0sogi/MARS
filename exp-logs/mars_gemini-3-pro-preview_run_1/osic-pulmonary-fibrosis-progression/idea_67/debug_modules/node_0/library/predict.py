import os
import torch
import pandas as pd
from library.config import Config
from library.model import get_extended_dataloaders, generate_submission


def inference_fn(
    model_path,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=False,
):
    """
    Generates predictions for the test set using the trained TSCG-Net model.

    This function orchestrates the inference pipeline:
    1. Initializes data loaders (fitting tabular preprocessors on training metadata).
    2. Loads the trained model checkpoint.
    3. Predicts trajectory parameters (alpha, sigma_base, sigma_growth).
    4. Computes FVC and Confidence for every requested Patient_Week.
    5. Saves the result to the submission file.

    Args:
        model_path (str): Path to the trained model checkpoint (.pth).
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for data loading.
        debug (bool): If True, runs on a small subset of data for debugging purposes.
    """
    print(f"Starting inference pipeline (Debug={debug})...")

    # Temporarily override the global DEBUG config to control dataset size
    original_debug = Config.DEBUG
    Config.DEBUG = debug

    try:
        # 1. Prepare Data Loaders
        # We call get_extended_dataloaders to ensure the TabularPreprocessor is correctly
        # fitted on the training data statistics before being applied to the test set.
        # We only need the test_loader for inference.
        print("Initializing data loaders and preprocessors...")
        _, _, test_loader = get_extended_dataloaders(
            batch_size=batch_size, num_workers=num_workers
        )

        # 2. Validate Model Path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # 3. Generate Submission
        # The generate_submission function in library.model handles:
        # - Loading the model state dict
        # - Iterating over the test_loader
        # - Calculating FVC = Baseline + alpha * delta_week
        # - Calculating Sigma = Sigma_base + Sigma_growth * |delta_week|
        # - Saving the formatted CSV to Config.SUBMISSION_PATH
        print(f"Running inference using model: {model_path}")
        generate_submission(model_path, test_loader)

        print("Inference completed successfully.")

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        raise e

    finally:
        # Restore the original configuration state
        Config.DEBUG = original_debug
