import sys
import os
import pandas as pd
import numpy as np
import torch

# Ensure the current directory is in the path so we can import the library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data_processing import get_dataloaders
from library.model import DSPFE
from library.train_eval import train_model, predict_submission


# ==========================================
# Demonstration Configuration
# ==========================================
class DemoConfig(Config):
    """
    Configuration overrides for the demonstration script.
    Optimized for speed by reducing data size and epochs.
    """

    # Run for fewer epochs to save time
    EPOCHS = 2

    # Use a smaller batch size suitable for the small dataset subset
    BATCH_SIZE = 128

    # Paths for demo outputs (stored in working directory)
    SUBMISSION_PATH = "./working/demo_submission.csv"
    MODEL_SAVE_PATH = "./working/demo_model.pth"

    # Limit samples to ensure quick execution (Speed Optimization)
    MAX_SAMPLES = 2000


def main():
    print("Starting Library Usage Demonstration...")

    # 1. Set Random Seed for Reproducibility
    set_seed(DemoConfig.SEED)
    print("Random seed set.")

    # 2. Data Loading and Processing
    # We use max_samples to load only a small subset of data for this demo.
    # This bypasses the cache saving mechanism in preprocess_data to avoid overwriting full data caches.
    print(f"Loading data (subset of {DemoConfig.MAX_SAMPLES} samples)...")

    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        batch_size=DemoConfig.BATCH_SIZE,
        load_cached_data=False,  # Force processing of the subset
        max_samples=DemoConfig.MAX_SAMPLES,
        config=DemoConfig,
    )

    # Validation: Check if loaders have data
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Validation: Check metadata content
    assert "vocab_sizes" in metadata
    assert "cont_cols" in metadata
    print("Data loaded successfully.")
    print(f"Vocab sizes: {metadata['vocab_sizes']}")
    print(f"Number of continuous features: {len(metadata['cont_cols'])}")

    # 3. Model Instantiation
    print("Initializing DSPFE Model...")
    model = DSPFE(
        vocab_sizes=metadata["vocab_sizes"],
        num_cont=len(metadata["cont_cols"]),
        stream_configs=DemoConfig.STREAMS_CONFIG,
        embed_dim=DemoConfig.EMBEDDING_DIM,
    )

    # Validation: Check model structure
    assert hasattr(model, "streams"), "Model should have 'streams' attribute."
    assert len(model.streams) == len(
        DemoConfig.STREAMS_CONFIG
    ), "Model stream count mismatch."
    print("Model initialized.")

    # 4. Training
    print(f"Starting training for {DemoConfig.EPOCHS} epochs...")
    train_model(model, train_loader, val_loader, DemoConfig)

    # Validation: Check if model weights were saved
    # Note: train_model saves the model only if validation AUC improves from 0.0.
    # With 2 epochs and random init, it is statistically guaranteed to be > 0.0.
    assert os.path.exists(
        DemoConfig.MODEL_SAVE_PATH
    ), f"Model file not found at {DemoConfig.MODEL_SAVE_PATH}"
    print("Training complete and model saved.")

    # 5. Prediction and Submission
    print("Generating predictions...")

    # We need the IDs for the test set to create the submission file.
    # Since we used a subset (max_samples), we must read the same subset of IDs from the source file.
    # The get_dataloaders function with max_samples slices the dataframe using iloc[:max_samples].
    df_test_subset = pd.read_csv(DemoConfig.TEST_PATH).iloc[: DemoConfig.MAX_SAMPLES]
    test_ids = df_test_subset["id"].values

    predict_submission(model, test_loader, test_ids, DemoConfig)

    # 6. Final Validation
    print("Verifying submission file...")
    assert os.path.exists(
        DemoConfig.SUBMISSION_PATH
    ), "Submission file was not created."

    submission_df = pd.read_csv(DemoConfig.SUBMISSION_PATH)

    # Check shape
    expected_shape = (DemoConfig.MAX_SAMPLES, 2)
    assert (
        submission_df.shape == expected_shape
    ), f"Submission shape mismatch. Expected {expected_shape}, got {submission_df.shape}"

    # Check columns
    assert list(submission_df.columns) == [
        "id",
        "target",
    ], "Submission columns mismatch."

    # Check value ranges
    probs = submission_df["target"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities must be between 0 and 1."

    print("Demonstration finished successfully. All checks passed.")


if __name__ == "__main__":
    main()
