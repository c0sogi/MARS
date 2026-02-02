import os
import pandas as pd
import torch
import numpy as np

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import HybridModel
from library.train import predict


def run_inference(load_cached_data=True, **kwargs):
    """
    Executes the inference pipeline: loads data, loads the model, generates predictions,
    and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached metadata files.
        **kwargs: Overrides for Config attributes (e.g., debug=True).
    """
    # 1. Initialize Configuration
    config = Config(**kwargs)

    # 2. Set Random Seed for Reproducibility
    set_seed(config.seed)

    print(f"Initializing inference with device: {config.device}")

    # 3. Prepare DataLoaders
    # We only need the test loader here.
    loaders = get_dataloaders(config, load_cached_data=load_cached_data)
    test_loader = loaders.get("test")

    if test_loader is None:
        raise ValueError(
            "Test loader could not be initialized. Check if test.csv exists."
        )

    print(f"Test loader ready. Batches: {len(test_loader)}")

    # 4. Initialize Model
    device = torch.device(config.device)
    model = HybridModel(config)
    model.to(device)

    # 5. Load Trained Weights
    if not os.path.exists(config.model_save_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {config.model_save_path}"
        )

    print(f"Loading model weights from {config.model_save_path}...")
    checkpoint = torch.load(config.model_save_path, map_location=device)
    model.load_state_dict(checkpoint)

    # 6. Generate Predictions
    # Using the predict function from library.train to ensure consistency
    print("Starting prediction loop...")
    test_probs = predict(model, test_loader, device)

    # 7. Format Submission
    print("Formatting submission...")

    # Load test metadata to retrieve eeg_ids
    # We use the path directly from config
    if not os.path.exists(config.test_csv):
        raise FileNotFoundError(f"Test metadata file not found at {config.test_csv}")

    df_test = pd.read_csv(config.test_csv)

    # If debug mode is on, the loader might have truncated the dataset.
    # We must ensure the dataframe matches the predictions length.
    if len(df_test) != len(test_probs):
        print(
            f"Note: Mismatch between metadata rows ({len(df_test)}) and predictions ({len(test_probs)}). Truncating metadata to match (Debug mode likely)."
        )
        df_test = df_test.iloc[: len(test_probs)]

    # Create submission DataFrame
    submission = pd.DataFrame(test_probs, columns=config.vote_cols)
    submission.insert(0, "eeg_id", df_test["eeg_id"])

    # 8. Save Submission
    os.makedirs(config.submission_dir, exist_ok=True)
    submission.to_csv(config.submission_path, index=False)

    print(f"Submission saved successfully to {config.submission_path}")
    print(f"Submission shape: {submission.shape}")

    # Print first few rows for verification (full precision)
    print("First 5 rows of submission:")
    print(submission.head().to_string())
