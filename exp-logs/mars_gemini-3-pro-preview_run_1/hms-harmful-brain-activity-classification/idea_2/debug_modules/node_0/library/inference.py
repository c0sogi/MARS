import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import EEGNet1D
from library.data_loader import get_dataloaders


def generate_submission(debug=False, load_cached_data=False):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        debug (bool): If True, runs on a subset of the data for debugging.
        load_cached_data (bool): If True, attempts to load predictions from cache.
    """
    # Ensure working directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    cache_path = Config.PREDS_PATH

    # Get Test Loader
    # We ignore train/val loaders here
    _, _, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
    )

    predictions = None

    # --- Caching Logic ---
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading cached predictions from {cache_path}")
            try:
                cached_preds = np.load(cache_path)
                # Verify consistency with current dataset
                if len(cached_preds) == len(test_loader.dataset):
                    predictions = cached_preds
                else:
                    print(
                        f"Cache shape mismatch (Cache: {len(cached_preds)}, "
                        f"Data: {len(test_loader.dataset)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cache file not found. Computing predictions...")

    # --- Inference ---
    if predictions is None:
        device = torch.device(Config.DEVICE)
        print(f"Running inference on device: {device}")

        # Initialize Model
        model = EEGNet1D(config=Config).to(device)

        # Load Weights
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {Config.MODEL_PATH}. "
                "Please train the model first."
            )

        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        all_probs = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)

                # Forward pass (returns probabilities)
                probs = model(inputs)
                all_probs.append(probs.cpu().numpy())

        if len(all_probs) > 0:
            predictions = np.concatenate(all_probs, axis=0)
        else:
            predictions = np.zeros((0, Config.NUM_CLASSES))

        # Save to cache
        try:
            np.save(cache_path, predictions)
            print(f"Predictions saved to cache at {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    # --- Submission Generation ---
    print("Generating submission file...")

    # Retrieve eeg_ids from the dataset
    # The loader preserves the order of the dataset
    eeg_ids = test_loader.dataset.df["eeg_id"].values

    if len(predictions) != len(eeg_ids):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match "
            f"ID count ({len(eeg_ids)})"
        )

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=Config.OUTPUT_COLS)
    submission_df.insert(0, "eeg_id", eeg_ids)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
