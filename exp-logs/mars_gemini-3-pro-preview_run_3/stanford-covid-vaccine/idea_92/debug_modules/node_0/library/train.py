import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_all
from library.data import get_dataloaders
from library.model import HighCapacityBiGRU, train_model, predict


def run_training(debug=Config.DEBUG):
    """
    Orchestrates the training, validation, and submission generation process.

    Args:
        debug (bool): If True, uses a small subset of the data for debugging purposes.
                      Defaults to the value in Config.DEBUG.
    """
    # 1. Setup and Reproducibility
    seed_all(Config.SEED)

    # Ensure working and output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    submission_dir = os.path.dirname(Config.SUBMISSION_PATH)
    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)

    print(f"Initializing training run (Debug={debug})...")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # get_dataloaders handles loading from cache or processing from parquet
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 3. Model Initialization
    model = HighCapacityBiGRU()
    # Note: train_model will move model to device, but we can do it here too
    model.to(Config.DEVICE)

    # 4. Training Loop
    # Delegates to the library function which implements the loop, validation,
    # optimizer setup, scheduler, and early stopping.
    train_model(model, train_loader, val_loader)

    # 5. Inference
    print("Loading best model for inference...")
    # Load the best model weights saved during training
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)

    print("Generating predictions on test set...")
    # predict returns a numpy array of shape (N_samples, seq_len, 5)
    preds = predict(model, test_loader, Config.DEVICE)

    # 6. Submission Generation
    print("Formatting submission file...")

    # Retrieve sample IDs from the test dataset
    # The loader is not shuffled, so order matches preds
    test_ids = test_loader.dataset.ids
    seq_len = Config.SEQ_LEN

    # Flatten predictions to (N_samples * seq_len, 5)
    preds_flat = preds.reshape(-1, 5)

    # Generate the 'id_seqpos' column
    # Format: {id}_{position}
    id_seqpos_list = []
    for sample_id in test_ids:
        for pos in range(seq_len):
            id_seqpos_list.append(f"{sample_id}_{pos}")

    # Create DataFrame
    submission_df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)

    # Insert id_seqpos as the first column
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
