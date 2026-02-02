import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.model import CWCDP_BiLSTM
from library.dataset import load_data


def generate_predictions(debug=Config.DEBUG, load_cached_data=True):
    """
    Generates predictions for the test set and saves them to a submission file.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting inference on device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Load Test Data
    # This handles feature engineering, scaling (using saved scaler), and caching.
    # The dataset returned is structured as (N_breaths, 80, Features).
    test_dataset = load_data("test", debug=debug, load_cached_data=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Crucial: Must not shuffle to maintain order
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Load Model
    model = CWCDP_BiLSTM().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train the model first."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Run Inference
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            # Forward pass
            # Output shape: (Batch, Seq_Len)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches
    # Shape: (N_breaths, 80)
    predictions_matrix = np.concatenate(all_preds, axis=0)

    # Flatten to (N_breaths * 80,) to match the ID list format
    predictions_flat = predictions_matrix.flatten()

    # 5. Align with IDs
    # We must reconstruct the ID order exactly as dataset.py processes it.
    # dataset.py logic:
    # 1. Load metadata to get breath_ids (filtering if debug)
    # 2. Load raw CSV
    # 3. Filter raw CSV by breath_ids
    # 4. Sort by [breath_id, time_step]

    print("Aligning predictions with IDs...")

    # Load Metadata to identify the correct breaths
    df_meta = pd.read_csv(Config.TEST_META)
    if debug:
        unique_breaths = df_meta["breath_id"].unique()
        sample_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
        target_breaths = set(sample_breaths)
    else:
        target_breaths = set(df_meta["breath_id"].unique())

    # Load Raw Test Data to get IDs and TimeSteps
    # We read only necessary columns to save memory/time
    df_test_raw = pd.read_csv(Config.TEST_CSV, usecols=["id", "breath_id", "time_step"])

    # Filter for the relevant breaths
    df_test = df_test_raw[df_test_raw["breath_id"].isin(target_breaths)].copy()

    # Sort (Crucial step to match dataset.py's X array structure)
    df_test = df_test.sort_values(["breath_id", "time_step"])

    # Extract IDs
    ids = df_test["id"].values

    # Integrity Check
    if len(ids) != len(predictions_flat):
        raise ValueError(
            f"Shape mismatch! IDs: {len(ids)}, Predictions: {len(predictions_flat)}. "
            "Ensure sorting and filtering logic matches dataset.py exactly."
        )

    # 6. Create Submission
    submission_df = pd.DataFrame({"id": ids, "pressure": predictions_flat})

    # 7. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH

    # If debug, we save to a debug file to avoid overwriting the main submission
    if debug:
        save_path = save_path.replace(".csv", "_debug.csv")

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Head:\n{submission_df.head()}")
