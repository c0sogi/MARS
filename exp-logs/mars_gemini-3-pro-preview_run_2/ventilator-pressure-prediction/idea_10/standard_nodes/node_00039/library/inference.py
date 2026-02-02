import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import load_and_preprocess_data
from library.model import RGIBiLSTM
from library.utils import seed_everything


def predict(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of worker threads for data loading.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # 2. Load Data
    # We need to call load_and_preprocess_data to ensure the scaler is fitted on train
    # and applied consistently to test.
    print("Loading and preprocessing data...")
    _, _, test_dataset = load_and_preprocess_data(load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_CHECKPOINT}...")
    model = RGIBiLSTM().to(device)

    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}. Train the model first."
        )

    state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    print("Starting inference...")
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            # Forward pass
            # Output shape: (Batch, Seq)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())

    # Flatten predictions to 1D array
    # The model outputs (Batch, 80), so we flatten to match the row-wise CSV format
    predictions = np.concatenate(all_preds).flatten()

    # 5. ID Alignment
    # The dataset logic sorts by breath_id and time_step. We must replicate this
    # to ensure the IDs match the flattened prediction array.
    print("Aligning IDs...")
    df_test = pd.read_csv(Config.TEST_CSV)

    # Sort exactly as done in library/dataset.py -> create_dataset
    df_test = df_test.sort_values([Config.BREATH_ID_COL, Config.TIME_COL])

    ids = df_test[Config.ID_COL].values

    # Sanity Check
    if len(ids) != len(predictions):
        raise ValueError(
            f"Shape mismatch! Test IDs count ({len(ids)}) does not match "
            f"Predictions count ({len(predictions)})."
        )

    # 6. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    submission_df = pd.DataFrame({"id": ids, "pressure": predictions})

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print("Submission generation complete.")
