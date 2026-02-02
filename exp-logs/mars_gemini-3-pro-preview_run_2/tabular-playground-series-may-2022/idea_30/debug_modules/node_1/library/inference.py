import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything
from library.model import RoPESwiGLURMSNet
from library.data import get_dataloaders


def predict(load_cached_data: bool = True, debug: bool = False):
    """
    Loads the trained model, generates predictions for the test set,
    and creates a submission file.

    Args:
        load_cached_data (bool): Whether to use cached processed data.
        debug (bool): If True, runs on a subset of data for debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting inference (Debug={debug})...")
    print(f"Device: {device}")

    # 2. Load Data
    # We only need the test loader here
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data, debug=debug)

    # 3. Load Test Metadata for IDs
    # The data loader preserves the order of the metadata file.
    # We need to load the IDs to construct the submission dataframe.
    if not os.path.exists(Config.TEST_META_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META_PATH}")

    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Apply the same slicing logic as in library.data.process_data if debug is True
    if debug:
        print(f"Debug mode: Slicing test metadata to {Config.DEBUG_SAMPLES} samples.")
        df_test_meta = df_test_meta.iloc[: Config.DEBUG_SAMPLES]

    test_ids = df_test_meta["id"].values

    # 4. Load Model
    model = RoPESwiGLURMSNet()
    model.to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_PATH}. Train the model first."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat = x_cat.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)

            # Forward pass
            logits = model(x_cat, x_cont)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and store
            all_preds.append(probs.cpu().numpy())

    # Concatenate all batches
    if len(all_preds) > 0:
        predictions = np.concatenate(all_preds).flatten()
    else:
        predictions = np.array([])

    # 6. Validation Check
    if len(predictions) != len(test_ids):
        raise ValueError(
            f"Mismatch between number of predictions ({len(predictions)}) "
            f"and number of test IDs ({len(test_ids)})."
        )

    # 7. Create Submission
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
