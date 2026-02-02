import os
import pandas as pd
import numpy as np
import torch
from library.config import Config, set_seed
from library.utils import get_device
from library.dataset import prepare_data
from library.model import FCPNet


def generate_submission(debug=False, load_cached_data=True):
    """
    Main inference function to generate the submission file.

    Args:
        debug (bool): If True, runs on a subset of data for testing.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = get_device()
    print(f"Inference Device: {device}")

    # 2. Load Data
    # We only need the test_loader. prepare_data handles caching and feature engineering.
    # It returns (train_loader, val_loader, test_loader)
    print("Loading test data...")
    _, _, test_loader = prepare_data(debug=debug, load_cached_data=load_cached_data)

    # 3. Initialize Model
    print("Initializing model architecture...")
    model = FCPNet(config=Config).to(device)

    # 4. Load Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train the model first."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 5. Run Inference
    print("Running inference on test set...")
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            # Output shape: (Batch, Seq_Len, 1)
            outputs = model(inputs)

            # Flatten to 1D array: (Batch * Seq_Len)
            # We use cpu().numpy() to move data back to host
            batch_preds = outputs.squeeze(-1).flatten().cpu().numpy()
            predictions.extend(batch_preds)

    predictions = np.array(predictions)
    print(f"Total predictions generated: {len(predictions)}")

    # 6. Align with IDs and Save
    # The dataset pipeline sorts data by [breath_id, time_step].
    # We must sort the metadata similarly to match predictions to IDs.
    print("Loading test metadata for ID alignment...")
    test_df = pd.read_csv(Config.TEST_FILE)

    if debug:
        # If debugging, we must filter the metadata to match the subset used in prepare_data
        test_breaths = test_df["breath_id"].unique()[:50]
        test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

    # Sort to ensure alignment with the sequence-based predictions
    test_df = test_df.sort_values(["breath_id", "time_step"])

    # Verification
    if len(test_df) != len(predictions):
        raise ValueError(
            f"Shape mismatch: Metadata has {len(test_df)} rows, "
            f"but generated {len(predictions)} predictions."
        )

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_df["id"], "pressure": predictions})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
