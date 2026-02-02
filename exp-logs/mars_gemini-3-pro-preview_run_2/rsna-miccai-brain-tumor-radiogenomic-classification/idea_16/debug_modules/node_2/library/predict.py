import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    TEST_METADATA_PATH,
    BEST_MODEL_PATH,
    SUBMISSION_PATH,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything
from library.data import get_dataloader
from library.model import get_model


def predict_and_submit(load_cached_data=True):
    """
    Runs inference on the test set using the best saved model.
    Applies Test-Time Augmentation (TTA) for robust predictions.
    Generates the submission.csv file.

    Args:
        load_cached_data (bool): Whether to attempt loading ROI cache.
                                 Note: This is handled internally by the library.data module,
                                 but kept here for interface consistency.
    """
    # 1. Setup
    seed_everything(SEED)
    print("Starting Inference...")

    # 2. Check Prerequisites
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model weights not found at {BEST_MODEL_PATH}. Train the model first."
        )

    # 3. Load Data
    df_test = pd.read_csv(TEST_METADATA_PATH)
    print(f"Loaded test metadata: {len(df_test)} samples.")

    # Initialize DataLoader
    # Note: get_dataloader initializes BraTSDataset, which handles ROI caching internally.
    test_loader = get_dataloader(
        df_test,
        phase="test",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    # 4. Load Model
    model = get_model()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 5. Inference Loop with TTA
    results = []

    # Disable gradients for inference
    with torch.no_grad():
        for inputs, _, brats_ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(DEVICE)

            # TTA Strategy:
            # 1. Original
            # 2. Horizontal Flip (dim 3)
            # 3. Vertical Flip (dim 2)

            # Forward pass - Original
            outputs_orig = model(inputs)
            probs_orig = torch.sigmoid(outputs_orig)

            # Forward pass - Horizontal Flip
            inputs_hflip = torch.flip(inputs, dims=[3])
            outputs_hflip = model(inputs_hflip)
            probs_hflip = torch.sigmoid(outputs_hflip)

            # Forward pass - Vertical Flip
            inputs_vflip = torch.flip(inputs, dims=[2])
            outputs_vflip = model(inputs_vflip)
            probs_vflip = torch.sigmoid(outputs_vflip)

            # Average predictions
            probs_avg = (probs_orig + probs_hflip + probs_vflip) / 3.0

            # Move to CPU
            probs_avg = probs_avg.cpu().numpy().flatten()
            ids = brats_ids.numpy().flatten()

            for bid, prob in zip(ids, probs_avg):
                results.append({"BraTS21ID": int(bid), "MGMT_value": float(prob)})

    # 6. Format Submission
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    submission_df = submission_df[["BraTS21ID", "MGMT_value"]]

    # Sort by ID for cleanliness
    submission_df = submission_df.sort_values("BraTS21ID")

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
