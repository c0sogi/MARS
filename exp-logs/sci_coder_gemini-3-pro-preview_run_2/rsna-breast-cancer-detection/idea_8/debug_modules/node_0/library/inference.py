import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config
from library.model import MultiTaskEfficientNet
from library.dataset import get_dataloaders


def generate_submission():
    """
    Generates the submission file for the breast cancer detection task.

    Steps:
    1. Loads the trained model.
    2. Loads the test dataset.
    3. Predicts cancer probability with TTA (Horizontal Flip).
    4. Aggregates predictions by prediction_id (Max).
    5. Saves to submission.csv.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # We only need the test loader.
    # load_cached_data=True allows using pre-processed parquet files if they exist.
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    if test_loader is None:
        print(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}. Skipping inference."
        )
        return

    # 3. Load Model
    # We initialize with pretrained=False to avoid downloading weights,
    # as we will load our own trained weights immediately.
    model = MultiTaskEfficientNet(pretrained=False)

    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading model weights from {Config.MODEL_PATH}...")
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Error: Model file not found at {Config.MODEL_PATH}. Cannot generate submission."
        )
        return

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print(f"Processing {len(test_loader.dataset)} test images...")

    with torch.no_grad():
        for images, meta_vec in test_loader:
            images = images.to(device)
            meta_vec = meta_vec.to(device)

            # --- TTA: Original Image ---
            with autocast():
                outputs_orig = model(images, meta_vec)
                # Extract cancer logits and convert to probability
                probs_orig = torch.sigmoid(outputs_orig["cancer"])

            # --- TTA: Horizontal Flip ---
            # Flip width dimension (B, C, H, W) -> dim 3
            images_flipped = torch.flip(images, dims=[3])

            with autocast():
                outputs_flip = model(images_flipped, meta_vec)
                probs_flip = torch.sigmoid(outputs_flip["cancer"])

            # --- Average Predictions ---
            probs_avg = (probs_orig + probs_flip) / 2.0

            # Collect results (flatten to 1D array)
            all_probs.extend(probs_avg.cpu().numpy().flatten())

    # 5. Map Predictions to IDs
    # The loader preserves the order of the dataset dataframe
    test_df = test_loader.dataset.df.copy()

    if len(all_probs) != len(test_df):
        print(
            f"Warning: Prediction count ({len(all_probs)}) matches metadata rows ({len(test_df)})."
        )
        # In case of mismatch (should not happen with drop_last=False), trim or pad
        min_len = min(len(all_probs), len(test_df))
        test_df = test_df.iloc[:min_len]
        all_probs = all_probs[:min_len]

    test_df["cancer_pred"] = all_probs

    # 6. Aggregation
    # Group by prediction_id and take the MAX probability
    print("Aggregating predictions by prediction_id...")
    submission_df = test_df.groupby("prediction_id")["cancer_pred"].max().reset_index()

    # Rename column to match submission format
    submission_df.rename(columns={"cancer_pred": "cancer"}, inplace=True)

    # 7. Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
