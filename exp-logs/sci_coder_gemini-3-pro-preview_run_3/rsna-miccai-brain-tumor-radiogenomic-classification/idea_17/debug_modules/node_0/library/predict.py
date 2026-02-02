import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import SSVEDataset
from library.model import SSVEModel


def run_inference(load_cached_data=True):
    """
    Executes the inference pipeline on the test set.

    Steps:
    1. Loads the test dataset (using cached data if available).
    2. Loads the trained SSVEModel.
    3. Iterates through the test set, computing predictions for both View A and View B.
    4. Averages the probabilities (Ensemble).
    5. Saves the results to submission.csv.
    """

    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Test Data
    print("Initializing Test Dataset...")
    # SSVEDataset with mode='test' returns (images, targets)
    # images shape: (2, 64, 256, 256) -> [View A, View B]
    test_dataset = SSVEDataset(mode="test", load_cached_data=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    print("Loading Model...")
    model = SSVEModel()
    model.to(device)

    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded weights from {Config.MODEL_PATH}")
    else:
        print(
            f"WARNING: Model file not found at {Config.MODEL_PATH}. Using random weights."
        )

    model.eval()

    # 4. Inference Loop
    print("Starting Inference...")
    all_preds = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            # images shape: (Batch, 2, 64, 256, 256)

            # Split views and move to device
            view_a = images[:, 0, ...].to(device)  # (Batch, 64, 256, 256)
            view_b = images[:, 1, ...].to(device)  # (Batch, 64, 256, 256)

            # Forward pass for View A
            logits_a = model(view_a)
            probs_a = torch.sigmoid(logits_a)

            # Forward pass for View B
            logits_b = model(view_b)
            probs_b = torch.sigmoid(logits_b)

            # Ensemble: Average the probabilities
            avg_probs = (probs_a + probs_b) / 2.0

            # Store predictions
            all_preds.extend(avg_probs.cpu().numpy().flatten())

    # 5. Format Submission
    print("Formatting Submission...")

    # Retrieve BraTS21IDs from the dataset
    # These are strings like "00001" based on the metadata generation
    patient_ids = test_dataset.get_ids()

    # Create DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": patient_ids, "MGMT_value": all_preds})

    # Ensure BraTS21ID is strictly integer if required by the specific sample_submission format
    # The prompt description shows "00001", but sample_submission.csv usually uses int64.
    # We will convert to int to be safe and consistent with standard Kaggle BraTS format.
    submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("-" * 30)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
    print("-" * 30)
    print(submission_df.head())
