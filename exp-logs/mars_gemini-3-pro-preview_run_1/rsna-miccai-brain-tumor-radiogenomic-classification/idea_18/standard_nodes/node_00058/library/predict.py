import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import prepare_data
from library.model import WIISNet


def predict_and_submit():
    """
    Executes the inference pipeline for the Weight-Inflated Independent-Slab (WIIS) Network.

    Steps:
    1. Sets random seeds for reproducibility.
    2. Loads the test dataset (generating slabs if not cached).
    3. Loads the trained WIISNet model from the checkpoint.
    4. Performs inference on the test slabs.
    5. Aggregates predictions (mean of 3 slabs) per subject.
    6. Saves the final predictions to submission.csv.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Inference using device: {device}")

    # 2. Prepare Data
    # prepare_data returns (train, val, test). We only need test.
    # The function handles caching internally based on Config.LOAD_CACHED_DATA.
    _, _, test_dataset = prepare_data(load_cached_data=Config.LOAD_CACHED_DATA)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    model = WIISNet()

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model weights from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.BEST_MODEL_PATH}. Using random weights."
        )

    model = model.to(device)
    model.eval()

    print(f"Generating predictions for {len(test_dataset)} slabs...")

    # 4. Inference Loop
    subject_predictions = defaultdict(list)

    with torch.no_grad():
        for images, subject_ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Map predictions to subject IDs
            # subject_ids is a tensor of shape (Batch,)
            current_ids = subject_ids.numpy()

            for sid, prob in zip(current_ids, probs):
                subject_predictions[sid].append(prob)

    # 5. Consensus Aggregation
    submission_data = []
    for sid, probs in subject_predictions.items():
        # Single prediction per subject
        mean_prob = np.mean(probs)
        submission_data.append({"BraTS21ID": sid, "MGMT_value": mean_prob})

    # 6. Save Submission
    df_submission = pd.DataFrame(submission_data)

    # Sort by ID for consistency
    df_submission = df_submission.sort_values("BraTS21ID")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total subjects predicted: {len(df_submission)}")
    print("Head of submission:")
    print(df_submission.head())
