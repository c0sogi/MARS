import os
import torch
import pandas as pd
import numpy as np
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    SUBMISSION_PATH,
    MODEL_NAME,
    PRETRAINED,
)
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet


def run_inference(load_cached_data=True):
    """
    Executes the inference pipeline on the test set.

    Steps:
    1. Loads test metadata.
    2. Initializes the test DataLoader (handling ROI caching).
    3. Loads the trained AsymmetricEfficientNet model.
    4. Iterates through the test set, applying Test-Time Augmentation (Original, HFlip, VFlip).
    5. Aggregates predictions and saves them to the submission CSV.

    Args:
        load_cached_data (bool): Whether to use cached ROI indices for the test set.
    """
    # 1. Setup
    seed_everything()
    device = get_device()
    print(f"Inference Device: {device}")

    # 2. Load Metadata
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    print(f"Loaded test metadata: {len(df_test)} samples")

    # 3. Get DataLoader
    # We pass None for train/val as we only need the test loader here
    _, _, test_loader = get_dataloaders(
        train_df=None, val_df=None, test_df=df_test, load_cached_data=load_cached_data
    )

    # 4. Load Model
    model = AsymmetricEfficientNet(model_name=MODEL_NAME, pretrained=PRETRAINED)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train the model first."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Model loaded successfully. Starting inference with TTA...")

    # 5. Inference Loop with TTA
    all_probs = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # TTA View 1: Original
            logits_orig = model(inputs)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA View 2: Horizontal Flip (dim 3 is Width)
            inputs_hflip = torch.flip(inputs, dims=[3])
            logits_hflip = model(inputs_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # TTA View 3: Vertical Flip (dim 2 is Height)
            inputs_vflip = torch.flip(inputs, dims=[2])
            logits_vflip = model(inputs_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average Predictions
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            # Collect results
            all_probs.extend(avg_probs.cpu().numpy().flatten().tolist())

    # 6. Generate Submission
    # Since shuffle=False in test_loader, the order matches df_test
    df_test["MGMT_value"] = all_probs

    # Format submission dataframe
    submission_df = df_test[["BraTS21ID", "MGMT_value"]]

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
