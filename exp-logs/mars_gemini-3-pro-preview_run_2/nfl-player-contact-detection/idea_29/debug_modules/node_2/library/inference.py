import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed
from library.data_processing import get_test_dataset
from library.model import SEARVN


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained SEA-RVN model.

    Steps:
    1. Load Test Dataset (uses pre-fitted scalers/encoders).
    2. Load Trained Model and Optimized Threshold.
    3. Perform Inference.
    4. Apply Threshold.
    5. Save submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached feature files if available.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # 2. Load Data
    # get_test_dataset handles feature generation, scaling, and encoding
    # It returns the PyTorch Dataset and the Series of contact_ids
    print("Loading test dataset...")
    test_ds, contact_ids = get_test_dataset(load_cached_data=load_cached_data)

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    print("Loading model...")
    model = SEARVN().to(device)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Load Threshold
    threshold_path = os.path.join(Config.WORKING_DIR, "best_threshold.npy")
    if os.path.exists(threshold_path):
        best_threshold = float(np.load(threshold_path))
        print(f"Loaded optimized threshold: {best_threshold}")
    else:
        print("Warning: Threshold file not found. Defaulting to 0.5.")
        best_threshold = 0.5

    # 5. Inference
    print("Running inference...")
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (test dataset returns X_kin_cont, X_kin_cat, X_vis)
            # Note: Test dataset __getitem__ does not return targets if y is None
            x_kin_cont, x_kin_cat, x_vis = [b.to(device) for b in batch]

            # Forward pass
            logits = model(x_kin_cont, x_kin_cat, x_vis)

            # Sigmoid to get probabilities
            probs = torch.sigmoid(logits).squeeze()

            # Handle single-element batch edge case where squeeze results in 0-d tensor
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            all_probs.append(probs.cpu().numpy())

    # Concatenate all batch results
    all_probs = np.concatenate(all_probs)

    # 6. Apply Threshold
    predictions = (all_probs >= best_threshold).astype(int)

    # 7. Create Submission DataFrame
    print("Creating submission file...")
    submission = pd.DataFrame({"contact_id": contact_ids, "contact": predictions})

    # 8. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(submission)}")
    print(
        f"Positive predictions: {submission['contact'].sum()} ({submission['contact'].mean():.4f} ratio)"
    )
