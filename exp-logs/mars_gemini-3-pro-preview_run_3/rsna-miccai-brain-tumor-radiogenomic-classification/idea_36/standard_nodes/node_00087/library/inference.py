import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.utils import seed_everything, get_device
from library.data import load_dataset_split, MGMTDataset
from library.model import Stacked25DNet


def predict_submission(
    model_path="./working/idea_opt/best_model.pth",
    save_path="./submission/submission.csv",
    batch_size=16,
    load_cached_data=True,
    device=None,
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    # 1. Setup
    seed_everything(42)
    if device is None:
        device = get_device()

    print(f"Running inference on device: {device}")

    # 2. Load Test Data
    try:
        X, y_test, ids = load_dataset_split("test", load_cached_data=load_cached_data)
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading test data: {e}")
        return

    # Handle case with no test data
    if len(ids) == 0:
        print("Warning: No test data found. Generating empty submission.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(columns=["BraTS21ID", "MGMT_value"]).to_csv(save_path, index=False)
        return

    # Create Dataset and Loader
    test_dataset = MGMTDataset(X, y_test, ids=ids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Load Model
    model = Stacked25DNet(model_name="efficientnet_b0", pretrained=False)

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print(f"Starting inference on {len(test_dataset)} samples...")
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)

            # Forward pass
            logits = model(x)

            # Apply Sigmoid to get probabilities (logits -> [0, 1])
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    # 5. Save Submission
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create DataFrame
    # ids are strings (e.g., "00001") as loaded from metadata
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": all_probs})

    # Save to CSV
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print("First 5 predictions:")
    print(submission_df.head())
