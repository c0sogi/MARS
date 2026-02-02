import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import PCCGNet
from library.utils import seed_everything


def run_inference():
    """
    Main function to run inference on the test set and generate the submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # 2. Load Data
    # We need the test metadata dataframe to get the Patient_Week IDs for the submission file
    # The loader returns tensors, not the string IDs.
    try:
        test_df = pd.read_csv(Config.TEST_CSV)
    except FileNotFoundError:
        print(f"Error: Test metadata file not found at {Config.TEST_CSV}")
        return

    # Get DataLoader (we only need the test loader)
    # get_dataloaders handles the stats calculation from train/val internally
    print("Initializing DataLoaders...")
    _, _, test_loader = get_dataloaders()

    # 3. Load Model
    print("Initializing PCCG-Net...")
    model = PCCGNet()

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_fvc_preds = []
    all_confidence_preds = []

    print(f"Predicting for {len(test_loader.dataset)} samples...")

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            # Move inputs to device
            axial = inputs["axial"].to(device)
            coronal = inputs["coronal"].to(device)
            tabular = inputs["tabular"].to(device)
            delta_week = inputs["delta_week"].to(device)
            base_fvc = inputs["base_fvc"].to(device)

            # Forward pass
            # Model returns [B, 2] -> (FVC, Confidence)
            preds = model(axial, coronal, tabular, delta_week, base_fvc)

            # Move to CPU and numpy
            preds_np = preds.cpu().numpy()

            # Append to lists
            all_fvc_preds.extend(preds_np[:, 0])
            all_confidence_preds.extend(preds_np[:, 1])

    # 5. Generate Submission
    # Ensure the number of predictions matches the metadata
    if len(all_fvc_preds) != len(test_df):
        print(
            f"CRITICAL ERROR: Prediction count ({len(all_fvc_preds)}) matches test set size ({len(test_df)}) mismatch."
        )
        # In a real scenario, we might raise an error, but here we proceed to save what we have
        # or truncate/pad if strictly necessary. Assuming correct loader behavior, this shouldn't happen.

    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": all_fvc_preds,
            "Confidence": all_confidence_preds,
        }
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_FILE}")

    # Print head for verification
    print("Submission Head:")
    print(submission.head())
