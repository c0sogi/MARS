import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.dataset import load_dataset
from library.model import Stabilized25DNet


def generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Generates the submission file for the test set.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed numpy arrays.
        batch_size (int): Batch size for inference.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference device: {device}")

    # 2. Load Test Data
    # load_dataset handles loading metadata, processing images (loading, normalizing, sampling, stacking),
    # and caching the result to disk.
    test_dataset = load_dataset("test", load_cached_data=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model
    model = Stabilized25DNet().to(device)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_probs = []

    print("Starting inference...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and store
            all_probs.extend(probs.cpu().numpy().flatten())

    # 5. Generate Submission File
    # Retrieve IDs from the dataset (order is preserved by DataLoader with shuffle=False)
    patient_ids = test_dataset.ids

    # Create DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": patient_ids, "MGMT_value": all_probs})

    # Ensure output directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    submission_path = os.path.join(submission_dir, "submission.csv")

    # Save to CSV
    # The task description specifies the format: BraTS21ID,MGMT_value
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
