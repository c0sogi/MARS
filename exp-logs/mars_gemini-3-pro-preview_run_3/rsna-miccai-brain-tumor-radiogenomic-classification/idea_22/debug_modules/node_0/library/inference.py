import os
import torch
import pandas as pd
import numpy as np
from library.config import CACHE_DIR, SUBMISSION_PATH, SEED
from library.model import MSSHDNetwork
from library.data_loader import get_test_dataloader
from library.utils import get_device, seed_everything


def predict_and_submit(load_cached_data=True):
    """
    Loads the best trained model, runs inference on the test set,
    and generates the submission CSV file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup Environment
    seed_everything(SEED)
    device = get_device()

    # 2. Initialize Model
    print("Initializing model architecture...")
    model = MSSHDNetwork()
    model.to(device)

    # 3. Load Trained Weights
    model_path = os.path.join(CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model checkpoint not found at {model_path}. Please run training first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Load Test Data
    # The get_test_dataloader function handles caching internally via load_or_create_dataset
    print("Loading test data...")
    test_loader = get_test_dataloader(load_cached_data=load_cached_data)

    # 5. Run Inference
    all_ids = []
    all_probs = []

    print(f"Starting inference on {len(test_loader.dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            ids = batch_data["BraTS21ID"]  # List of strings (e.g., "00001")

            # Forward pass
            logits = model(images)

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_ids.extend(ids)
            all_probs.extend(probs)

    # 6. Format Submission
    # The competition expects BraTS21ID as integers (e.g., 1, 13, 15)
    # Our metadata pipeline uses 5-digit strings ("00001"). We convert them back.
    formatted_ids = [int(pid) for pid in all_ids]

    submission_df = pd.DataFrame({"BraTS21ID": formatted_ids, "MGMT_value": all_probs})

    # 7. Save Submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved successfully to {SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(submission_df.head())
