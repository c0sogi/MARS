import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device, seed_everything
from library.model import BraTS25DNet
from library.data_loader import get_test_loader


def run_inference():
    """
    Loads the trained model, performs inference on the test set,
    and generates the submission CSV file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Inference Device: {device}")

    # 2. Data Loader
    # load_cached=True allows utilizing pre-processed numpy arrays if they exist
    test_loader = get_test_loader(load_cached=True)
    print(f"Test Data Loaded. Batches: {len(test_loader)}")

    # 3. Model Initialization
    model = BraTS25DNet()
    model.to(device)

    # 4. Load Weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}...")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    # 5. Inference Loop
    model.eval()
    all_ids = []
    all_probs = []

    print("Starting inference...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Flatten predictions and extend lists
            all_probs.extend(probs.flatten().tolist())
            all_ids.extend(ids)

    # 6. Generate Submission
    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(submission_df.head())

    return submission_df
