import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SpatialDifferenceCNN


def predict_test_set(debug: bool = False):
    """
    Generates predictions for the test set using the trained SpatialDifferenceCNN model
    and saves the result to a submission CSV file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data for debugging.
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Prepare Data
    # We only need the test_loader (3rd element returned by get_dataloaders)
    _, _, test_loader = get_dataloaders(debug=debug)

    # 3. Load Model
    model = SpatialDifferenceCNN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_SAVE_PATH}. "
            "Please train the model before running inference."
        )

    # Load weights
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from {Config.MODEL_SAVE_PATH}")
    print("Starting inference on test set...")

    # 4. Inference Loop
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass to get logits
            logits = model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Flatten to 1D array and move to CPU
            probs_np = probs.cpu().numpy().flatten()

            all_ids.extend(ids)
            all_probs.extend(probs_np)

    # 5. Save Submission
    submission_df = pd.DataFrame({"id": all_ids, "target": all_probs})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    # Print head without formatting to comply with full precision requirement (though mostly relevant for floats)
    print(f"First 5 predictions:\n{submission_df.head()}")

    return submission_df
