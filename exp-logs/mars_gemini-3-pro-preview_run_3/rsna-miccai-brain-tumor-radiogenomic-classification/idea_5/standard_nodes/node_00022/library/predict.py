import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import VFPNet
from library.dataset import get_dataloader


def generate_submission(weights_path=None, batch_size=None, device=None):
    """
    Loads the trained model, performs inference on the test set, and generates
    the submission.csv file.

    Args:
        weights_path (str, optional): Path to the model checkpoint. Defaults to best_model.pth.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        device (str, optional): Device to run inference on. Defaults to Config.DEVICE.
    """
    # 1. Setup Configuration
    if weights_path is None:
        weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if device is None:
        device = Config.DEVICE

    print(f"Generating submission using model at: {weights_path}")
    print(f"Device: {device}")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Initialize Model
    # We must match the architecture used in training
    model = VFPNet(num_classes=1, pretrained=False)

    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 3. Load Test Data
    # get_dataloader handles the caching and dataset creation
    # For inference, shuffle must be False
    test_loader = get_dataloader(
        "test", batch_size=batch_size, shuffle=False, load_cached_data=True
    )

    # 4. Inference Loop
    all_ids = []
    all_probs = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass (returns logits)
            logits = model(inputs)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Flatten to 1D array
            probs = probs.cpu().numpy().flatten()

            # Collect results
            all_ids.extend(ids)
            all_probs.extend(probs)

    # 5. Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_probs})

    # Ensure BraTS21ID is formatted correctly (though logic preserves input format)
    # The sample submission expects IDs like '00001', '00013', etc.
    # The metadata/dataset pipeline preserves these as strings.

    # 6. Save to CSV
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(submission_df.head())
