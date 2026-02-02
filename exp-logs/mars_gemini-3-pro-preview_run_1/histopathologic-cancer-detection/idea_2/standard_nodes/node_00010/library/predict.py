import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ModifiedDenseNet
from library.dataset import get_dataloaders


def predict_submission(debug=False):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Saves the result to submission.csv.

    Args:
        debug (bool): If True, runs on a subset of the data for debugging purposes.
    """
    print("Initializing prediction pipeline...")

    # 1. Setup Device
    device = torch.device(Config.DEVICE)

    # 2. Load Model Architecture
    # We set pretrained=False because we are loading our own trained weights.
    # This avoids downloading ImageNet weights unnecessarily.
    model = ModifiedDenseNet(pretrained=False)
    model = model.to(device)

    # 3. Load Trained Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    print(f"Loading model weights from {Config.MODEL_PATH}")
    # Load state dictionary
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    # 4. Load Test Data
    # get_dataloaders returns (train, val, test). We only need the test loader.
    _, _, test_loader = get_dataloaders(debug=debug)

    ids_list = []
    preds_list = []

    print(
        f"Processing {len(test_loader.dataset)} test images with TTA ({Config.TTA_STEPS} views)..."
    )

    # 5. Inference Loop with TTA
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            # images shape: (B, 3, 48, 48)

            # --- Test Time Augmentation (TTA) ---

            # View 1: Original
            out1 = model(images)
            prob1 = torch.sigmoid(out1)

            # View 2: Horizontal Flip
            # Flip along width (dim 3)
            img_hflip = torch.flip(images, dims=[3])
            out2 = model(img_hflip)
            prob2 = torch.sigmoid(out2)

            # View 3: Vertical Flip
            # Flip along height (dim 2)
            img_vflip = torch.flip(images, dims=[2])
            out3 = model(img_vflip)
            prob3 = torch.sigmoid(out3)

            # View 4: Rotate 90 degrees
            # Rotate in the spatial plane (dims 2 and 3)
            img_rot90 = torch.rot90(images, k=1, dims=[2, 3])
            out4 = model(img_rot90)
            prob4 = torch.sigmoid(out4)

            # Average Probabilities
            # Averaging probabilities (not logits) is the standard TTA approach for classification
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            # --- Collection ---

            # Flatten to 1D array and extend lists
            preds_list.extend(avg_prob.cpu().numpy().flatten())
            ids_list.extend(ids)

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": ids_list, "label": preds_list})

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Prediction generation complete.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(submission_df)}")
