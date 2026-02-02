import os
import torch
import pandas as pd
import numpy as np
from library.config import Config, setup_reproducibility
from library.data import get_dataloaders
from library.model import SiameseEfficientNet


def predict_submission(load_cached=True):
    """
    Loads the best trained model, runs inference on the test set with
    Test-Time Augmentation (TTA), and generates the submission file.

    Args:
        load_cached (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup Reproducibility and Device
    setup_reproducibility(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 2. Prepare Data
    # We unpack the test loader and the corresponding IDs
    # train_loader and val_loader are ignored
    print("Loading test data...")
    _, _, test_loader, test_ids = get_dataloaders(load_cached=load_cached)

    # 3. Load Model
    print("Loading model...")
    model = SiameseEfficientNet()
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Ensure training has completed."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference with TTA
    print("Running inference with Test-Time Augmentation (Original + HFlip + VFlip)...")
    all_preds = []

    with torch.no_grad():
        for texture, context in test_loader:
            texture = texture.to(device)
            context = context.to(device)

            # --- Pass 1: Original ---
            logits_orig = model(texture, context)
            probs_orig = torch.sigmoid(logits_orig)

            # --- Pass 2: Horizontal Flip ---
            # Input is (B, C, H, W). Width is dim 3.
            tex_h = torch.flip(texture, dims=[3])
            ctx_h = torch.flip(context, dims=[3])
            logits_h = model(tex_h, ctx_h)
            probs_h = torch.sigmoid(logits_h)

            # --- Pass 3: Vertical Flip ---
            # Input is (B, C, H, W). Height is dim 2.
            tex_v = torch.flip(texture, dims=[2])
            ctx_v = torch.flip(context, dims=[2])
            logits_v = model(tex_v, ctx_v)
            probs_v = torch.sigmoid(logits_v)

            # --- Average Predictions ---
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Flatten and store
            all_preds.extend(avg_probs.cpu().numpy().flatten())

    # 5. Generate Submission File
    # Ensure the number of predictions matches the number of IDs
    if len(all_preds) != len(test_ids):
        print(
            f"Warning: Number of predictions ({len(all_preds)}) does not match number of test IDs ({len(test_ids)})."
        )

    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_preds})

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission generated successfully.")
    print(f"Saved to: {Config.SUBMISSION_PATH}")
    print(submission_df.head())
