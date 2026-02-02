import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SiameseEfficientNet


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the trained Siamese EfficientNet model.
    Applies Test-Time Augmentation (TTA) (Original + HFlip + VFlip) for robust estimation.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed test data from cache.
                                 If False or cache missing, processes data from scratch.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    # We only need the test loader. get_dataloaders handles the caching logic internally.
    print("Initializing data loaders...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    print(f"Loading model architecture: {Config.MODEL_NAME}...")
    model = SiameseEfficientNet()
    model.to(device)

    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading weights from {Config.MODEL_PATH}...")
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model checkpoint not found at {Config.MODEL_PATH}. Predictions will be random."
        )

    model.eval()

    # 4. Inference Loop with TTA
    results = []
    print("Starting inference with Test-Time Augmentation (Original, HFlip, VFlip)...")

    with torch.no_grad():
        for view_bulk, view_core, subject_ids in test_loader:
            view_bulk = view_bulk.to(device)
            view_core = view_core.to(device)

            # --- Pass 1: Original ---
            logits_1 = model(view_bulk, view_core)
            probs_1 = torch.sigmoid(logits_1)

            # --- Pass 2: Horizontal Flip ---
            # TF.hflip works on (..., H, W) tensors, so it handles batches correctly
            vb_h = TF.hflip(view_bulk)
            vc_h = TF.hflip(view_core)
            logits_2 = model(vb_h, vc_h)
            probs_2 = torch.sigmoid(logits_2)

            # --- Pass 3: Vertical Flip ---
            vb_v = TF.vflip(view_bulk)
            vc_v = TF.vflip(view_core)
            logits_3 = model(vb_v, vc_v)
            probs_3 = torch.sigmoid(logits_3)

            # --- Average Predictions ---
            avg_probs = (probs_1 + probs_2 + probs_3) / 3.0

            # Collect results
            # Move to CPU and flatten
            batch_probs = avg_probs.cpu().numpy().flatten()
            batch_ids = subject_ids.numpy().flatten()

            for pid, prob in zip(batch_ids, batch_probs):
                # Format ID as 5-digit string (e.g., 1 -> "00001")
                results.append({"BraTS21ID": f"{int(pid):05d}", "MGMT_value": prob})

    # 5. Save Submission
    df_sub = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission generated successfully.")
    print(f"Saved to: {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(df_sub)}")
    print("First 5 rows:")
    print(df_sub.head())
