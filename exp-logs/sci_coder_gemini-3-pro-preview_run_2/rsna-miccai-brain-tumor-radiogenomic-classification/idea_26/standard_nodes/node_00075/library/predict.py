import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet


def predict_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the trained model and Test-Time Augmentation (TTA).
    Saves the results to the submission file specified in Config.

    Args:
        load_cached_data (bool): Whether to use cached ROI data for the test set.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting inference on device: {device}")

    # 2. Load Model
    model = AsymmetricEfficientNet()
    model = model.to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_SAVE_PATH}. "
            "Please ensure training has completed successfully."
        )

    # Load weights
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Load Test Data
    # shuffle=False is critical for maintaining order, though we also track IDs explicitly
    test_loader = get_dataloader(
        "test",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=load_cached_data,
    )

    results = []

    print("Running inference with Test-Time Augmentation (Original + HFlip + VFlip)...")

    # 4. Inference Loop with TTA
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # --- TTA 1: Original ---
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # --- TTA 2: Horizontal Flip ---
            # Input shape is (B, C, H, W). W is dim 3.
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # --- TTA 3: Vertical Flip ---
            # Input shape is (B, C, H, W). H is dim 2.
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # --- Average Predictions ---
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            # Store results
            # Convert to numpy for storage
            batch_probs = avg_probs.cpu().numpy().flatten()

            # Handle IDs (Tensor to numpy)
            if isinstance(ids, torch.Tensor):
                batch_ids = ids.numpy()
            else:
                batch_ids = np.array(ids)

            for subject_id, prob in zip(batch_ids, batch_probs):
                results.append({"BraTS21ID": subject_id, "MGMT_value": prob})

    # 5. Generate Submission File
    df_submission = pd.DataFrame(results)

    # Ensure IDs are sorted for consistency
    df_submission = df_submission.sort_values("BraTS21ID")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission generated successfully with {len(df_submission)} entries.")
    print(f"Saved to: {Config.SUBMISSION_PATH}")
