import os
import torch
import pandas as pd
import numpy as np

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    BATCH_SIZE,
)
from library.utils import set_seed, rle_encode
from library.dataset import get_dataloader
from library.model import UNet


def predict_and_submit(debug=False):
    """
    Loads the trained model, performs inference on the test set,
    encodes the predictions using RLE, and saves the submission file.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
    """
    # 1. Setup
    set_seed()
    print(f"Starting inference on device: {DEVICE}")

    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_SAVE_PATH}. Please train the model first."
        )

    # 2. Load Model
    model = UNet().to(DEVICE)
    # Load state dict
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Test Data
    # shuffle=False is guaranteed by get_dataloader for split='test'
    test_loader = get_dataloader("test", batch_size=BATCH_SIZE, debug=debug)

    # Access the underlying dataframe to map indices to record_ids
    test_df = test_loader.dataset.df

    submission_data = []

    print(f"Processing {len(test_loader)} batches...")

    # 4. Inference Loop
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(DEVICE)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid and Threshold
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()

            # Retrieve record_ids for the current batch
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_record_ids = test_df.iloc[start_idx:end_idx]["record_id"].values

            # Encode each mask in the batch
            for j, pred_mask in enumerate(preds):
                # pred_mask shape is (1, H, W), squeeze to (H, W)
                mask_2d = pred_mask.squeeze(0)

                rle_str = rle_encode(mask_2d)
                record_id = str(batch_record_ids[j])

                submission_data.append(
                    {"record_id": record_id, "encoded_pixels": rle_str}
                )

    # 5. Save Submission
    df_sub = pd.DataFrame(submission_data)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE_PATH), exist_ok=True)

    df_sub.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_FILE_PATH}")
    print(f"Total records processed: {len(df_sub)}")
