import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    TEST_CSV,
    SUBMISSION_DIR,
    CHECKPOINT_DIR,
    PATCH_SIZE,
    OVERLAP_RATIO,
)
from library.utils import load_checkpoint
from library.model import ICResUNet
from library.dataset import DenoisingDataset
from library.train import predict_tiled


def predict_with_tta(model, noisy_tensor):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from: Original, H-Flip, V-Flip, and Rot90.

    Args:
        model: Trained neural network.
        noisy_tensor: Input tensor of shape (C, H, W) on the correct device.

    Returns:
        avg_pred: Averaged denoised tensor of shape (C, H, W).
    """
    preds = []

    # 1. Original
    p_orig = predict_tiled(model, noisy_tensor)
    preds.append(p_orig)

    # 2. Horizontal Flip
    img_hf = torch.flip(noisy_tensor, [-1])
    p_hf = predict_tiled(model, img_hf)
    preds.append(torch.flip(p_hf, [-1]))

    # 3. Vertical Flip
    img_vf = torch.flip(noisy_tensor, [-2])
    p_vf = predict_tiled(model, img_vf)
    preds.append(torch.flip(p_vf, [-2]))

    # 4. Rotate 90 degrees (k=1)
    # rot90 dims: (..., H, W) -> (-2, -1)
    img_r90 = torch.rot90(noisy_tensor, 1, [-2, -1])
    p_r90 = predict_tiled(model, img_r90)
    # Inverse is rot90 with k=3
    preds.append(torch.rot90(p_r90, 3, [-2, -1]))

    # Stack and Average
    # Stack shape: (4, C, H, W)
    avg_pred = torch.stack(preds).mean(dim=0)

    return avg_pred


def generate_submission(checkpoint_name="best_model.pth"):
    """
    Generates the submission CSV file for the test set.

    Args:
        checkpoint_name (str): Name of the checkpoint file to load.
    """
    print(f"Generating submission using checkpoint: {checkpoint_name}")

    # 1. Load Model
    model = ICResUNet().to(DEVICE)
    epoch, loss = load_checkpoint(model, filename=checkpoint_name)
    if epoch == 0 and loss == float("inf"):
        print("Warning: Checkpoint not found or failed to load. Using random weights.")
    else:
        print(f"Loaded model from epoch {epoch} with loss {loss:.6f}")

    model.eval()

    # 2. Load Test Data
    # Batch size must be 1 because images have different sizes
    test_dataset = DenoisingDataset(mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    results_dfs = []

    print(f"Processing {len(test_dataset)} test images...")

    with torch.no_grad():
        for i, (noisy_imgs, img_ids) in enumerate(test_loader):
            # Unpack batch (size 1)
            noisy_tensor = noisy_imgs[0].to(DEVICE)  # (C, H, W)
            img_id = img_ids[0]

            # Predict
            clean_tensor = predict_with_tta(model, noisy_tensor)

            # Move to CPU and numpy
            # Shape (1, H, W) -> (H, W)
            clean_img = clean_tensor.squeeze(0).cpu().numpy()
            H, W = clean_img.shape

            # Format for Submission
            # Create coordinate grids (1-based indexing)
            # rows: 1..H, cols: 1..W
            rows, cols = np.indices((H, W))
            rows = rows.flatten() + 1
            cols = cols.flatten() + 1
            values = clean_img.flatten()

            # Create DataFrame for this image
            # Vectorized string creation is faster than list comprehension
            df_img = pd.DataFrame(
                {
                    "row": rows,
                    "col": cols,
                    "value": values,
                }
            )

            # Construct ID: {img_id}_{row}_{col}
            # We use a temporary prefix to speed up concatenation
            df_img["id"] = (
                f"{img_id}_"
                + df_img["row"].astype(str)
                + "_"
                + df_img["col"].astype(str)
            )

            # Keep only required columns
            df_img = df_img[["id", "value"]]
            results_dfs.append(df_img)

            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} images")

    # 3. Concatenate and Save
    print("Concatenating results...")
    final_df = pd.concat(results_dfs, ignore_index=True)

    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    print(f"Saving submission to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("Submission generation complete.")
