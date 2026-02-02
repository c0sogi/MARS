import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.model import CoRes2NetUNet
from library.utils import load_checkpoint
from library.dataset import DenoisingDataset


def predict_tiled(model, image, patch_size=Config.PATCH_SIZE, overlap=0.5):
    """
    Performs sliding window inference on a large image.

    Args:
        model: The trained PyTorch model.
        image: Input tensor of shape [1, 1, H, W].
        patch_size: Size of the square patch.
        overlap: Overlap ratio between patches (0.0 to 1.0).

    Returns:
        Tensor of shape [1, 1, H, W] containing the prediction.
    """
    b, c, h, w = image.shape
    stride = int(patch_size * (1 - overlap))

    # Calculate padding to ensure the sliding window covers the entire image
    # We want the last patch to start at index k*stride and cover the end of the image.
    # So, k*stride + patch_size >= h
    if h < patch_size:
        pad_h = patch_size - h
    else:
        k_h = math.ceil((h - patch_size) / stride)
        target_h = patch_size + k_h * stride
        pad_h = target_h - h

    if w < patch_size:
        pad_w = patch_size - w
    else:
        k_w = math.ceil((w - patch_size) / stride)
        target_w = patch_size + k_w * stride
        pad_w = target_w - w

    # Pad image (Reflect padding reduces border artifacts)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    ph, pw = image_padded.shape[2], image_padded.shape[3]

    # Accumulators
    output_sum = torch.zeros_like(image_padded)
    output_count = torch.zeros_like(image_padded)

    # Sliding Window Loop
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            # Extract patch
            patch = image_padded[:, :, y : y + patch_size, x : x + patch_size]

            # Predict
            with torch.no_grad():
                pred_patch = model(patch)

            # Accumulate
            output_sum[:, :, y : y + patch_size, x : x + patch_size] += pred_patch
            output_count[:, :, y : y + patch_size, x : x + patch_size] += 1.0

    # Average overlapping regions
    # Avoid division by zero (though logic guarantees count >= 1)
    output = output_sum / output_count

    # Crop back to original size
    return output[:, :, :h, :w]


def predict_tta(model, image, patch_size=Config.PATCH_SIZE):
    """
    Performs Test-Time Augmentation (TTA) using 8 geometric transformations.

    Args:
        model: The trained PyTorch model.
        image: Input tensor of shape [1, 1, H, W].
        patch_size: Patch size for tiled inference.

    Returns:
        Tensor of shape [1, 1, H, W] containing the averaged prediction.
    """
    preds = []

    # Iterate through 8 combinations: 2 Flips * 4 Rotations
    for flip in [False, True]:
        for k in [0, 1, 2, 3]:
            # --- Transform ---
            img_aug = image.clone()

            if flip:
                # Flip Horizontal (dim 3)
                img_aug = torch.flip(img_aug, [3])

            if k > 0:
                # Rotate 90 degrees k times
                img_aug = torch.rot90(img_aug, k, [2, 3])

            # --- Inference ---
            pred_aug = predict_tiled(model, img_aug, patch_size)

            # --- Inverse Transform ---
            # Reverse order of operations: Un-Rotate then Un-Flip
            if k > 0:
                pred_aug = torch.rot90(pred_aug, -k, [2, 3])

            if flip:
                pred_aug = torch.flip(pred_aug, [3])

            preds.append(pred_aug)

    # Average all predictions
    return torch.stack(preds).mean(dim=0)


def create_submission(checkpoint_path=Config.MODEL_CHECKPOINT_PATH):
    """
    Generates predictions for the test set and creates the submission CSV.
    """
    device = Config.DEVICE
    print(f"Starting inference on device: {device}")

    # 1. Load Model
    model = CoRes2NetUNet().to(device)
    try:
        epoch, score = load_checkpoint(model, checkpoint_path, device=device)
        print(f"Loaded model checkpoint from epoch {epoch} (Val RMSE: {score})")
    except FileNotFoundError:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights (for debugging only)."
        )

    model.eval()

    # 2. Load Test Data
    # DenoisingDataset in 'test' mode returns (noisy_tensor [1, H, W], img_id)
    test_dataset = DenoisingDataset(mode="test", load_cached_data=True)
    print(f"Found {len(test_dataset)} test images.")

    # 3. Prepare Submission File
    submission_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Write Header
    with open(submission_path, "w") as f:
        f.write("id,value\n")

    print("Generating predictions...")

    # 4. Inference Loop
    for i in range(len(test_dataset)):
        noisy_t, img_id = test_dataset[i]

        # Add batch dimension: [1, H, W] -> [1, 1, H, W]
        input_tensor = noisy_t.unsqueeze(0).to(device)

        with torch.no_grad():
            # Model predicts noise residual
            pred_noise = predict_tta(model, input_tensor)

            # Clean = Noisy - Noise
            pred_clean = input_tensor - pred_noise

            # Clamp to valid range
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

        # Move to CPU and numpy
        pred_clean_np = pred_clean.squeeze().cpu().numpy()  # [H, W]
        h, w = pred_clean_np.shape

        # 5. Format Output (Melt to Pixels)
        # Create ID strings: {img_id}_{row}_{col}
        # Note: 1-based indexing for rows and cols

        # Vectorized ID generation
        r_range = np.arange(1, h + 1)
        c_range = np.arange(1, w + 1)

        # Repeat rows for each col (1,1,1... 2,2,2...)
        rows = np.repeat(r_range, w)
        # Tile cols for each row (1,2,3... 1,2,3...)
        cols = np.tile(c_range, h)

        flat_ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]
        flat_values = pred_clean_np.flatten()

        # Create DataFrame chunk
        df_chunk = pd.DataFrame({"id": flat_ids, "value": flat_values})

        # Append to CSV
        df_chunk.to_csv(submission_path, mode="a", header=False, index=False)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(test_dataset)} images.")

    print(f"Submission generation complete. Saved to {submission_path}")
