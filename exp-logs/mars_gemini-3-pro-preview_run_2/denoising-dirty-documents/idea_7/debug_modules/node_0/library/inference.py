import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.model import DS_AG_CAC_ResUNet
from library.dataset import DenoisingDataset
from library.utils import get_device, seed_everything


def get_gaussian_window(size, sigma=None, device="cpu"):
    """
    Generates a 2D Gaussian window for weighted patch merging.

    Args:
        size (int): The size of the window (size x size).
        sigma (float): Standard deviation of the Gaussian.
        device (torch.device): Device to create the tensor on.

    Returns:
        torch.Tensor: 2D Gaussian window of shape (1, 1, size, size).
    """
    if sigma is None:
        # 3 standard deviations cover the window effectively
        sigma = size / 6

    coords = torch.arange(size, device=device).float() - (size - 1) / 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    dist_sq = x**2 + y**2
    window = torch.exp(-dist_sq / (2 * sigma**2))

    return window.view(1, 1, size, size)


def apply_tta(model, x):
    """
    Applies Test-Time Augmentation (TTA) using 8 geometric transformations.

    Args:
        model (nn.Module): The trained model.
        x (torch.Tensor): Input patch of shape (1, C, H, W).

    Returns:
        torch.Tensor: Averaged prediction from all augmented views.
    """
    # Remove batch dim for processing: (C, H, W)
    x_s = x.squeeze(0)

    # --- 1. Forward Transforms ---
    aug_list = []

    # 0. Original
    aug_list.append(x_s)
    # 1. Rot90 (k=1)
    aug_list.append(torch.rot90(x_s, 1, [1, 2]))
    # 2. Rot180 (k=2)
    aug_list.append(torch.rot90(x_s, 2, [1, 2]))
    # 3. Rot270 (k=3)
    aug_list.append(torch.rot90(x_s, 3, [1, 2]))

    # Flip Horizontal (dim 2 is Width in C,H,W)
    x_flip = torch.flip(x_s, [2])

    # 4. Flip
    aug_list.append(x_flip)
    # 5. Flip + Rot90
    aug_list.append(torch.rot90(x_flip, 1, [1, 2]))
    # 6. Flip + Rot180
    aug_list.append(torch.rot90(x_flip, 2, [1, 2]))
    # 7. Flip + Rot270
    aug_list.append(torch.rot90(x_flip, 3, [1, 2]))

    # Stack into batch: (8, C, H, W)
    batch = torch.stack(aug_list)

    # --- 2. Inference ---
    with torch.no_grad():
        # Model returns list [final, aux...], we only want final
        preds = model(batch)
        if isinstance(preds, list):
            preds = preds[0]

    # --- 3. Inverse Transforms ---
    deaug_list = []

    # 0. Original
    deaug_list.append(preds[0])
    # 1. Rot90 inv -> Rot270 (k=3)
    deaug_list.append(torch.rot90(preds[1], 3, [1, 2]))
    # 2. Rot180 inv -> Rot180 (k=2)
    deaug_list.append(torch.rot90(preds[2], 2, [1, 2]))
    # 3. Rot270 inv -> Rot90 (k=1)
    deaug_list.append(torch.rot90(preds[3], 1, [1, 2]))

    # 4. Flip inv -> Flip
    deaug_list.append(torch.flip(preds[4], [2]))

    # For combination Flip + Rot(k):
    # y = Rot(Flip(x))
    # x = Flip(Rot_inv(y))

    # 5. (Flip + Rot90) inv -> Flip(Rot270)
    deaug_list.append(torch.flip(torch.rot90(preds[5], 3, [1, 2]), [2]))
    # 6. (Flip + Rot180) inv -> Flip(Rot180)
    deaug_list.append(torch.flip(torch.rot90(preds[6], 2, [1, 2]), [2]))
    # 7. (Flip + Rot270) inv -> Flip(Rot90)
    deaug_list.append(torch.flip(torch.rot90(preds[7], 1, [1, 2]), [2]))

    # --- 4. Aggregate ---
    # Average the predictions
    mean_pred = torch.stack(deaug_list).mean(dim=0, keepdim=True)
    return mean_pred


def predict_tiled(model, image, device):
    """
    Performs sliding window inference with Gaussian weighting and TTA.

    Args:
        model (nn.Module): Trained model.
        image (torch.Tensor): Full size image tensor (1, C, H, W).
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: The cleaned image prediction (1, C, H, W).
    """
    patch_size = Config.PATCH_SIZE
    overlap = Config.TILE_OVERLAP
    stride = int(patch_size * (1 - overlap))

    b, c, h, w = image.shape

    # Calculate padding to ensure full coverage and stride alignment
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride

    # Ensure dimensions are at least patch_size
    if h < patch_size:
        pad_h += patch_size - h
    if w < patch_size:
        pad_w += patch_size - w

    # Apply reflection padding (left, right, top, bottom)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    new_h, new_w = image_padded.shape[2:]

    # Initialize accumulators
    prediction_map = torch.zeros((1, 1, new_h, new_w), device=device)
    weight_map = torch.zeros((1, 1, new_h, new_w), device=device)

    # Generate Gaussian window
    window = get_gaussian_window(patch_size, device=device)

    # Sliding Window Loop
    for y in range(0, new_h - patch_size + 1, stride):
        for x in range(0, new_w - patch_size + 1, stride):
            patch = image_padded[:, :, y : y + patch_size, x : x + patch_size]

            # Predict Noise Residual
            if Config.USE_TTA:
                pred_noise = apply_tta(model, patch)
            else:
                with torch.no_grad():
                    preds = model(patch)
                    pred_noise = preds[0] if isinstance(preds, list) else preds

            # Accumulate weighted predictions
            prediction_map[:, :, y : y + patch_size, x : x + patch_size] += (
                pred_noise * window
            )
            weight_map[:, :, y : y + patch_size, x : x + patch_size] += window

    # Normalize by weights
    prediction_map /= weight_map + 1e-8

    # Crop back to original size
    pred_noise_cropped = prediction_map[:, :, :h, :w]

    # Calculate Clean Image: Clean = Input - Noise
    clean_pred = image - pred_noise_cropped
    clean_pred = torch.clamp(clean_pred, 0, 1)

    return clean_pred


def generate_submission(
    model_path=Config.MODEL_CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Generates the submission file for the test dataset.

    Args:
        model_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the submission CSV.
    """
    seed_everything()
    device = get_device()

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = DS_AG_CAC_ResUNet().to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random initialization (for debugging only)."
        )

    model.eval()

    # 2. Load Test Data
    print("Initializing Test Loader...")
    # mode='test' ensures full images are loaded without cropping
    test_dataset = DenoisingDataset(Config.TEST_METADATA_PATH, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # 3. Prepare Submission File
    # Write header first
    with open(output_path, "w") as f:
        f.write("id,value\n")

    print(f"Starting inference on {len(test_dataset)} images...")

    # 4. Inference Loop
    for noisy_img, img_id_tuple in test_loader:
        img_id = img_id_tuple[0]  # Extract string from tuple
        noisy_img = noisy_img.to(device)

        # Run Tiled Inference with TTA
        clean_pred = predict_tiled(model, noisy_img, device)

        # Post-process
        clean_np = clean_pred.squeeze().cpu().numpy()  # (H, W)

        # 5. Format Output
        h, w = clean_np.shape

        # Create coordinate grids (1-based indexing)
        rows, cols = np.indices((h, w))
        rows += 1
        cols += 1

        # Flatten arrays
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        vals_flat = clean_np.flatten()

        # Create DataFrame chunk for efficient string manipulation
        df_chunk = pd.DataFrame({"row": rows_flat, "col": cols_flat, "val": vals_flat})

        # Construct ID: image_row_col
        # Using vectorization is faster than list comprehension
        df_chunk["id"] = (
            f"{img_id}_"
            + df_chunk["row"].astype(str)
            + "_"
            + df_chunk["col"].astype(str)
        )

        # Append to CSV
        df_chunk[["id", "val"]].to_csv(output_path, mode="a", header=False, index=False)

    print(f"Submission generation complete. Saved to {output_path}")
