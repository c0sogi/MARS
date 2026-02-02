import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from library.config import Config
from library.model import ResUNetPlusPlus
from library.dataset import DenoisingDataset
from library.utils import get_device


def predict_tiled(
    model, image_tensor, tile_size=Config.TILE_SIZE, overlap=Config.TILE_OVERLAP
):
    """
    Performs inference on a large image by splitting it into overlapping tiles.

    Args:
        model: The trained PyTorch model.
        image_tensor: Input image tensor of shape (C, H, W).
        tile_size: Size of the square tile.
        overlap: Overlap ratio (0.0 to 1.0).

    Returns:
        Tensor of shape (C, H, W) containing the prediction.
    """
    c, h, w = image_tensor.shape
    device = image_tensor.device

    stride = int(tile_size * (1 - overlap))

    # Calculate padding to ensure the image is covered by tiles
    pad_h = (tile_size - h % stride) % stride
    pad_w = (tile_size - w % stride) % stride

    # Also ensure image is at least the size of one tile
    if h < tile_size:
        pad_h += tile_size - h
    if w < tile_size:
        pad_w += tile_size - w

    # Pad image (Replicate padding to handle small images safely)
    padded_image = F.pad(
        image_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
    ).squeeze(0)
    p_h, p_w = padded_image.shape[1], padded_image.shape[2]

    # Output containers
    output_sum = torch.zeros((c, p_h, p_w), device=device)
    output_count = torch.zeros((c, p_h, p_w), device=device)

    # Generate tile coordinates
    y_starts = list(range(0, p_h - tile_size + 1, stride))
    x_starts = list(range(0, p_w - tile_size + 1, stride))

    # If the last tile doesn't reach the end exactly (due to integer stride), add a final tile aligned to the edge
    if y_starts[-1] + tile_size < p_h:
        y_starts.append(p_h - tile_size)
    if x_starts[-1] + tile_size < p_w:
        x_starts.append(p_w - tile_size)

    # Iterate over tiles
    for y in y_starts:
        for x in x_starts:
            # Extract tile
            tile = padded_image[:, y : y + tile_size, x : x + tile_size].unsqueeze(
                0
            )  # (1, C, H, W)

            # Inference
            with torch.no_grad():
                # Model returns noise residual
                pred_tile = model(tile)

            # Accumulate
            output_sum[:, y : y + tile_size, x : x + tile_size] += pred_tile.squeeze(0)
            output_count[:, y : y + tile_size, x : x + tile_size] += 1.0

    # Average overlapping areas
    output = output_sum / output_count

    # Crop back to original size
    return output[:, :h, :w]


def apply_tta(model, image_tensor):
    """
    Applies Test-Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model: Trained model.
        image_tensor: Input image tensor (C, H, W).

    Returns:
        Averaged prediction tensor.
    """
    preds = []

    # 1. Original
    pred_orig = predict_tiled(model, image_tensor)
    preds.append(pred_orig)

    if Config.TTA_ENABLED:
        # 2. Horizontal Flip
        img_hflip = torch.flip(image_tensor, [2])
        pred_hflip = predict_tiled(model, img_hflip)
        preds.append(torch.flip(pred_hflip, [2]))

        # 3. Vertical Flip
        img_vflip = torch.flip(image_tensor, [1])
        pred_vflip = predict_tiled(model, img_vflip)
        preds.append(torch.flip(pred_vflip, [1]))

    # Average predictions
    return torch.stack(preds).mean(dim=0)


def generate_submission():
    """
    Main function to generate the submission file.
    Loads model, runs inference on test set, formats output, and saves CSV.
    """
    print("Generating submission...")

    # 1. Setup
    device = get_device()
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Model
    model = ResUNetPlusPlus().to(device)
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print(
            f"Warning: Model file not found at {Config.MODEL_PATH}. Using random initialization (likely to fail)."
        )

    model.eval()

    # 3. Load Test Data
    # Batch size 1 because we handle variable image sizes manually
    test_dataset = DenoisingDataset(
        metadata_file=Config.TEST_METADATA,
        mode="test",
        load_cached_data=Config.LOAD_CACHED_DATA,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # 4. Inference Loop
    submission_ids = []
    submission_values = []

    print(f"Processing {len(test_dataset)} test images...")

    with torch.no_grad():
        for i, (noisy_img, img_id_batch) in enumerate(test_loader):
            img_id = img_id_batch[0]
            noisy_img = noisy_img.squeeze(0).to(
                device
            )  # Remove batch dim: (1, H, W) -> (C, H, W)

            # Predict Noise Residual
            pred_residual = apply_tta(model, noisy_img)

            # Denoise: Clean = Noisy - Residual
            pred_clean = noisy_img - pred_residual

            # Clip values to valid range [0, 1]
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            # Convert to numpy
            pred_clean_np = pred_clean.cpu().numpy().squeeze(0)  # (H, W)

            # Format for Submission
            h, w = pred_clean_np.shape

            # Create coordinate grids
            # Rows are 1-based, Cols are 1-based
            rows, cols = np.indices((h, w))
            rows = rows + 1
            cols = cols + 1

            # Flatten arrays
            flat_vals = pred_clean_np.flatten()
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()

            # Generate IDs: "imageid_row_col"
            # Using list comprehension is reasonably fast for this scale
            current_ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

            submission_ids.extend(current_ids)
            submission_values.extend(flat_vals)

            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} images")

    # 5. Save Submission
    print("Constructing DataFrame...")
    df_sub = pd.DataFrame({"id": submission_ids, "value": submission_values})

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
