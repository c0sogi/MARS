import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import UNet
from library.dataset import get_test_dataset
from library.utils import seed_everything


def apply_tta(model, x, device):
    """
    Applies D4 Group Test-Time Augmentation (8 views) to a single image tensor.

    Args:
        model: The loaded PyTorch model.
        x: Input tensor of shape (1, 1, H, W).
        device: Computation device.

    Returns:
        torch.Tensor: The averaged prediction of shape (1, 1, H, W).
    """
    # Generate 8 views
    # 1. Original
    x1 = x
    # 2. Rot90 (k=1)
    x2 = torch.rot90(x, 1, [2, 3])
    # 3. Rot180 (k=2)
    x3 = torch.rot90(x, 2, [2, 3])
    # 4. Rot270 (k=3)
    x4 = torch.rot90(x, 3, [2, 3])
    # 5. Flip Horizontal
    x5 = torch.flip(x, [3])
    # 6. Flip H + Rot90
    x6 = torch.rot90(x5, 1, [2, 3])
    # 7. Flip H + Rot180
    x7 = torch.rot90(x5, 2, [2, 3])
    # 8. Flip H + Rot270
    x8 = torch.rot90(x5, 3, [2, 3])

    # Check dimensions to handle non-square images
    height, width = x.shape[2], x.shape[3]

    if height == width:
        # Batch inputs for efficiency
        batch = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=0)

        # Predict
        with torch.no_grad():
            preds = model(batch.to(device))

        # Split predictions
        p1, p2, p3, p4, p5, p6, p7, p8 = torch.chunk(preds, 8, dim=0)
    else:
        # Split into dimension-compatible batches
        # Group 1: Preserves shape (H, W) -> x1, x3, x5, x7
        batch1 = torch.cat([x1, x3, x5, x7], dim=0)
        with torch.no_grad():
            preds1 = model(batch1.to(device))
        p1, p3, p5, p7 = torch.chunk(preds1, 4, dim=0)

        # Group 2: Swaps shape to (W, H) -> x2, x4, x6, x8
        batch2 = torch.cat([x2, x4, x6, x8], dim=0)
        with torch.no_grad():
            preds2 = model(batch2.to(device))
        p2, p4, p6, p8 = torch.chunk(preds2, 4, dim=0)

    # Inverse Transform to align with Original view
    # 1. Original
    y1 = p1
    # 2. Rot90 -> Inverse is Rot270 (k=3)
    y2 = torch.rot90(p2, 3, [2, 3])
    # 3. Rot180 -> Inverse is Rot180 (k=2)
    y3 = torch.rot90(p3, 2, [2, 3])
    # 4. Rot270 -> Inverse is Rot90 (k=1)
    y4 = torch.rot90(p4, 1, [2, 3])
    # 5. Flip H -> Inverse is Flip H
    y5 = torch.flip(p5, [3])
    # 6. Flip H + Rot90 -> Inverse is Flip H + Rot270
    y6 = torch.flip(torch.rot90(p6, 3, [2, 3]), [3])
    # 7. Flip H + Rot180 -> Inverse is Flip H + Rot180
    y7 = torch.flip(torch.rot90(p7, 2, [2, 3]), [3])
    # 8. Flip H + Rot270 -> Inverse is Flip H + Rot90
    y8 = torch.flip(torch.rot90(p8, 1, [2, 3]), [3])

    # Average all views
    avg_pred = (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8) / 8.0
    return avg_pred


def predict_and_submit(load_cached_data=True):
    """
    Runs the inference pipeline using the ensemble of models.
    Generates predictions, applies TTA, aggregates results, and saves the submission CSV.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    seed_everything(42)
    device = Config.DEVICE
    seeds = Config.SEEDS

    print("Initializing Inference Pipeline...")

    # Load Test Dataset
    # We access the underlying lists in test_ds to get original shapes and IDs
    test_ds = get_test_dataset(load_cached_data=load_cached_data)

    # Dictionary to accumulate predictions: {img_id: accumulated_numpy_array}
    acc_preds = {}

    # Iterate through each model in the ensemble
    for seed in seeds:
        model_path = Config.get_model_path(seed)
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model seed {seed}...")
        model = UNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Process each image in the test set
        for i in range(len(test_ds)):
            # Get data: noisy_tensor is padded to multiples of 16
            noisy_tensor, _, img_id = test_ds[i]

            # Apply TTA (returns tensor of shape (1, 1, H_pad, W_pad))
            pred_tensor = apply_tta(model, noisy_tensor.unsqueeze(0), device)
            pred_np = pred_tensor.squeeze().cpu().numpy()

            # Unpad: Crop back to original dimensions
            # Albumentations PadIfNeeded centers the image
            orig_img = test_ds.noisy_imgs[i]
            h_orig, w_orig = orig_img.shape
            h_pad, w_pad = pred_np.shape

            diff_h = h_pad - h_orig
            diff_w = w_pad - w_orig

            top = diff_h // 2
            left = diff_w // 2

            cropped_pred = pred_np[top : top + h_orig, left : left + w_orig]

            # Accumulate
            if img_id not in acc_preds:
                acc_preds[img_id] = np.zeros_like(cropped_pred, dtype=np.float64)

            acc_preds[img_id] += cropped_pred

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    print("Aggregating predictions and generating submission...")

    submission_ids = []
    submission_values = []

    # Process in order of IDs for consistency
    for img_id in test_ds.ids:
        if img_id not in acc_preds:
            continue

        # Average over the ensemble
        # Note: Assuming all seeds ran. If skipping logic is active, divisor should be adjusted.
        # Here we assume standard execution where Config.SEEDS is the target count.
        avg_pred = acc_preds[img_id] / len(seeds)

        # Re-invert Signal
        # Model output: 1.0 = Text, 0.0 = Background (Inverted space)
        # Submission:   0.0 = Text (Black), 1.0 = Background (White)
        # Formula:      Value = 1.0 - Prediction
        final_img = 1.0 - avg_pred

        # Clip to ensure valid range
        final_img = np.clip(final_img, 0, 1)

        # Flatten to pixel rows
        rows, cols = final_img.shape
        for r in range(rows):
            for c in range(cols):
                # ID format: {image_id}_{row}_{col} (1-based indexing)
                pixel_id = f"{img_id}_{r+1}_{c+1}"
                submission_ids.append(pixel_id)
                submission_values.append(final_img[r, c])

    # Create DataFrame and save
    df = pd.DataFrame({"id": submission_ids, "value": submission_values})

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
