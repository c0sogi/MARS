import os
import torch
import numpy as np
import cv2
from library.config import Config
from library.model import UNet
from library.dataset import get_dataloaders
from library.utils import create_submission_file, load_checkpoint


def apply_tta_and_predict(model, x):
    """
    Applies D4 Group Test-Time Augmentation (8 views), predicts,
    inverses transformations, and returns the average tensor.

    Args:
        model: The trained PyTorch model.
        x: Input tensor of shape (1, C, H, W).

    Returns:
        Tensor of shape (1, C, H, W) containing the averaged prediction.
    """
    # Define transformations
    # dims=(2, 3) corresponds to spatial dimensions (H, W) for (B, C, H, W)

    views = []

    # --- 1. Standard Rotations ---
    # 0 degrees
    views.append(x)
    # 90 degrees
    views.append(torch.rot90(x, 1, [2, 3]))
    # 180 degrees
    views.append(torch.rot90(x, 2, [2, 3]))
    # 270 degrees
    views.append(torch.rot90(x, 3, [2, 3]))

    # --- 2. Flipped Rotations (Horizontal Flip base) ---
    x_flip = torch.flip(x, [3])
    # Flip + 0 deg
    views.append(x_flip)
    # Flip + 90 deg
    views.append(torch.rot90(x_flip, 1, [2, 3]))
    # Flip + 180 deg
    views.append(torch.rot90(x_flip, 2, [2, 3]))
    # Flip + 270 deg
    views.append(torch.rot90(x_flip, 3, [2, 3]))

    # --- Batch Prediction ---
    # Stack views to predict in a single batch if memory allows,
    # or loop if constrained. Given batch size 1 and A100, stacking is fine.
    batch_input = torch.cat(views, dim=0)

    with torch.no_grad():
        batch_output = model(batch_input)

    # --- Inverse Transformations ---
    outputs = []

    # 0 deg -> No op
    outputs.append(batch_output[0:1])
    # 90 deg -> Rot -1
    outputs.append(torch.rot90(batch_output[1:2], -1, [2, 3]))
    # 180 deg -> Rot -2
    outputs.append(torch.rot90(batch_output[2:3], -2, [2, 3]))
    # 270 deg -> Rot -3
    outputs.append(torch.rot90(batch_output[3:4], -3, [2, 3]))

    # Flip -> Flip
    outputs.append(torch.flip(batch_output[4:5], [3]))
    # Flip + 90 -> Flip(Rot -1)
    outputs.append(torch.flip(torch.rot90(batch_output[5:6], -1, [2, 3]), [3]))
    # Flip + 180 -> Flip(Rot -2)
    outputs.append(torch.flip(torch.rot90(batch_output[6:7], -2, [2, 3]), [3]))
    # Flip + 270 -> Flip(Rot -3)
    outputs.append(torch.flip(torch.rot90(batch_output[7:8], -3, [2, 3]), [3]))

    # Average
    avg_output = torch.mean(torch.cat(outputs, dim=0), dim=0, keepdim=True)
    return avg_output


def predict_and_save():
    """
    Main inference function.
    Loads ensemble models, processes test set with TTA, aggregates results,
    performs signal re-inversion, and saves the submission file.
    """
    print("Starting Inference Pipeline...")
    Config.initialize()

    # 1. Prepare Data
    # get_dataloaders ensures cache exists. We discard train/val loaders.
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load raw cache to get original image dimensions for unpadding
    # The cache file is created by get_dataloaders in Config.WORKING_DIR
    cache_path = os.path.join(Config.WORKING_DIR, "test_cache.npz")
    print(f"Loading original image shapes from {cache_path}...")
    try:
        test_cache = np.load(cache_path, allow_pickle=False)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load test cache to retrieve original shapes: {e}"
        )

    # 2. Initialize Accumulator
    # Dictionary to store summed predictions: img_id -> np.array (H_orig, W_orig)
    ensemble_accum = {}

    # 3. Ensemble Loop
    device = torch.device(Config.DEVICE)

    for seed in Config.ENSEMBLE_SEEDS:
        model_path = Config.get_model_path(seed)
        print(f"Processing with model seed {seed} (Path: {model_path})...")

        # Initialize and load model
        model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS)
        model.to(device)

        try:
            load_checkpoint(model_path, model, device=Config.DEVICE)
        except FileNotFoundError:
            print(f"Warning: Model for seed {seed} not found. Skipping.")
            continue

        model.eval()

        # Iterate through test images
        for noisy_tensor, _, img_ids in test_loader:
            img_id = img_ids[0]  # Batch size is 1
            noisy_tensor = noisy_tensor.to(device)

            # Apply TTA and Predict
            pred_tensor = apply_tta_and_predict(model, noisy_tensor)

            # Post-process: Unpad
            # Retrieve original shape
            raw_key = f"noisy_{img_id}"
            if raw_key not in test_cache:
                raise KeyError(f"Image ID {img_id} not found in test cache.")

            orig_h, orig_w = test_cache[raw_key].shape

            # Calculate padding (Albumentations PadIfNeeded centers the image)
            # Tensor shape is (1, 1, H_pad, W_pad)
            _, _, pad_h, pad_w = pred_tensor.shape

            diff_h = pad_h - orig_h
            diff_w = pad_w - orig_w

            top = diff_h // 2
            left = diff_w // 2

            # Crop to original size
            # Output is (1, 1, H, W), we want (H, W) numpy array
            pred_np = pred_tensor.squeeze().cpu().numpy()

            # Handle case where squeeze removes too many dims if H or W is 1 (unlikely but safe)
            if pred_np.ndim == 3:
                pred_np = pred_np[0]  # (1, H, W) -> (H, W)

            cropped_pred = pred_np[top : top + orig_h, left : left + orig_w]

            # Accumulate
            if img_id not in ensemble_accum:
                ensemble_accum[img_id] = np.zeros_like(cropped_pred, dtype=np.float64)

            ensemble_accum[img_id] += cropped_pred

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # 4. Average and Post-process
    final_predictions = {}
    num_models = len(Config.ENSEMBLE_SEEDS)

    print(f"Aggregating predictions from {num_models} models...")

    for img_id, accum_pred in ensemble_accum.items():
        # Average
        avg_pred = accum_pred / num_models

        # Re-Invert Signal
        # Model predicted in inverted space (Bg=0, Text=1)
        # We need original space (Bg=1, Text=0)
        # Formula: 1.0 - pred
        if Config.INVERT_SIGNAL:
            final_pred = 1.0 - avg_pred
        else:
            final_pred = avg_pred

        # Clip to valid range [0, 1] just in case
        final_pred = np.clip(final_pred, 0, 1)

        final_predictions[img_id] = final_pred

    # 5. Generate Submission
    create_submission_file(final_predictions, Config.SUBMISSION_FILE)
    print("Inference complete.")
