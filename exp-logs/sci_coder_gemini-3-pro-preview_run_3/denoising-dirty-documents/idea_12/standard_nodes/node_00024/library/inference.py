import os
import torch
import numpy as np
import pandas as pd
from library.config import DEVICE, SUBMISSION_PATH, WORKING_DIR
from library.model import CAResDnCNN
from library.dataset import load_test_data


def geometric_self_ensemble(model, x, device):
    """
    Applies Test-Time Augmentation (TTA) using the Dihedral group D4 (8 symmetries).
    Predicts noise for original and augmented versions, then averages the inverse-augmented predictions.

    Args:
        model (nn.Module): The trained denoising model.
        x (torch.Tensor): Input image tensor of shape (1, 1, H, W).
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Averaged noise prediction of shape (1, 1, H, W).
    """
    x = x.to(device)
    noise_accum = torch.zeros_like(x)

    # Iterate through 2 flip states (None, Horizontal) and 4 rotation states (0, 90, 180, 270)
    # This covers all 8 symmetries of the square.
    for flip in [False, True]:
        for k in range(4):
            # --- Augment ---
            aug_input = x

            # 1. Flip Horizontal (dim 3 is width)
            if flip:
                aug_input = torch.flip(aug_input, [3])

            # 2. Rotate 90*k degrees
            if k > 0:
                aug_input = torch.rot90(aug_input, k, [2, 3])

            # --- Predict ---
            with torch.no_grad():
                aug_noise = model(aug_input)

            # --- Inverse Augment ---
            # Operations must be reversed: Inverse Rotate then Inverse Flip

            # 1. Inverse Rotate (rotate by -k)
            if k > 0:
                aug_noise = torch.rot90(aug_noise, -k, [2, 3])

            # 2. Inverse Flip
            if flip:
                aug_noise = torch.flip(aug_noise, [3])

            noise_accum += aug_noise

    # Average the predictions
    return noise_accum / 8.0


def generate_submission(model_path=None, output_path=SUBMISSION_PATH):
    """
    Generates the submission file by running inference on the test set.

    Args:
        model_path (str, optional): Path to the trained model weights.
                                    Defaults to 'final_model.pth' in WORKING_DIR.
        output_path (str): Path to save the submission CSV.
    """
    # Determine model path
    if model_path is None:
        model_path = os.path.join(WORKING_DIR, "final_model.pth")

    if not os.path.exists(model_path):
        # Fallback to stage 2 best model if final model doesn't exist
        fallback = os.path.join(WORKING_DIR, "best_model_Stage_2.pth")
        if os.path.exists(fallback):
            print(f"Final model not found. Using best Stage 2 model: {fallback}")
            model_path = fallback
        else:
            raise FileNotFoundError(f"No model found at {model_path} or {fallback}")

    print(f"Loading model from {model_path}...")

    # Initialize Model
    model = CAResDnCNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load Test Data
    print("Loading test data...")
    test_ids, test_images = load_test_data()
    print(f"Loaded {len(test_images)} test images.")

    # Prepare Output File
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating predictions and writing to {output_path}...")

    # Open file context for writing
    with open(output_path, "w") as f:
        # Write Header
        f.write("id,value\n")

        for img_id_str, img_arr in zip(test_ids, test_images):
            # img_arr is (H, W) numpy float32 [0, 1]
            h, w = img_arr.shape

            # Convert to Tensor (1, 1, H, W)
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)

            # Predict Noise using Geometric Self-Ensemble
            noise_pred = geometric_self_ensemble(model, img_tensor, DEVICE)

            # Reconstruct Clean Image: Input - Noise
            # Move to CPU and numpy
            noise_pred_np = noise_pred.cpu().numpy()[0, 0]
            clean_pred = img_arr - noise_pred_np

            # Clip values to [0, 1]
            clean_pred = np.clip(clean_pred, 0.0, 1.0)

            # --- Format for Submission ---
            # ID format: {image_id}_{row}_{col}
            # image_id comes from filename, e.g., "110.png" -> "110"
            base_id = os.path.splitext(img_id_str)[0]

            # Create coordinate grids
            # Rows are 1-indexed, Cols are 1-indexed
            rows, cols = np.indices((h, w))
            rows = rows + 1
            cols = cols + 1

            # Flatten arrays
            flat_vals = clean_pred.flatten()
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()

            # Efficiently write to file
            # We construct a list of strings to join, which is faster than individual writes
            buffer = []
            for r, c, val in zip(flat_rows, flat_cols, flat_vals):
                buffer.append(f"{base_id}_{r}_{c},{val:.6f}\n")

            f.writelines(buffer)

    print("Submission generation complete.")
