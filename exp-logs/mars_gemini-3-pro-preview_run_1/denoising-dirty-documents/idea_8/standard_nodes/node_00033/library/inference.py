import os
import torch
import numpy as np
import pandas as pd
from library.config import DEVICE, WORKING_DIR, SUBMISSION_PATH, SEEDS
from library.model import ASPPUNet
from library.dataset import get_dataloaders
from library.utils import load_checkpoint


def apply_d4_tta(model, x):
    """
    Applies Test-Time Augmentation (TTA) using the D4 symmetry group (8 views).

    Args:
        model (nn.Module): The trained model.
        x (torch.Tensor): Input image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: The averaged prediction across all 8 views.
    """
    preds = []

    # Define transformations as (k_rotations, do_horizontal_flip)
    # k=0,1,2,3 corresponds to 0, 90, 180, 270 degrees counter-clockwise
    transforms = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]

    for k, flip in transforms:
        # --- 1. Augment ---
        aug_x = x.clone()

        # Apply Horizontal Flip (along width dimension, dim=3)
        if flip:
            aug_x = torch.flip(aug_x, [3])

        # Apply Rotation (in H-W plane, dims=[2, 3])
        if k > 0:
            aug_x = torch.rot90(aug_x, k, [2, 3])

        # --- 2. Predict ---
        with torch.no_grad():
            pred = model(aug_x)

        # --- 3. De-augment (Inverse Transform) ---
        # We must reverse the operations in reverse order:
        # Inverse of (Flip -> Rotate) is (Inverse Rotate -> Inverse Flip)

        # Inverse Rotation: Rotate by -k (or 4-k)
        if k > 0:
            pred = torch.rot90(pred, -k, [2, 3])

        # Inverse Flip: Flip again
        if flip:
            pred = torch.flip(pred, [3])

        preds.append(pred)

    # Average predictions across all views
    avg_pred = torch.stack(preds).mean(dim=0)
    return avg_pred


def generate_predictions(seeds=SEEDS, load_cached_data=True):
    """
    Generates predictions for the test set using an ensemble of models trained with different seeds.
    Applies D4 TTA and averages predictions across all models.

    Args:
        seeds (list): List of seeds corresponding to the trained model checkpoints.
        load_cached_data (bool): Whether to use cached dataset files.
    """
    print(f"Generating predictions using ensemble of {len(seeds)} models...")

    # 1. Load Test Data
    # We only need the test loader
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Load All Models
    models = []
    for seed in seeds:
        checkpoint_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for seed {seed} not found at {checkpoint_path}. Skipping."
            )
            continue

        model = ASPPUNet().to(DEVICE)
        load_checkpoint(checkpoint_path, model)
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError(
            "No valid model checkpoints found. Cannot generate predictions."
        )

    print(f"Successfully loaded {len(models)} models.")

    # 3. Inference Loop
    results_ids = []
    results_values = []

    print("Starting inference on test set...")

    for i, batch in enumerate(test_loader):
        # Unpack batch (dataset returns noisy_t, img_id)
        noisy, img_id_batch = batch
        noisy = noisy.to(DEVICE)
        img_id = img_id_batch[0]  # batch_size is 1

        # Ensemble Prediction
        final_pred = None

        for model in models:
            # Apply TTA for this model
            tta_pred = apply_d4_tta(model, noisy)

            if final_pred is None:
                final_pred = tta_pred
            else:
                final_pred += tta_pred

        # Average across ensemble
        final_pred /= len(models)

        # 4. Format for Submission
        # Remove batch and channel dimensions -> (H, W)
        pred_np = final_pred.squeeze().cpu().numpy()
        h, w = pred_np.shape

        # Generate pixel IDs: "{img_id}_{row}_{col}"
        # Note: Rows and Cols are 1-indexed based on the task description
        rows, cols = np.indices((h, w))
        rows = rows + 1
        cols = cols + 1

        # Flatten arrays
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        vals_flat = pred_np.flatten()

        # Create ID strings
        # Using list comprehension is generally efficient for string formatting
        current_ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows_flat, cols_flat)]

        results_ids.extend(current_ids)
        results_values.extend(vals_flat)

    # 5. Save Submission
    print("Creating submission DataFrame...")
    df = pd.DataFrame({"id": results_ids, "value": results_values})

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission to {SUBMISSION_PATH}...")
    df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
