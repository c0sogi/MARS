import os
import torch
import numpy as np
import pandas as pd
from library.config import DEVICE, WORKING_DIR, SUBMISSION_DIR, N_FOLDS, set_seed, SEED
from library.model import ResidualShallowUNet
from library.dataset import get_test_loader


def predict_tta(model, x, device):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Averages predictions across 8 geometric transformations (D4 symmetry group).

    Args:
        model (nn.Module): The trained model.
        x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        device (str): Device to perform inference on.

    Returns:
        torch.Tensor: Averaged prediction tensor.
    """
    model.eval()
    preds = []

    # The 8 symmetries of a square: combinations of 4 rotations and horizontal flip
    # k: number of 90-degree rotations (0, 1, 2, 3)
    # flip: apply horizontal flip (True/False)

    with torch.no_grad():
        for k in [0, 1, 2, 3]:
            for flip in [False, True]:
                # 1. Augment
                aug_x = x.clone()

                # Apply flip (dim 3 is width)
                if flip:
                    aug_x = torch.flip(aug_x, dims=[3])

                # Apply rotation (dims 2,3 are H,W)
                if k > 0:
                    aug_x = torch.rot90(aug_x, k=k, dims=[2, 3])

                # 2. Predict (Model predicts noise residual)
                aug_x = aug_x.to(device)
                pred_residual = model(aug_x)

                # 3. Inverse Augment
                # Inverse rotation
                if k > 0:
                    pred_residual = torch.rot90(pred_residual, k=-k, dims=[2, 3])

                # Inverse flip
                if flip:
                    pred_residual = torch.flip(pred_residual, dims=[3])

                preds.append(pred_residual.cpu())

    # Stack and average
    avg_residual = torch.stack(preds).mean(dim=0)
    return avg_residual


def generate_submission(load_cached_data=True):
    """
    Generates the submission file by running inference on the test set.
    Ensembles predictions from all available fold models.

    Args:
        load_cached_data (bool): Whether to use cached dataset files.
    """
    set_seed(SEED)

    # 1. Load Data
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    # 2. Load Models
    models = []
    print(f"Loading {N_FOLDS} models for ensemble...")
    for fold_idx in range(N_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")

        # Initialize model structure
        model = ResidualShallowUNet(n_channels=1, n_classes=1)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            models.append(model)
            print(f"Loaded model fold {fold_idx}")
        else:
            print(
                f"Warning: Model for fold {fold_idx} not found at {model_path}. Skipping."
            )

    if not models:
        raise RuntimeError("No models loaded. Cannot generate submission.")

    # 3. Prepare Submission File
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    print(f"Generating submission at {submission_path}...")

    # Write header first
    with open(submission_path, "w") as f:
        f.write("id,value\n")

    # 4. Inference Loop
    for i, (noisy_img, img_ids) in enumerate(test_loader):
        # noisy_img shape: (B, 1, H, W), B=1
        # img_ids is a tuple of size B

        img_id = img_ids[0]

        # Ensemble Prediction
        fold_residuals = []
        for model in models:
            # Predict residual using TTA
            res = predict_tta(model, noisy_img, DEVICE)
            fold_residuals.append(res)

        # Average across folds
        avg_residual = torch.stack(fold_residuals).mean(dim=0)

        # Reconstruct Clean Image: Clean = Noisy - Residual
        # noisy_img is on CPU from loader, avg_residual is on CPU
        clean_tensor = noisy_img - avg_residual

        # Clamp to valid range [0, 1]
        clean_tensor = torch.clamp(clean_tensor, 0.0, 1.0)

        # Convert to numpy (H, W)
        clean_img = clean_tensor.squeeze().numpy()

        # 5. Format and Write to CSV
        h, w = clean_img.shape

        # Generate indices (1-based)
        # rows: 1..H, cols: 1..W
        rows, cols = np.indices((h, w))
        rows = rows + 1
        cols = cols + 1

        # Flatten arrays
        flat_vals = clean_img.flatten()
        flat_rows = rows.flatten()
        flat_cols = cols.flatten()

        # Create ID strings: "{img_id}_{row}_{col}"
        # Using list comprehension which is reasonably efficient for this size
        ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

        # Create DataFrame chunk
        df_chunk = pd.DataFrame({"id": ids, "value": flat_vals})

        # Append to CSV
        df_chunk.to_csv(submission_path, mode="a", header=False, index=False)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1} test images.")

    print("Submission generation complete.")
