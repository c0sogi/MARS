import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from library.config import Config
from library.model import DeepSupervisionUNet
from library.dataset import get_dataloaders
from library.utils import seed_everything


class TTAHandler:
    """
    Handles Test-Time Augmentation (TTA) using the Dihedral Group D4 (8 symmetries).
    Includes methods to apply geometric transformations and their inverses.
    """

    def __init__(self):
        # Transformations defined by (k=rotations_90_deg, flip=horizontal_flip)
        # We apply rotation first, then flip.
        self.transforms = [
            (0, False),  # Original
            (1, False),  # Rot 90
            (2, False),  # Rot 180
            (3, False),  # Rot 270
            (0, True),  # H-Flip
            (1, True),  # Rot 90 + H-Flip
            (2, True),  # Rot 180 + H-Flip
            (3, True),  # Rot 270 + H-Flip
        ]

    def apply_transforms(self, image):
        """
        Applies all 8 geometric transformations to the input image.
        Groups them by shape to handle non-square images (Cite debug_lesson_2).

        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W).

        Returns:
            list: A list of tuples (batch_tensor, transform_params_list).
        """
        # Group 1: Preserves aspect ratio (k=0, 2)
        group_preserve = []
        params_preserve = []

        # Group 2: Swaps aspect ratio (k=1, 3)
        group_swap = []
        params_swap = []

        for k, flip in self.transforms:
            img_aug = image.clone()

            # Apply Rotation (k * 90 degrees counter-clockwise)
            if k > 0:
                img_aug = torch.rot90(img_aug, k, dims=[-2, -1])

            # Apply Horizontal Flip
            if flip:
                img_aug = torch.flip(img_aug, dims=[-1])

            # Check if dimensions are swapped (odd rotations swap H and W)
            if k % 2 != 0:
                group_swap.append(img_aug)
                params_swap.append((k, flip))
            else:
                group_preserve.append(img_aug)
                params_preserve.append((k, flip))

        results = []
        if group_preserve:
            results.append((torch.cat(group_preserve, dim=0), params_preserve))
        if group_swap:
            results.append((torch.cat(group_swap, dim=0), params_swap))

        return results

    def reverse_transforms(self, predictions_groups):
        """
        Reverses the applied transformations to align predictions back to the original orientation.

        Args:
            predictions_groups (list): List of (prediction_batch, transform_params_list).

        Returns:
            torch.Tensor: Batch of aligned predictions (8, C, H, W).
        """
        aligned_preds = []

        for batch, params_list in predictions_groups:
            for i, (k, flip) in enumerate(params_list):
                pred = batch[i : i + 1]  # Keep batch dim

                # Inverse Flip (Flip is its own inverse)
                if flip:
                    pred = torch.flip(pred, dims=[-1])

                # Inverse Rotation (Rotate by -k or 4-k)
                if k > 0:
                    pred = torch.rot90(pred, -k, dims=[-2, -1])

                aligned_preds.append(pred)

        return torch.cat(aligned_preds, dim=0)


def predict_with_ensemble(load_cached_data=True):
    """
    Performs inference using an ensemble of models and Test-Time Augmentation.

    Args:
        load_cached_data (bool): Whether to load dataset from cache.

    Returns:
        dict: A dictionary mapping image_id (str) to predicted pixel array (np.ndarray).
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We only need the test loader
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Load Ensemble Models
    models = []
    for i in range(Config.NUM_MODELS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model checkpoint {model_path} not found. Skipping.")
            continue

        model = DeepSupervisionUNet()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No models found for ensemble inference.")

    print(f"Loaded {len(models)} models for inference.")

    tta = TTAHandler()
    results = {}

    print("Starting inference on test set...")

    # Disable gradient calculation for inference
    with torch.no_grad():
        for noisy_img, img_id_batch in test_loader:
            img_id = str(img_id_batch[0])
            noisy_img = noisy_img.to(device)  # Shape: (1, 1, H, W)

            # Generate TTA batches (grouped by shape)
            tta_groups = tta.apply_transforms(noisy_img)

            ensemble_preds = []

            # Pass TTA batches through each model in the ensemble
            for model in models:
                model_preds_groups = []

                for batch, params in tta_groups:
                    # Model output: (BatchSize, 1, H, W)
                    preds = model(batch)
                    model_preds_groups.append((preds, params))

                # Reverse TTA to align predictions
                aligned_preds = tta.reverse_transforms(
                    model_preds_groups
                )  # (8, 1, H, W)

                # Average over TTA variations for this model
                model_avg = torch.mean(
                    aligned_preds, dim=0, keepdim=True
                )  # (1, 1, H, W)
                ensemble_preds.append(model_avg)

            # Average over all models in the ensemble
            ensemble_tensor = torch.cat(ensemble_preds, dim=0)  # (N_models, 1, H, W)
            final_pred = torch.mean(ensemble_tensor, dim=0)  # (1, 1, H, W)

            # Clamp predictions to valid pixel range [0, 1]
            final_pred = torch.clamp(final_pred, 0.0, 1.0)

            # Convert to numpy array (H, W)
            final_pred_np = final_pred.squeeze().cpu().numpy()
            results[img_id] = final_pred_np

    return results


def create_submission_file(predictions):
    """
    Formats predictions into the submission CSV format: id,value.
    id is formatted as {image_id}_{row}_{col}.

    Args:
        predictions (dict): Dictionary mapping image_id to numpy array (H, W).
    """
    print("Generating submission file...")

    submission_dfs = []

    # Process images in deterministic order
    sorted_ids = sorted(predictions.keys())

    for img_id in sorted_ids:
        pred_map = predictions[img_id]
        h, w = pred_map.shape

        # Generate coordinate grids (1-based indexing)
        # np.indices returns (2, h, w) where [0] is row indices, [1] is col indices
        grid = np.indices((h, w))
        rows = grid[0].flatten() + 1
        cols = grid[1].flatten() + 1
        values = pred_map.flatten()

        # Construct ID strings efficiently
        # Format: image_row_col
        ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

        # Create DataFrame for this image
        df_img = pd.DataFrame({"id": ids, "value": values})
        submission_dfs.append(df_img)

    # Concatenate all image DataFrames
    if submission_dfs:
        full_df = pd.concat(submission_dfs, ignore_index=True)
    else:
        full_df = pd.DataFrame(columns=["id", "value"])

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    full_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
