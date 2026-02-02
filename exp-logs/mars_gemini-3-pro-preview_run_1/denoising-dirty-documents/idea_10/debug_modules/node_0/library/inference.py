import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import (
    get_config,
    DEVICE,
    WORKING_DIR,
    SUBMISSION_FILE_PATH,
    ENSEMBLE_SIZE,
)
from library.models import get_context_specialist, get_texture_specialist
from library.dataset import get_dataloaders
from library.utils import revert_intensity


def get_d4_transforms():
    """
    Returns a list of (transform_fn, inverse_fn) tuples for D4 group TTA.
    The D4 group consists of 8 symmetries of the square:
    Identity, Rot90, Rot180, Rot270, FlipH, FlipV, Transpose, Anti-Transpose.
    """

    # Helper functions for transformations on NCHW tensors (dim 2, 3)
    def identity(x):
        return x

    def rot90(x):
        return torch.rot90(x, 1, [2, 3])

    def rot90_inv(x):
        return torch.rot90(x, 3, [2, 3])

    def rot180(x):
        return torch.rot90(x, 2, [2, 3])  # Inverse is self

    def rot270(x):
        return torch.rot90(x, 3, [2, 3])

    def rot270_inv(x):
        return torch.rot90(x, 1, [2, 3])

    def hflip(x):
        return torch.flip(x, [3])  # Inverse is self

    transforms = []

    # 1. Identity
    transforms.append((identity, identity))

    # 2. Rot90
    transforms.append((rot90, rot90_inv))

    # 3. Rot180
    transforms.append((rot180, rot180))

    # 4. Rot270
    transforms.append((rot270, rot270_inv))

    # 5. HFlip
    transforms.append((hflip, hflip))

    # 6. HFlip + Rot90
    def t6_fwd(x):
        return rot90(hflip(x))

    def t6_inv(x):
        return hflip(rot90_inv(x))

    transforms.append((t6_fwd, t6_inv))

    # 7. HFlip + Rot180 (Vertical Flip)
    def t7_fwd(x):
        return rot180(hflip(x))

    def t7_inv(x):
        return hflip(rot180(x))

    transforms.append((t7_fwd, t7_inv))

    # 8. HFlip + Rot270
    def t8_fwd(x):
        return rot270(hflip(x))

    def t8_inv(x):
        return hflip(rot270_inv(x))

    transforms.append((t8_fwd, t8_inv))

    return transforms


def load_ensemble(device):
    """
    Loads all trained models in the ensemble (Context and Texture specialists).
    """
    models = []
    cfg = get_config()

    print(f"Loading ensemble models from {cfg['working_dir']}...")

    # Load Context Specialists
    for i in range(cfg["ensemble_size"]):
        path = os.path.join(cfg["working_dir"], f"context_model_{i}.pth")
        if os.path.exists(path):
            try:
                model = get_context_specialist()
                model.load_state_dict(torch.load(path, map_location=device))
                model.to(device)
                model.eval()
                models.append(model)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Warning: Model {path} not found. Skipping.")

    # Load Texture Specialists
    for i in range(cfg["ensemble_size"]):
        path = os.path.join(cfg["working_dir"], f"texture_model_{i}.pth")
        if os.path.exists(path):
            try:
                model = get_texture_specialist()
                model.load_state_dict(torch.load(path, map_location=device))
                model.to(device)
                model.eval()
                models.append(model)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"Warning: Model {path} not found. Skipping.")

    print(f"Successfully loaded {len(models)} models.")
    return models


def predict_with_tta(models, x):
    """
    Applies D4 Test-Time Augmentation and averages predictions across all models and views.

    Args:
        models (list): List of loaded PyTorch models.
        x (torch.Tensor): Input tensor of shape (1, 1, H, W).

    Returns:
        torch.Tensor: Averaged prediction tensor.
    """
    transforms = get_d4_transforms()
    accumulated_pred = None
    count = 0

    with torch.no_grad():
        for fwd, inv in transforms:
            # Augment input
            x_aug = fwd(x)

            # Pass through all models
            for model in models:
                y_aug = model(x_aug)

                # Inverse augment output
                y = inv(y_aug)

                if accumulated_pred is None:
                    accumulated_pred = y
                else:
                    accumulated_pred += y
                count += 1

    return accumulated_pred / count


def generate_submission(debug=False):
    """
    Generates the submission file for the test set.

    Args:
        debug (bool): If True, processes only a small subset of the test data.
    """
    print("Starting submission generation...")

    # 1. Load Data
    # Note: get_dataloaders for test mode returns a loader for the full test set.
    test_loader = get_dataloaders(batch_size=1, mode="test", load_cached_data=True)

    # 2. Load Models
    models = load_ensemble(DEVICE)
    if not models:
        print("No models loaded. Cannot generate submission.")
        return

    results = []

    # 3. Inference Loop
    print(f"Processing {len(test_loader)} test images...")

    for i, batch in enumerate(test_loader):
        if debug and i >= 5:
            print("Debug mode: Stopping after 5 images.")
            break

        img_id = str(batch["id"][0])
        inputs = batch["input"].to(DEVICE)  # (1, 1, H, W)

        # Predict
        # The models were trained on inverted intensity (Text=1, Bg=0).
        # Output is also inverted.
        pred_tensor = predict_with_tta(models, inputs)

        # Revert intensity to original space (Text=0, Bg=1)
        pred_tensor = revert_intensity(pred_tensor)

        # Clip to valid range [0, 1] to respect physical intensity limits
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)

        # Move to CPU numpy
        pred_img = pred_tensor.squeeze().cpu().numpy()  # (H, W)

        # 4. Format for Submission
        # Required format: id, value
        # id is {image_id}_{row}_{col} where row, col are 1-based indices
        h, w = pred_img.shape

        # Create coordinate grids (1-based)
        rows, cols = np.indices((h, w))
        rows = rows.flatten() + 1
        cols = cols.flatten() + 1

        flat_vals = pred_img.flatten()

        # Generate ID strings efficiently
        # Vectorized string creation is tricky in numpy, list comp is reasonably fast
        img_ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

        # Create a DataFrame for this image
        df_img = pd.DataFrame({"id": img_ids, "value": flat_vals})
        results.append(df_img)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1} images...")

    # 5. Concatenate and Save
    if results:
        print("Concatenating results...")
        final_df = pd.concat(results, ignore_index=True)

        print(f"Saving submission to {SUBMISSION_FILE_PATH}...")
        final_df.to_csv(SUBMISSION_FILE_PATH, index=False)
        print("Submission generation complete.")
    else:
        print("No results generated.")
