import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import UNet
from library.dataset import get_dataloaders
from library.utils import (
    pad_to_multiple,
    unpad,
    get_tta_transforms,
    inverse_tta_transforms,
)


def load_ensemble_models(device):
    """
    Loads the ensemble of trained models from the working directory.
    """
    models = []
    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

        # Check if model exists (robustness for partial runs)
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        # Initialize and load model
        model = UNet(n_channels=1, n_classes=1)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError(
            "No models loaded. Ensure training has completed successfully."
        )

    return models


def predict_with_tta_ensemble(models, x, device):
    """
    Performs inference on a single image tensor using the model ensemble
    and Test-Time Augmentation (TTA).

    Args:
        models: List of loaded PyTorch models.
        x: Input tensor of shape (B, C, H, W).
        device: Torch device.

    Returns:
        final_pred: Prediction tensor of shape (B, C, H, W).
    """
    x = x.to(device)

    # Pad input to be divisible by 16 (required for 4-level U-Net)
    padded_x, padding_info = pad_to_multiple(x, divisor=16)

    transforms = get_tta_transforms()
    inv_transforms = inverse_tta_transforms()

    accumulated_pred = torch.zeros_like(padded_x)
    count = 0

    with torch.no_grad():
        # Iterate over all geometric views (TTA)
        for t, inv_t in zip(transforms, inv_transforms):
            # Apply augmentation
            aug_x = t(padded_x)

            # Pass through each model in the ensemble
            for model in models:
                pred = model(aug_x)

                # Invert augmentation to restore original orientation
                pred = inv_t(pred)

                accumulated_pred += pred
                count += 1

    # Average predictions across all models and views
    avg_pred = accumulated_pred / count

    # Remove padding to restore original dimensions
    final_pred = unpad(avg_pred, padding_info)

    return final_pred


def generate_submission():
    """
    Main function to generate predictions for the test set and save the submission CSV.
    """
    device = Config.DEVICE

    print(f"Using device: {device}")

    # 1. Load Ensemble Models
    print("Loading ensemble models...")
    try:
        models = load_ensemble_models(device)
        print(f"Successfully loaded {len(models)} models.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Load Test Data
    # We use get_dataloaders which handles caching internally
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    results_ids = []
    results_values = []

    print("Starting inference...")

    # 3. Inference Loop
    for batch_idx, (noisy_img, img_ids) in enumerate(test_loader):
        # noisy_img shape: (1, 1, H, W) because batch_size=1
        img_id = img_ids[0]

        # Run prediction pipeline
        pred_tensor = predict_with_tta_ensemble(models, noisy_img, device)

        # Convert to numpy array (H, W)
        # Squeeze batch (0) and channel (1) dimensions
        pred_img = pred_tensor.squeeze().cpu().numpy()

        # 4. Format for Submission
        h, w = pred_img.shape

        # Create coordinate grids (1-based indexing)
        rows, cols = np.indices((h, w))
        rows = rows + 1
        cols = cols + 1

        # Flatten arrays
        flat_rows = rows.flatten()
        flat_cols = cols.flatten()
        flat_vals = pred_img.flatten()

        # Generate ID strings: {img_id}_{row}_{col}
        # Using list comprehension for string construction
        current_ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

        results_ids.extend(current_ids)
        results_values.extend(flat_vals)

        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {batch_idx + 1} images...")

    print("Inference complete. Constructing submission DataFrame...")

    # 5. Create DataFrame and Save
    df_sub = pd.DataFrame({"id": results_ids, "value": results_values})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Saving submission file to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation finished.")
