import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.models import UNet
from library.dataset import get_test_dataloader
from library.utils import seed_everything


def load_trained_models(device):
    """
    Loads the ensemble of trained models based on the configuration.

    Iterates through the defined streams (Context and Texture) and seeds,
    instantiating the specific UNet architecture for each and loading weights.

    Args:
        device: The torch device to load models onto.

    Returns:
        list: A list of loaded PyTorch models in eval mode.
    """
    models = []

    for stream_config in Config.STREAMS:
        depth = stream_config["depth"]
        base_channels = stream_config["base_channels"]
        stream_name = stream_config["name"]

        for seed in stream_config["seeds"]:
            model_name = f"{stream_name}_seed_{seed}.pth"
            model_path = os.path.join(Config.WORKING_DIR, model_name)

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}. Skipping.")
                continue

            # Instantiate model with stream-specific architecture
            model = UNet(
                n_channels=1, n_classes=1, depth=depth, base_channels=base_channels
            )

            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            model.to(device)
            model.eval()
            models.append(model)

    print(f"Loaded {len(models)} models for inference.")
    return models


def predict_with_tta(model, images):
    """
    Predicts using D4 Group Test-Time Augmentation (8 views).

    Applies 4 rotations and horizontal flips to create 8 views of the input,
    predicts on each, applies the inverse transformation, and averages the results.

    Args:
        model: Loaded PyTorch model.
        images: Input tensor (B, 1, H, W).

    Returns:
        Tensor: Averaged prediction (B, 1, H, W).
    """
    # List to store probabilities/predictions from different views
    preds = []

    # Define the spatial dimensions for rotation/flipping
    # images shape: (Batch, Channel, Height, Width) -> dims 2 and 3
    dims = [2, 3]

    with torch.no_grad():
        # 1. Original
        pred = model(images)
        preds.append(pred)

        # 2. Rot90
        img_rot90 = torch.rot90(images, k=1, dims=dims)
        pred_rot90 = model(img_rot90)
        preds.append(torch.rot90(pred_rot90, k=3, dims=dims))

        # 3. Rot180
        img_rot180 = torch.rot90(images, k=2, dims=dims)
        pred_rot180 = model(img_rot180)
        preds.append(torch.rot90(pred_rot180, k=2, dims=dims))

        # 4. Rot270
        img_rot270 = torch.rot90(images, k=3, dims=dims)
        pred_rot270 = model(img_rot270)
        preds.append(torch.rot90(pred_rot270, k=1, dims=dims))

        # 5. Horizontal Flip
        img_hflip = torch.flip(images, dims=[3])
        pred_hflip = model(img_hflip)
        preds.append(torch.flip(pred_hflip, dims=[3]))

        # 6. HFlip + Rot90
        # Flip then Rotate
        img_hflip_rot90 = torch.rot90(img_hflip, k=1, dims=dims)
        pred_hflip_rot90 = model(img_hflip_rot90)
        # Inverse: Rotate back (Rot270) then Flip back
        pred_inv = torch.rot90(pred_hflip_rot90, k=3, dims=dims)
        preds.append(torch.flip(pred_inv, dims=[3]))

        # 7. HFlip + Rot180
        img_hflip_rot180 = torch.rot90(img_hflip, k=2, dims=dims)
        pred_hflip_rot180 = model(img_hflip_rot180)
        pred_inv = torch.rot90(pred_hflip_rot180, k=2, dims=dims)
        preds.append(torch.flip(pred_inv, dims=[3]))

        # 8. HFlip + Rot270
        img_hflip_rot270 = torch.rot90(img_hflip, k=3, dims=dims)
        pred_hflip_rot270 = model(img_hflip_rot270)
        pred_inv = torch.rot90(pred_hflip_rot270, k=1, dims=dims)
        preds.append(torch.flip(pred_inv, dims=[3]))

    # Stack and average
    preds = torch.stack(preds, dim=0)
    avg_pred = torch.mean(preds, dim=0)

    return avg_pred


def generate_submission():
    """
    Generates the submission file by running inference on the test set.

    Steps:
    1. Loads the heterogeneous ensemble of models.
    2. Loads the test dataset (using caching if available).
    3. Predicts each image using TTA and ensemble averaging.
    4. Formats predictions into the required pixel-wise CSV format.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Models
    models = load_trained_models(device)
    if not models:
        print("Error: No models loaded. Cannot generate submission.")
        return

    # 2. Load Test Data
    # load_cached_data=True allows using cached test npz if available
    test_loader = get_test_dataloader(load_cached_data=True)

    print(f"Starting inference on {len(test_loader)} test images...")

    # 3. Prepare Submission File
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = Config.SUBMISSION_PATH

    # We will write to the file incrementally to save memory given the large number of pixels
    with open(submission_path, "w") as f:
        # Write Header
        f.write("id,value\n")

        # 4. Inference Loop
        for i, (images, img_ids) in enumerate(test_loader):
            images = images.to(device)
            img_id = str(img_ids[0])  # batch_size is 1

            # Aggregate predictions from all models
            ensemble_pred = None

            for model in models:
                # Get TTA prediction for this model
                tta_pred = predict_with_tta(model, images)

                if ensemble_pred is None:
                    ensemble_pred = tta_pred
                else:
                    ensemble_pred += tta_pred

            # Average across ensemble
            ensemble_pred /= len(models)

            # Post-process
            # Clamp between 0 and 1 to ensure valid intensity range
            ensemble_pred = torch.clamp(ensemble_pred, 0, 1)

            # Move to CPU numpy
            # Shape: (1, 1, H, W) -> (H, W)
            pred_np = ensemble_pred.squeeze().cpu().numpy()

            # 5. Format for Submission
            h, w = pred_np.shape

            # Create coordinate grids (1-based indexing as per task description)
            # np.indices returns (2, h, w) where [0] is rows, [1] is cols
            grid = np.indices((h, w))
            rows = grid[0] + 1
            cols = grid[1] + 1

            # Flatten arrays in row-major order
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()
            flat_vals = pred_np.flatten()

            # Construct formatted strings for the CSV
            # Format: {img_id}_{row}_{col},{value}
            lines = [
                f"{img_id}_{r}_{c},{v:.6f}"
                for r, c, v in zip(flat_rows, flat_cols, flat_vals)
            ]

            # Write to file
            f.write("\n".join(lines))
            f.write("\n")

            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1} images...")

    print(f"Submission generated at {submission_path}")
