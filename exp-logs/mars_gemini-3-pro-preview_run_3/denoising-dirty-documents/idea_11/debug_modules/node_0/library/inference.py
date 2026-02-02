import os
import torch
import pandas as pd
import numpy as np
import cv2

from library.config import Config
from library.network import CAResDnCNN
from library.dataset import load_image


def geometric_self_ensemble(model, x):
    """
    Performs Geometric Self-Ensemble (Test-Time Augmentation) by averaging
    predictions across 8 geometric transformations (D4 Group).

    Args:
        model: The trained PyTorch model.
        x: Input tensor of shape (B, C, H, W).

    Returns:
        Averaged predicted tensor of shape (B, C, H, W).
    """
    # Define pairs of (forward_transform, inverse_transform)
    # Dimensions: 2=H, 3=W
    transforms = []

    # 1. Identity
    transforms.append((lambda t: t, lambda t: t))

    # 2. Rotate 90
    transforms.append(
        (lambda t: torch.rot90(t, 1, [2, 3]), lambda t: torch.rot90(t, -1, [2, 3]))
    )

    # 3. Rotate 180
    transforms.append(
        (lambda t: torch.rot90(t, 2, [2, 3]), lambda t: torch.rot90(t, -2, [2, 3]))
    )

    # 4. Rotate 270
    transforms.append(
        (lambda t: torch.rot90(t, 3, [2, 3]), lambda t: torch.rot90(t, -3, [2, 3]))
    )

    # 5. Flip Horizontal
    transforms.append((lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])))

    # 6. Flip Vertical
    transforms.append((lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])))

    # 7. Transpose (Flip H + Rot 90)
    transforms.append(
        (
            lambda t: torch.rot90(torch.flip(t, [3]), 1, [2, 3]),
            lambda t: torch.flip(torch.rot90(t, -1, [2, 3]), [3]),
        )
    )

    # 8. Anti-Transpose (Flip V + Rot 90)
    transforms.append(
        (
            lambda t: torch.rot90(torch.flip(t, [2]), 1, [2, 3]),
            lambda t: torch.flip(torch.rot90(t, -1, [2, 3]), [2]),
        )
    )

    accumulated_output = 0.0

    for fwd, inv in transforms:
        # Augment
        aug_x = fwd(x)

        # Predict
        # Model predicts noise residual
        pred_noise = model(aug_x)

        # Inverse Augment
        pred_original_orientation = inv(pred_noise)

        accumulated_output += pred_original_orientation

    return accumulated_output / len(transforms)


def generate_submission():
    """
    Loads the trained model, runs inference on the test set using TTA,
    and generates the submission CSV file.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running inference on {device}...")

    # 1. Load Model
    model = CAResDnCNN(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=Config.NUM_FEATURES,
        num_blocks=Config.NUM_BLOCKS,
    ).to(device)

    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Loading weights from {Config.CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    else:
        print(
            f"Warning: Checkpoint not found at {Config.CHECKPOINT_PATH}. Using random weights."
        )

    model.eval()

    # 2. Prepare Output File
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # We will write to file incrementally to save memory
    with open(Config.SUBMISSION_PATH, "w") as f:
        f.write("id,value\n")

        # 3. Load Test Metadata
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        print(f"Processing {len(df_test)} test images...")

        with torch.no_grad():
            for _, row in df_test.iterrows():
                image_filename = row["image_id"]
                # Extract ID (e.g., "110.png" -> "110")
                img_id_str = os.path.splitext(image_filename)[0]

                input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

                # Load Image (H, W) normalized [0, 1]
                img_in_np = load_image(input_path)
                h, w = img_in_np.shape

                # To Tensor (1, 1, H, W)
                img_in_tensor = (
                    torch.from_numpy(img_in_np).unsqueeze(0).unsqueeze(0).to(device)
                )

                # Inference with TTA
                if Config.TTA_ENABLED:
                    pred_noise = geometric_self_ensemble(model, img_in_tensor)
                else:
                    pred_noise = model(img_in_tensor)

                # Reconstruct Clean Image: Clean = Input - Noise
                pred_clean_tensor = img_in_tensor - pred_noise

                # Clamp to valid range
                pred_clean_tensor = torch.clamp(pred_clean_tensor, 0.0, 1.0)

                # Convert back to numpy (H, W)
                pred_clean_np = pred_clean_tensor.squeeze().cpu().numpy()

                # 4. Format for Submission
                # Format: id = {image_id}_{row}_{col}, value = intensity
                # Rows and Cols are 1-based

                # Create coordinate grids
                # rows: 1..H, cols: 1..W
                row_indices = np.arange(1, h + 1)
                col_indices = np.arange(1, w + 1)

                # Meshgrid to get coordinates for every pixel
                # indexing='ij' ensures we get (row, col) order matching the image array
                r_grid, c_grid = np.meshgrid(row_indices, col_indices, indexing="ij")

                # Flatten everything
                flat_vals = pred_clean_np.flatten()
                flat_rows = r_grid.flatten()
                flat_cols = c_grid.flatten()

                # Construct ID strings efficiently
                # Using list comprehension which is generally fast enough for this scale
                ids = [f"{img_id_str}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

                # Write chunk to file
                # We construct a string buffer for this image
                lines = [f"{i},{v:.4f}\n" for i, v in zip(ids, flat_vals)]
                f.writelines(lines)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
