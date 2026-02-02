import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, pad_image, unpad_image
from library.model import UNet
from library.dataset import get_dataloaders


def generate_submission(dataset_limit=None):
    """
    Generates the submission file by loading the trained model and running inference
    on the test dataset. Applies padding to handle dimension mismatches and
    formats the output as required.

    Args:
        dataset_limit (int, optional): Limit the number of test images for debugging.
    """
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running inference on device: {device}")

    # 2. Data Loading
    # We only need the test loader
    _, _, test_loader = get_dataloaders(dataset_limit=dataset_limit)

    # 3. Model Initialization
    # Initialize the U-Net model with the same configuration as training
    model = UNet(n_channels=Config.NUM_CHANNELS, n_classes=1, bilinear=True)
    model.to(device)

    # 4. Load Weights
    checkpoint_path = Config.MODEL_CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    # 5. Inference Loop
    model.eval()
    results = []

    print("Starting prediction loop...")

    with torch.no_grad():
        for i, (inputs, img_ids) in enumerate(test_loader):
            inputs = inputs.to(device)
            img_id = img_ids[0]  # Batch size is 1

            # Capture original dimensions for unpadding
            # Shape: (B, C, H, W) -> (1, 1, H, W)
            original_shape = (inputs.shape[2], inputs.shape[3])

            # Pad image to be divisible by 32 (safe for U-Net depth)
            # Using 'reflect' mode to minimize boundary artifacts
            padded_inputs = pad_image(inputs, divisor=32, mode="reflect")

            # Cite solution_lesson_node_00006: Test-Time Augmentation (TTA)
            # Create augmented versions
            # 1. Original
            inp_orig = padded_inputs
            # 2. Horizontal Flip
            inp_h = torch.flip(padded_inputs, [3])
            # 3. Vertical Flip
            inp_v = torch.flip(padded_inputs, [2])
            # 4. Both (180 rotation)
            inp_hv = torch.flip(padded_inputs, [2, 3])

            # Process all views
            out_orig = model(inp_orig)
            out_h = model(inp_h)
            out_v = model(inp_v)
            out_hv = model(inp_hv)

            # Inverse transforms
            pred_orig = out_orig
            pred_h = torch.flip(out_h, [3])
            pred_v = torch.flip(out_v, [2])
            pred_hv = torch.flip(out_hv, [2, 3])

            # Average predictions
            outputs = (pred_orig + pred_h + pred_v + pred_hv) / 4.0

            # Crop back to original size
            cropped_outputs = unpad_image(outputs, original_shape)

            # Process output
            # Remove batch and channel dims: (1, 1, H, W) -> (H, W)
            pred_mask = cropped_outputs.squeeze().cpu().numpy()

            h, w = pred_mask.shape

            # Generate pixel IDs: {img_id}_{row}_{col} (1-based)
            # Create coordinate grids
            # Note: np.indices returns (2, h, w), so we unpack
            grid_rows, grid_cols = np.indices((h, w))

            # Flatten everything
            flat_vals = pred_mask.flatten()
            flat_rows = grid_rows.flatten() + 1  # 1-based
            flat_cols = grid_cols.flatten() + 1  # 1-based

            # Create ID strings
            # Using list comprehension
            flat_ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

            # Create DataFrame for this image
            df_img = pd.DataFrame({"id": flat_ids, "value": flat_vals})

            results.append(df_img)

    # 6. Save Submission
    if results:
        final_df = pd.concat(results, ignore_index=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        final_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission generated successfully. Saved to {Config.SUBMISSION_PATH}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("No results generated. Check if test dataset is empty.")
