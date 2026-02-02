import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import UNet
from library.dataset import get_dataloaders
from library.utils import seed_everything, pad_image, unpad_image


def predict_with_tta(model, image_tensor):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Averages predictions of the original image and its geometric transformations
    (Horizontal Flip, Vertical Flip, and Combined Flip).

    Args:
        model (torch.nn.Module): The trained model.
        image_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Averaged prediction tensor.
    """
    # 1. Original
    pred_1 = model(image_tensor)

    # 2. Horizontal Flip
    # Flip width dimension (dim 3)
    img_h = torch.flip(image_tensor, dims=[3])
    out_h = model(img_h)
    pred_2 = torch.flip(out_h, dims=[3])

    # 3. Vertical Flip
    # Flip height dimension (dim 2)
    img_v = torch.flip(image_tensor, dims=[2])
    out_v = model(img_v)
    pred_3 = torch.flip(out_v, dims=[2])

    # 4. Rotate 180 (Horizontal + Vertical Flip)
    # Flip both height and width dimensions
    img_hv = torch.flip(image_tensor, dims=[2, 3])
    out_hv = model(img_hv)
    pred_4 = torch.flip(out_hv, dims=[2, 3])

    # Average predictions
    return (pred_1 + pred_2 + pred_3 + pred_4) / 4.0


def generate_submission(
    checkpoint_path=Config.MODEL_CHECKPOINT,
    output_file=Config.SUBMISSION_FILE,
    device=Config.DEVICE,
    use_tta=Config.TTA_ENABLED,
    batch_size=1,
):
    """
    Generates the submission CSV file by running inference on the test set.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_file (str): Path to save the submission CSV.
        device (str): Device to run inference on ('cpu' or 'cuda').
        use_tta (bool): Whether to use Test-Time Augmentation.
        batch_size (int): Batch size for the data loader (default 1 for variable image sizes).
    """
    # Ensure reproducibility
    seed_everything()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Initializing inference on {device}...")

    # Initialize Model
    model = UNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES,
    ).to(device)

    # Load Weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Using initialized weights (random)."
        )

    model.eval()

    # Get Test DataLoader
    # We only need the test loader
    _, _, test_loader = get_dataloaders(
        test_batch_size=batch_size, num_workers=Config.NUM_WORKERS
    )

    submission_data = []
    print("Starting prediction loop...")

    with torch.no_grad():
        for noisy_batch, img_ids in test_loader:
            # noisy_batch: (B, 1, H, W)
            # img_ids: tuple of size B

            # Iterate through batch (usually size 1 due to variable image sizes)
            for i in range(len(img_ids)):
                img_id = img_ids[i]
                noisy_tensor = noisy_batch[i].unsqueeze(0)  # (1, 1, H, W)

                # Preprocessing: Pad image to be divisible by 32
                # Convert to numpy for padding logic
                noisy_np = noisy_tensor.squeeze().numpy()  # (H, W)
                padded_noisy = pad_image(noisy_np, factor=32)

                # Convert back to tensor
                input_tensor = (
                    torch.from_numpy(padded_noisy).unsqueeze(0).unsqueeze(0).to(device)
                )

                # Inference
                if use_tta:
                    output_tensor = predict_with_tta(model, input_tensor)
                else:
                    output_tensor = model(input_tensor)

                # Post-processing: Unpad and Clip
                output_np = output_tensor.squeeze().cpu().numpy()  # (H_pad, W_pad)
                output_clean = unpad_image(output_np, noisy_np.shape)
                output_clean = np.clip(output_clean, 0, 1)

                # Format for Submission
                h, w = output_clean.shape

                # Create 1-based indices for rows and columns
                # np.indices creates grid of indices
                grid = np.indices((h, w))
                rows = grid[0].flatten() + 1
                cols = grid[1].flatten() + 1

                flat_vals = output_clean.flatten()

                # Construct IDs: "{img_id}_{row}_{col}"
                # Using list comprehension for string formatting
                ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

                # Add to submission data
                submission_data.extend(zip(ids, flat_vals))

    # Save to CSV
    print(f"Saving submission to {output_file}...")
    df = pd.DataFrame(submission_data, columns=["id", "value"])
    df.to_csv(output_file, index=False)
    print("Submission generation complete.")
