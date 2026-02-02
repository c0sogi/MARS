import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import load_checkpoint
from library.model import CoSPResUNet
from library.dataset import TextDenoisingDataset, get_transforms


def predict_tiled(model, image, patch_size=128, overlap=0.5, device="cuda"):
    """
    Performs tiled inference on a single image tensor using a sliding window approach.

    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        patch_size (int): Size of the square patch.
        overlap (float): Overlap ratio between patches (0 to 1).
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: The reconstructed clean image tensor of shape (1, C, H, W).
    """
    model.eval()
    _, _, h, w = image.shape

    # Calculate stride
    stride = int(patch_size * (1 - overlap))

    # Pad image reflectively to ensure we can extract patches covering the edges
    # We pad right and bottom to ensure the loop covers the full image
    # We use reflection padding to provide realistic context at boundaries
    pad_right = patch_size
    pad_bottom = patch_size
    padding = (0, pad_right, 0, pad_bottom)  # (left, right, top, bottom)

    image_padded = F.pad(image, padding, mode="reflect")
    pad_h, pad_w = image_padded.shape[2], image_padded.shape[3]

    # Initialize accumulators
    output_sum = torch.zeros_like(image_padded)
    count_map = torch.zeros_like(image_padded)

    # Generate sliding window coordinates
    y_steps = range(0, pad_h - patch_size + 1, stride)
    x_steps = range(0, pad_w - patch_size + 1, stride)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                # Skip patches that are entirely outside the original image area
                # (Optimization: though with reflection padding, context is useful,
                # but we only care about the original HxW area eventually)
                if y > h + stride or x > w + stride:
                    continue

                # Extract patch
                patch_noisy = image_padded[:, :, y : y + patch_size, x : x + patch_size]

                # Predict noise residual
                patch_noise_pred = model(patch_noisy)

                # Reconstruct clean patch: Clean = Noisy - Noise
                patch_clean_pred = patch_noisy - patch_noise_pred

                # Accumulate
                output_sum[
                    :, :, y : y + patch_size, x : x + patch_size
                ] += patch_clean_pred
                count_map[:, :, y : y + patch_size, x : x + patch_size] += 1.0

    # Normalize by overlap count
    # Add epsilon to avoid division by zero (though padding ensures coverage)
    output_avg = output_sum / (count_map + 1e-6)

    # Crop back to original dimensions
    clean_image = output_avg[:, :, 0:h, 0:w]

    return clean_image


def predict_with_tta(model, image, patch_size=128, overlap=0.5, device="cuda"):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions from:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. 90-degree Rotation

    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Input image tensor (1, C, H, W).

    Returns:
        np.ndarray: The averaged clean image prediction (H, W).
    """
    preds = []

    # 1. Original
    p1 = predict_tiled(model, image, patch_size, overlap, device)
    preds.append(p1)

    # 2. Horizontal Flip
    img_hflip = torch.flip(image, [3])
    p2_hflip = predict_tiled(model, img_hflip, patch_size, overlap, device)
    p2 = torch.flip(p2_hflip, [3])
    preds.append(p2)

    # 3. Vertical Flip
    img_vflip = torch.flip(image, [2])
    p3_vflip = predict_tiled(model, img_vflip, patch_size, overlap, device)
    p3 = torch.flip(p3_vflip, [2])
    preds.append(p3)

    # 4. Rotate 90 degrees (counter-clockwise)
    # Dimensions: (N, C, H, W) -> dims 2, 3 are spatial
    img_rot = torch.rot90(image, 1, [2, 3])
    p4_rot = predict_tiled(model, img_rot, patch_size, overlap, device)
    p4 = torch.rot90(p4_rot, -1, [2, 3])  # Rotate back
    preds.append(p4)

    # Stack and Average
    stacked = torch.stack(preds, dim=0)
    avg_pred = torch.mean(stacked, dim=0)

    # Clamp values to valid range [0, 1]
    avg_pred = torch.clamp(avg_pred, 0, 1)

    # Convert to numpy and remove batch/channel dims
    return avg_pred.cpu().numpy().squeeze()


def generate_submission(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=1,
    device=Config.DEVICE,
):
    """
    Generates the submission file for the test dataset.

    Args:
        model_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size (must be 1 for varying image sizes).
        device (str): Device to run inference on.
    """
    print(f"Loading model from {model_path}...")

    # Initialize Model
    model = CoSPResUNet().to(device)

    # Load Checkpoint
    epoch, loss = load_checkpoint(model, filename=model_path, device=device)
    print(f"Model loaded successfully (Epoch: {epoch}, Validation Loss: {loss})")

    model.eval()

    # Initialize Test Dataset
    # Using 'test' mode which returns (image, id)
    test_dataset = TextDenoisingDataset(
        metadata_path=Config.TEST_METADATA,
        mode="test",
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Starting inference on {len(test_dataset)} test images...")

    submission_ids = []
    submission_values = []

    with torch.no_grad():
        for i, (noisy_img, img_id_tuple) in enumerate(test_loader):
            img_id = img_id_tuple[0]  # Extract string from tuple
            noisy_img = noisy_img.to(device)

            # Perform Inference with TTA
            clean_numpy = predict_with_tta(
                model,
                noisy_img,
                patch_size=Config.PATCH_SIZE,
                overlap=Config.TILE_OVERLAP,
                device=device,
            )

            # Flatten predictions
            h, w = clean_numpy.shape
            flat_values = clean_numpy.flatten()

            # Generate IDs: {img_id}_{row}_{col}
            # Note: Task uses 1-based indexing for rows and cols
            # Efficient list comprehension for ID generation
            current_ids = [
                f"{img_id}_{r}_{c}" for r in range(1, h + 1) for c in range(1, w + 1)
            ]

            # Append to lists
            submission_ids.extend(current_ids)
            submission_values.extend(flat_values)

            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} images.")

    # Create DataFrame
    print("Constructing submission DataFrame...")
    df_submission = pd.DataFrame({"id": submission_ids, "value": submission_values})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)
    print("Submission generation complete.")
