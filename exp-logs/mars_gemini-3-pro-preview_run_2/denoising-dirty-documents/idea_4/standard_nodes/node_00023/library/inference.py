import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.model import ResUNet
from library.dataset import DenoisingDataset
from library.utils import load_checkpoint


def predict_tiled(
    model, image, tile_size=Config.TILE_SIZE, overlap=Config.TILE_OVERLAP
):
    """
    Performs tiled inference on a large image using a sliding window approach.
    Ensures that the input to the model is always of size (tile_size, tile_size).

    Args:
        model: The trained PyTorch model.
        image: Input tensor of shape (B, C, H, W).
        tile_size: Size of the square tile.
        overlap: Overlap between tiles.

    Returns:
        Tensor of shape (B, C, H, W) containing the prediction.
    """
    b, c, h, w = image.shape
    stride = tile_size - overlap

    # Calculate required padding to ensure the image is at least the size of one tile
    # and to handle edges gracefully.
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)

    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    # Update dimensions after padding
    _, _, h_padded, w_padded = image.shape

    # Initialize accumulation tensors
    output_map = torch.zeros_like(image)
    weight_map = torch.zeros_like(image)

    # Generate tile coordinates (top-left corners)
    # We ensure the last tile aligns with the bottom/right edge
    y_starts = list(range(0, h_padded - tile_size + 1, stride))
    if y_starts[-1] + tile_size < h_padded:
        y_starts.append(h_padded - tile_size)

    x_starts = list(range(0, w_padded - tile_size + 1, stride))
    if x_starts[-1] + tile_size < w_padded:
        x_starts.append(w_padded - tile_size)

    # Sliding window inference
    for y in y_starts:
        for x in x_starts:
            # Extract tile
            tile = image[:, :, y : y + tile_size, x : x + tile_size]

            with torch.no_grad():
                # Model prediction
                pred_tile = model(tile)

            # Accumulate prediction and weights
            output_map[:, :, y : y + tile_size, x : x + tile_size] += pred_tile
            weight_map[:, :, y : y + tile_size, x : x + tile_size] += 1.0

    # Average the overlapping regions
    output_map /= weight_map

    # Crop back to the original image dimensions
    prediction = output_map[:, :, :h, :w]

    return prediction


def apply_tta(model, image):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions
    from the original image and its geometric transformations.

    Transformations:
    1. Identity
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 90 degrees

    Args:
        model: The trained PyTorch model.
        image: Input tensor of shape (B, C, H, W).

    Returns:
        Tensor containing the averaged prediction.
    """
    preds = []

    # 1. Original
    preds.append(predict_tiled(model, image))

    # 2. Horizontal Flip
    img_h = torch.flip(image, [3])
    pred_h = predict_tiled(model, img_h)
    preds.append(torch.flip(pred_h, [3]))

    # 3. Vertical Flip
    img_v = torch.flip(image, [2])
    pred_v = predict_tiled(model, img_v)
    preds.append(torch.flip(pred_v, [2]))

    # 4. Rotate 90 degrees (Counter-Clockwise)
    # dims=[2, 3] corresponds to spatial H, W dimensions
    img_r90 = torch.rot90(image, k=1, dims=[2, 3])
    pred_r90 = predict_tiled(model, img_r90)
    # Inverse transform: Rotate -90 (or 270) -> k=-1
    preds.append(torch.rot90(pred_r90, k=-1, dims=[2, 3]))

    # Average all predictions
    avg_pred = torch.stack(preds).mean(dim=0)

    return avg_pred


def generate_submission(output_path=Config.SUBMISSION_FILE_PATH):
    """
    Generates the submission file for the test dataset.
    Loads the best model, runs inference with TTA, and saves predictions in the melted format.

    Args:
        output_path: Path to save the submission CSV.
    """
    device = Config.DEVICE
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")

    # Initialize and load model
    model = ResUNet().to(device)
    load_checkpoint(model, Config.MODEL_SAVE_PATH, device=device)
    model.eval()

    # Prepare Test Loader
    # Batch size 1 is required for variable image sizes and TTA
    test_dataset = DenoisingDataset(mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    print(f"Generating submission to {output_path}...")

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open file for writing results incrementally
    with open(output_path, "w") as f:
        # Write Header
        f.write("id,value\n")

        with torch.no_grad():
            for noisy, img_id in test_loader:
                noisy = noisy.to(device)
                img_id = img_id[0]  # Extract ID string from tuple

                # Predict Noise Residual
                if Config.USE_TTA:
                    pred_residual = apply_tta(model, noisy)
                else:
                    pred_residual = predict_tiled(model, noisy)

                # Reconstruct Clean Image
                # Ground Truth Formula: Clean = Noisy - Noise
                # Therefore: Pred_Clean = Noisy - Pred_Residual
                pred_clean = torch.clamp(noisy - pred_residual, 0, 1)

                # Convert to numpy array (H, W)
                pred_clean_np = pred_clean.squeeze().cpu().numpy()

                # Format output: Melt pixels to rows
                rows, cols = pred_clean_np.shape
                for r in range(rows):
                    for c in range(cols):
                        # Construct ID: image_row_col (1-based indexing)
                        pixel_id = f"{img_id}_{r+1}_{c+1}"
                        val = pred_clean_np[r, c]

                        # Write row
                        f.write(f"{pixel_id},{val}\n")

    print("Submission generation complete.")
