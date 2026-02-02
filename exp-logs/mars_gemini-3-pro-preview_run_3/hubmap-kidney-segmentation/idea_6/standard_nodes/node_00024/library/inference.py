import os
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.windows import Window
import cv2
from torch.cuda.amp import autocast

from library.config import Config
from library.model import UnetPlusPlus
from library.dataset import HubmapDataset
from library.utils import rle_encode, get_tissue_mask


def get_gaussian_window(size, sigma=None):
    """
    Generates a 2D Gaussian window for weighting predictions.

    Args:
        size (int): Size of the window (assumed square).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 2D Gaussian window of shape (size, size).
    """
    if sigma is None:
        sigma = size / 2.0

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # Calculate Gaussian
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    return g


def predict_sliding_window(
    model, image_path, image_id, anatomical_json_path, height, width, device
):
    """
    Performs sliding window inference on a large image with ROI constraints and Gaussian blending.

    Args:
        model (nn.Module): Trained model.
        image_path (str): Path to the TIFF image.
        image_id (str): ID of the image.
        anatomical_json_path (str): Path to the anatomical structure JSON.
        height (int): Image height.
        width (int): Image width.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Binary prediction mask.
    """
    tile_size = Config.TILE_SIZE
    stride = int(tile_size * 0.5)  # 50% overlap

    # 1. Load Tissue Mask (ROI)
    # We only run inference on tiles that intersect with the tissue mask
    tissue_mask = get_tissue_mask(image_id, width, height, anatomical_json_path)

    # 2. Initialize Accumulators
    # We use float16 to save memory if needed, but float32 is safer for precision
    prob_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # 3. Prepare Gaussian Window
    gaussian_window = get_gaussian_window(tile_size)

    # 4. Sliding Window Loop
    # We iterate over coordinates
    y_steps = list(range(0, height - tile_size + 1, stride))
    if (height - tile_size) % stride != 0:
        y_steps.append(height - tile_size)

    x_steps = list(range(0, width - tile_size + 1, stride))
    if (width - tile_size) % stride != 0:
        x_steps.append(width - tile_size)

    # Open image once
    with rasterio.open(image_path) as src:
        for y in y_steps:
            for x in x_steps:
                # ROI Check: Skip tile if it doesn't contain any tissue
                # Slicing numpy array is fast
                roi = tissue_mask[y : y + tile_size, x : x + tile_size]
                if not np.any(roi):
                    continue

                # Read Image Tile
                window = Window(x, y, tile_size, tile_size)
                # Check channel count (Cite debug_lesson_6)
                if src.count >= 3:
                    img = src.read([1, 2, 3], window=window)
                else:
                    img = src.read(1, window=window)
                    img = np.stack([img, img, img], axis=0)

                # Preprocess: (C, H, W) -> (1, C, H, W), Normalize
                img = np.moveaxis(img, 0, -1)  # H, W, C
                tensor_img = (
                    torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                )
                tensor_img = tensor_img.to(device)

                # Inference
                with torch.no_grad():
                    with autocast():
                        outputs = model(tensor_img)
                        # Deep supervision returns list; take the first (highest res)
                        if isinstance(outputs, list):
                            pred = outputs[0]
                        else:
                            pred = outputs

                        prob = torch.sigmoid(pred).squeeze().cpu().numpy()

                # Accumulate with Gaussian Weighting
                prob_map[y : y + tile_size, x : x + tile_size] += prob * gaussian_window
                weight_map[y : y + tile_size, x : x + tile_size] += gaussian_window

    # 5. Normalize and Threshold
    # Avoid division by zero
    mask = weight_map > 0
    prob_map[mask] /= weight_map[mask]

    # Apply threshold
    binary_mask = (prob_map > Config.THRESHOLD).astype(np.uint8)

    # Mask out non-tissue areas one last time to be clean
    binary_mask = binary_mask * (tissue_mask > 0)

    return binary_mask


def make_submission(
    checkpoint_path=Config.CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_path (str): Path to save the submission CSV.
    """
    print("Starting Submission Generation...")

    # 1. Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Load Model
    model = UnetPlusPlus(
        backbone_name=Config.BACKBONE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 3. Load Test Data
    # We use the Dataset class to get metadata easily
    test_dataset = HubmapDataset(mode="test")

    submission_data = []

    # 4. Inference Loop
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        img_id = sample["id"]
        img_path = sample["image_path"]
        anat_path = sample["anatomical_json_path"]
        h = sample["height"]
        w = sample["width"]

        print(f"Processing {img_id} ({w}x{h})...")

        try:
            # Run sliding window inference
            mask = predict_sliding_window(
                model=model,
                image_path=img_path,
                image_id=img_id,
                anatomical_json_path=anat_path,
                height=h,
                width=w,
                device=device,
            )

            # Encode
            rle = rle_encode(mask)
            submission_data.append({"id": img_id, "predicted": rle})

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            # Append empty prediction on failure to avoid submission errors
            submission_data.append({"id": img_id, "predicted": ""})

    # 5. Save Submission
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
