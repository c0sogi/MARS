import os
import numpy as np
import pandas as pd
import rasterio
import torch
import cv2
import torch.nn.functional as F

from library.config import Config
from library.utils import rle_encode, polygons_to_mask
from library.model import AttentionUNetResNet34


def get_gaussian_kernel(size, sigma_scale=1 / 8):
    """
    Generates a 2D Gaussian kernel for weighting tile predictions.

    Args:
        size (int): The height/width of the square kernel.
        sigma_scale (float): Scaling factor for standard deviation relative to size.

    Returns:
        np.ndarray: 2D Gaussian kernel of shape (size, size).
    """
    # Create a 1D Gaussian distribution
    x = np.linspace(-1, 1, size)
    sigma = sigma_scale  # relative to the range [-1, 1]
    gauss_1d = np.exp(-0.5 * (x / sigma) ** 2)

    # Create 2D kernel via outer product
    gauss_2d = np.outer(gauss_1d, gauss_1d)

    # Normalize so max is 1.0 (weighting factor)
    gauss_2d /= gauss_2d.max()

    return gauss_2d.astype(np.float32)


def predict_sliding_window(image, model, device, tile_size, overlap=0.5):
    """
    Performs sliding window inference on a large 4-channel image.

    Args:
        image (np.ndarray): Input image of shape (H, W, 4).
        model (torch.nn.Module): Trained model.
        device (torch.device): Compute device.
        tile_size (int): Size of the tiles.
        overlap (float): Overlap fraction between tiles (0.0 to 1.0).

    Returns:
        np.ndarray: Probability map of shape (H, W).
    """
    h, w, c = image.shape

    # Calculate stride
    stride = int(tile_size * (1 - overlap))

    # Pad image to ensure we can cover the edges with full tiles
    pad_h = (tile_size - (h % stride)) % stride + (tile_size - stride)
    pad_w = (tile_size - (w % stride)) % stride + (tile_size - stride)

    # If the image is smaller than the tile size, we need more padding
    if h < tile_size:
        pad_h = tile_size - h
    if w < tile_size:
        pad_w = tile_size - w

    # Pad input image
    # (top, bottom), (left, right), (channels_before, channels_after)
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    padded_h, padded_w, _ = image_padded.shape

    # Initialize accumulators
    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    # Gaussian weighting kernel
    kernel = get_gaussian_kernel(tile_size)
    kernel_tensor = torch.from_numpy(kernel).to(device)

    # Normalization constants (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for y in range(0, padded_h - tile_size + 1, stride):
            for x in range(0, padded_w - tile_size + 1, stride):
                # Extract tile
                tile = image_padded[y : y + tile_size, x : x + tile_size, :]

                # Preprocess Tile
                # 1. Normalize RGB
                rgb = tile[..., :3].astype(np.float32) / 255.0
                rgb = (rgb - mean) / std

                # 2. Prepare Anatomy Channel
                anatomy = tile[..., 3].astype(np.float32)
                anatomy = np.expand_dims(anatomy, axis=-1)

                # 3. Recombine
                tile_tensor_np = np.concatenate([rgb, anatomy], axis=-1)

                # 4. To Tensor (C, H, W)
                tile_tensor = torch.from_numpy(
                    tile_tensor_np.transpose(2, 0, 1)
                ).float()
                tile_tensor = tile_tensor.unsqueeze(0).to(device)  # Add batch dim

                # Inference
                logits = model(tile_tensor)
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # (H, W)

                # Accumulate
                # Move probs to CPU for accumulation to save GPU memory on large maps
                # Or keep on GPU if memory allows. Here we accumulate on CPU to be safe with 2.5GB images.
                # Actually, accumulating on CPU is safer for the full map.

                # Apply Gaussian weight
                weighted_probs = probs * kernel_tensor

                # Transfer to CPU
                weighted_probs_np = weighted_probs.cpu().numpy()

                prob_map[y : y + tile_size, x : x + tile_size] += weighted_probs_np
                weight_map[y : y + tile_size, x : x + tile_size] += kernel

    # Normalize by weights
    # Avoid division by zero
    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map

    # Crop back to original size
    prob_map = prob_map[:h, :w]

    return prob_map


def process_test_image(row, model, device):
    """
    Loads, processes, and predicts a mask for a single test image.

    Args:
        row (pd.Series): Metadata row containing file paths.
        model (torch.nn.Module): Loaded model.
        device (torch.device): Compute device.

    Returns:
        str: RLE encoded mask.
    """
    img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
    anat_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])

    print(f"Processing {row['id']}...")

    # 1. Load Image
    try:
        with rasterio.open(img_path) as src:
            image = src.read().transpose(1, 2, 0)  # (H, W, C)
            if image.shape[2] > 3:
                image = image[:, :, :3]
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return "nan"

    h, w = image.shape[:2]

    # 2. Load Anatomical Mask (Cortex)
    # This serves as the 4th channel AND the post-processing filter
    cortex_mask = polygons_to_mask(anat_path, (h, w), label_name="Cortex")

    # 3. Create 4-Channel Input
    cortex_channel = np.expand_dims(cortex_mask, axis=-1)  # (H, W, 1)
    input_image = np.concatenate([image, cortex_channel], axis=2)  # (H, W, 4)

    # 4. Predict
    prob_map = predict_sliding_window(
        input_image,
        model,
        device,
        tile_size=Config.TILE_SIZE,
        overlap=Config.OVERLAP_STRIDE,
    )

    # 5. Anatomical Post-Filter
    # Zero out predictions outside the cortex
    prob_map = prob_map * cortex_mask

    # 6. Threshold
    binary_mask = (prob_map > Config.PREDICTION_THRESHOLD).astype(np.uint8)

    # 7. RLE Encode
    rle = rle_encode(binary_mask)

    return rle


def generate_submission():
    """
    Main function to generate the submission file for the test set.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Load Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    print(f"Found {len(test_df)} test images.")

    # 3. Load Model
    model = AttentionUNetResNet34(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # No need to download weights, we load checkpoint
    )

    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(f"Loading model weights from {Config.MODEL_CHECKPOINT_PATH}...")
        state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    for idx, row in test_df.iterrows():
        rle = process_test_image(row, model, device)
        results.append({"id": row["id"], "predicted": rle})

    # 5. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    submission_df = submission_df[["id", "predicted"]]

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
