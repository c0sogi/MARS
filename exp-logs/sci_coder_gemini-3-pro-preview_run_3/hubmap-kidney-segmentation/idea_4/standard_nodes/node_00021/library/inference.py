import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import HuBMAPDataset
from library.model import StainNet
from library.utils import rle_encode


def get_gaussian_window(size, sigma_scale=0.125):
    """
    Generates a 2D Gaussian window for weighted stitching.

    Args:
        size (int): The height/width of the square window.
        sigma_scale (float): Sigma relative to the window size (range [-1, 1]).

    Returns:
        np.ndarray: 2D Gaussian window of shape (size, size).
    """
    # Create coordinate grid from -1 to 1
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Calculate distance from center
    d = np.sqrt(xx**2 + yy**2)

    # Gaussian function: exp(-x^2 / (2*sigma^2))
    # We want the window to decay towards edges.
    # sigma_scale controls the width.
    sigma = 1.0 * sigma_scale  # Adjust as needed, usually heuristic
    # Using a slightly wider sigma to avoid too much drop at edges if overlap is small
    # But with 50% overlap, standard gaussian is fine.
    # Let's use a standard implementation approximation or simple exp.
    # Sigma=0.5 covers [-1, 1] with 2 std devs.
    sigma = 0.5

    window = np.exp(-(d**2) / (2.0 * sigma**2))

    return window.astype(np.float32)


def process_image_prediction(
    image_id, tiles_data, image_shape, device, model, gaussian_weight
):
    """
    Reconstructs a single image from its tiles and returns the RLE.

    Args:
        image_id (str): ID of the image.
        tiles_data (list): List of tuples (prediction_tensor, tile_metadata).
        image_shape (tuple): (height, width) of the full image.
        device (torch.device): Device.
        model (nn.Module): Loaded model.
        gaussian_weight (np.ndarray): Weight map for a single tile.

    Returns:
        str: RLE encoded mask.
    """
    h, w = image_shape

    # Use float32 for accumulation to prevent overflow and maintain precision
    prob_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for pred_sigmoid, meta in tiles_data:
        # pred_sigmoid is numpy array (H_tile, W_tile)
        x, y = meta["x"], meta["y"]

        # Determine dimensions of the current tile (might be smaller at edges if logic allowed,
        # but dataset usually shifts to fit TILE_SIZE. We assume TILE_SIZE here based on config)
        tile_h, tile_w = pred_sigmoid.shape

        # Add to accumulators
        prob_map[y : y + tile_h, x : x + tile_w] += pred_sigmoid * gaussian_weight
        weight_map[y : y + tile_h, x : x + tile_w] += gaussian_weight

    # Normalize
    # Avoid division by zero
    mask = weight_map > 0
    prob_map[mask] /= weight_map[mask]

    # Threshold
    binary_mask = prob_map > Config.THRESHOLD
    binary_mask = binary_mask.astype(np.uint8)

    # Optional: Post-processing (remove small objects)
    if Config.MIN_OBJECT_SIZE > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        for i in range(1, num_labels):  # 0 is background
            if stats[i, cv2.CC_STAT_AREA] < Config.MIN_OBJECT_SIZE:
                binary_mask[labels == i] = 0

    return rle_encode(binary_mask)


def predict_test_set(load_cached_data=True):
    """
    Main inference function. Generates predictions for the test set.

    Args:
        load_cached_data (bool): Whether to use cached dataset tiles.
    """
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 1. Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model = StainNet()
    model.to(device)

    # Load weights
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Prepare Dataset
    print("Initializing Test Dataset...")
    test_dataset = HuBMAPDataset(mode="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Total test tiles: {len(test_dataset)}")

    # 3. Inference Loop
    results = []

    # Gaussian window for stitching
    gaussian_weight = get_gaussian_window(Config.TILE_SIZE)

    # Buffers for the current image being processed
    current_image_id = None
    current_image_tiles = []  # List to store (prediction, metadata)
    current_image_shape = (0, 0)

    # We iterate through the loader. Since shuffle=False, tiles for the same image are contiguous.
    # We need to access the metadata. The dataset stores it in self.tiles.
    # We track the global index.
    global_idx = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)

            # Handle Deep Supervision (returns list)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Apply sigmoid
            probs = torch.sigmoid(outputs)
            probs = probs.cpu().numpy()  # (B, 1, H, W)

            # Process batch
            for b in range(batch_size):
                # Get metadata for this specific tile
                tile_meta = test_dataset.get_tile_metadata(global_idx)
                img_id = tile_meta["id"]
                h_img = tile_meta["h"]
                w_img = tile_meta["w"]

                # Check if we switched to a new image
                if current_image_id is not None and img_id != current_image_id:
                    print(f"Reconstructing mask for {current_image_id}...")
                    rle = process_image_prediction(
                        current_image_id,
                        current_image_tiles,
                        current_image_shape,
                        device,
                        model,
                        gaussian_weight,
                    )
                    results.append({"id": current_image_id, "predicted": rle})

                    # Reset buffers
                    current_image_tiles = []

                # Update current image info
                current_image_id = img_id
                current_image_shape = (h_img, w_img)

                # Store prediction (squeeze channel dim: 1, H, W -> H, W)
                pred_tile = probs[b, 0, :, :]
                current_image_tiles.append((pred_tile, tile_meta))

                global_idx += 1

        # Process the final image
        if current_image_id is not None and current_image_tiles:
            print(f"Reconstructing mask for {current_image_id}...")
            rle = process_image_prediction(
                current_image_id,
                current_image_tiles,
                current_image_shape,
                device,
                model,
                gaussian_weight,
            )
            results.append({"id": current_image_id, "predicted": rle})

    # 4. Save Submission
    print("Saving submission...")
    df_sub = pd.DataFrame(results)

    # Ensure all test IDs are present (even if no tiles were generated due to masking)
    # Load test metadata to get full list of IDs
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    all_test_ids = df_test_meta["id"].unique()

    # Merge to ensure coverage
    df_final = pd.DataFrame({"id": all_test_ids})
    df_final = df_final.merge(df_sub, on="id", how="left")

    # Fill NaN with empty string (no mask)
    df_final["predicted"] = df_final["predicted"].fillna("")

    # Save
    df_final.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_final.head())
