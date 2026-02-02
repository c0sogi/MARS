import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.signal
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from library.utils import CFG, rle_encode
from library.model import UNetPlusPlus
from library.data_processing import (
    read_tiff,
    read_tiff_region,
    get_anatomical_mask,
    get_tile_coordinates,
)


def get_gaussian_kernel(size=1024, sigma=None):
    """
    Generates a 2D Gaussian kernel for weighting tile overlaps.
    """
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    g = scipy.signal.windows.gaussian(size, std=sigma)
    gaussian_2d = np.outer(g, g)
    return gaussian_2d


def predict_tile(model, image, device):
    """
    Predicts a single tile with Test-Time Augmentation (TTA).
    TTA: Original, Horizontal Flip, Vertical Flip.
    """
    # Define transform for normalization
    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    # Preprocess
    augmented = transform(image=image)
    img_tensor = augmented["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        # 1. Original
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)

        # 2. Horizontal Flip
        img_h = torch.flip(img_tensor, dims=[3])
        logits_h = model(img_h)
        prob_h = torch.flip(torch.sigmoid(logits_h), dims=[3])

        # 3. Vertical Flip
        img_v = torch.flip(img_tensor, dims=[2])
        logits_v = model(img_v)
        prob_v = torch.flip(torch.sigmoid(logits_v), dims=[2])

        # Average
        avg_prob = (prob + prob_h + prob_v) / 3.0

    return avg_prob.squeeze().cpu().numpy()  # (H, W)


def predict_whole_image(model, image_path, anatomical_json_path, device):
    """
    Performs tiled inference on a whole image using anatomical masks and Gaussian stitching.
    """
    # Open image to get dimensions
    with read_tiff(image_path) as src:
        h_img, w_img = src.height, src.width

    # Get anatomical mask to filter tiles
    # Extract ID from filename for caching purposes
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # We use the cache directory defined in CFG
    # Note: anatomical_json_path might be NaN or invalid for some test sets if not provided,
    # but the problem description implies they are available.
    if pd.isna(anatomical_json_path):
        # Fallback: process everything if no anatomical structure file
        mask_shape = (h_img, w_img)
        roi_mask = np.ones(mask_shape, dtype=np.uint8)
    else:
        roi_mask = get_anatomical_mask(
            image_id,
            anatomical_json_path,
            (h_img, w_img),
            CFG.cache_dir,
            load_cached_data=True,
        )

    # Get tile coordinates
    # We use a strict overlap for inference to ensure smooth stitching
    tile_size = CFG.img_size
    overlap = tile_size // 2  # 50% overlap for high quality stitching

    coords = get_tile_coordinates(roi_mask, tile_size, overlap, threshold=0.01)

    # Initialize accumulators
    # Using float16 to save memory if needed, but float32 is safer for precision
    full_prob = np.zeros((h_img, w_img), dtype=np.float32)
    weight_map = np.zeros((h_img, w_img), dtype=np.float32)

    # Precompute Gaussian kernel
    gaussian_kernel = get_gaussian_kernel(tile_size, sigma=tile_size / 4)

    # Open image for reading tiles
    with read_tiff(image_path) as src:
        for coord in coords:
            x, y = coord["x"], coord["y"]
            w, h = tile_size, tile_size

            # Read tile
            tile = read_tiff_region(src, x, y, w, h)

            # Handle edge cases where tile might be smaller than tile_size
            # (though get_tile_coordinates logic usually handles this by shifting back)
            # If read_tiff_region returns smaller tile (e.g. at boundaries if logic differs), pad it.
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                pad_h = tile_size - tile.shape[0]
                pad_w = tile_size - tile.shape[1]
                tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

            # Ensure 3 channels
            if tile.shape[2] == 1:
                tile = np.repeat(tile, 3, axis=2)
            elif tile.shape[2] > 3:
                tile = tile[:, :, :3]

            # Predict
            prob_tile = predict_tile(model, tile, device)

            # Accumulate
            # We need to handle the padding if we padded the input
            # But since we stitch back to (x, y), we just take the valid region
            # However, our coordinate logic shifts back to fit tile_size, so we always have full tiles
            # except potentially very small images (unlikely here).

            full_prob[y : y + tile_size, x : x + tile_size] += (
                prob_tile * gaussian_kernel
            )
            weight_map[y : y + tile_size, x : x + tile_size] += gaussian_kernel

    # Normalize
    # Avoid division by zero
    mask = weight_map > 0
    full_prob[mask] /= weight_map[mask]

    # Threshold
    binary_mask = (full_prob > CFG.threshold).astype(np.uint8)

    return binary_mask


def generate_submission(model_path, output_dir="./submission"):
    """
    Generates the submission.csv file for the test set.
    """
    device = CFG.device
    print(f"Inference using device: {device}")

    # Load Metadata
    test_df = pd.read_csv(CFG.test_metadata_path)

    # Load Model
    model = UNetPlusPlus(
        backbone_name=CFG.backbone,
        in_channels=3,
        classes=CFG.num_classes,
        pretrained=False,  # No need to download weights, loading state_dict
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Error: Model weights not found at {model_path}")
        return

    model.to(device)
    model.eval()

    submission_data = []

    print(f"Processing {len(test_df)} test images...")

    for idx, row in test_df.iterrows():
        img_id = row["id"]
        # Construct paths based on metadata relative paths
        # Metadata paths are like "test/id.tiff" or "input/test/id.tiff" depending on generation
        # CFG.input_root is "./input".
        # If metadata path is "test/id.tiff", join yields "./input/test/id.tiff"
        img_path = os.path.join(CFG.input_root, row["image_path"])

        # Anatomical path
        if "anatomical_json_path" in row and pd.notna(row["anatomical_json_path"]):
            anat_path = os.path.join(CFG.input_root, row["anatomical_json_path"])
        else:
            anat_path = None

        print(f"Predicting {img_id}...")

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}. Skipping.")
            submission_data.append({"id": img_id, "predicted": ""})
            continue

        try:
            # Predict
            binary_mask = predict_whole_image(model, img_path, anat_path, device)

            # Encode
            rle = rle_encode(binary_mask)
            submission_data.append({"id": img_id, "predicted": rle})

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            submission_data.append({"id": img_id, "predicted": ""})

    # Save Submission
    os.makedirs(output_dir, exist_ok=True)
    sub_df = pd.DataFrame(submission_data)

    # Ensure columns are correct
    # The sample submission has 'id' and 'predicted'
    sub_df = sub_df[["id", "predicted"]]

    output_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference():
    # Define model path
    model_path = os.path.join(CFG.cache_dir, "best_model.pth")

    # Generate submission
    generate_submission(model_path)
