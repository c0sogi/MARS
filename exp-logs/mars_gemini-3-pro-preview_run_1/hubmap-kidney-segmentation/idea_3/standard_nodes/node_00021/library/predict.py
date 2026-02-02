import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
import json
import cv2
import gc
from rasterio.windows import Window
from tqdm import tqdm

from library.config import Config
from library.arch import ResNet34UNetPlusPlus
from library.utils import rle_encode, polygons_to_mask, set_seed


def get_gaussian_window(size, sigma_scale=0.125):
    """
    Generates a 2D Gaussian window for weighting tile predictions.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx * xx + yy * yy)
    sigma = sigma_scale  # Standard deviation relative to window size
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    return g


def predict_slide(model, image_path, anatomical_json_path, device):
    """
    Performs sliding-window inference on a large image with TTA and overlap.
    """
    tile_size = Config.TILE_SIZE
    stride = int(tile_size * 0.5)  # 50% overlap

    # Gaussian weight map
    weight_map = get_gaussian_window(tile_size)
    weight_map = torch.from_numpy(weight_map).float().to(device)

    # Open image
    with rasterio.open(image_path) as src:
        h_img, w_img = src.height, src.width

        # Initialize accumulators on CPU to save VRAM
        # Using float16 for accumulator to save RAM if needed, but float32 is safer for precision
        full_prob = torch.zeros((h_img, w_img), dtype=torch.float32, device="cpu")
        full_weight = torch.zeros((h_img, w_img), dtype=torch.float32, device="cpu")

        # Pre-calculate coordinates
        y_steps = list(range(0, h_img - tile_size, stride))
        if (h_img - tile_size) % stride != 0:
            y_steps.append(h_img - tile_size)
        # Ensure we cover the start if image is small (though unlikely for this dataset)
        if not y_steps and h_img <= tile_size:
            y_steps = [0]

        x_steps = list(range(0, w_img - tile_size, stride))
        if (w_img - tile_size) % stride != 0:
            x_steps.append(w_img - tile_size)
        if not x_steps and w_img <= tile_size:
            x_steps = [0]

        # Normalization constants
        mean = torch.tensor(Config.NORM_MEAN).view(3, 1, 1).to(device)
        std = torch.tensor(Config.NORM_STD).view(3, 1, 1).to(device)

        model.eval()

        with torch.no_grad():
            for y in tqdm(
                y_steps, desc=f"Inference {os.path.basename(image_path)}", leave=False
            ):
                for x in x_steps:
                    # Handle edge cases for small images
                    y = max(0, y)
                    x = max(0, x)

                    # Read tile
                    window = Window(x, y, tile_size, tile_size)
                    img = src.read(window=window)

                    # Ensure 3 channels for consistency
                    if img.shape[0] == 1:
                        img = np.repeat(img, 3, axis=0)

                    # Pad if tile is smaller than expected (at edges of small images)
                    if img.shape[1] != tile_size or img.shape[2] != tile_size:
                        pad_h = tile_size - img.shape[1]
                        pad_w = tile_size - img.shape[2]
                        img = np.pad(
                            img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"
                        )

                    # Preprocess
                    img_tensor = torch.from_numpy(img).float().to(device) / 255.0
                    img_tensor = (img_tensor - mean) / std

                    # Prepare TTA Batch: [Orig, HFlip, VFlip, Rot90]
                    batch = [img_tensor]
                    batch.append(torch.flip(img_tensor, [2]))  # HFlip
                    batch.append(torch.flip(img_tensor, [1]))  # VFlip
                    batch.append(torch.rot90(img_tensor, 1, [1, 2]))  # Rot90

                    batch_tensor = torch.stack(batch)  # (4, 3, H, W)

                    # Inference
                    logits = model(batch_tensor)
                    probs = torch.sigmoid(logits).squeeze(1)  # (4, H, W)

                    # Inverse TTA
                    p_orig = probs[0]
                    p_hflip = torch.flip(probs[1], [1])
                    p_vflip = torch.flip(probs[2], [0])
                    p_rot90 = torch.rot90(probs[3], -1, [0, 1])

                    # Average
                    avg_prob = (p_orig + p_hflip + p_vflip + p_rot90) / 4.0

                    # Accumulate
                    # Move to CPU for accumulation
                    avg_prob_cpu = avg_prob.cpu()
                    weight_map_cpu = weight_map.cpu()

                    # Crop back if we padded
                    valid_h = min(tile_size, h_img - y)
                    valid_w = min(tile_size, w_img - x)

                    full_prob[y : y + valid_h, x : x + valid_w] += (
                        avg_prob_cpu[:valid_h, :valid_w]
                        * weight_map_cpu[:valid_h, :valid_w]
                    )
                    full_weight[y : y + valid_h, x : x + valid_w] += weight_map_cpu[
                        :valid_h, :valid_w
                    ]

        # Normalize accumulated probabilities
        full_weight[full_weight == 0] = 1.0  # Avoid div by zero
        full_prob /= full_weight

        # Anatomical Filtering (ROI)
        if Config.USE_ANATOMICAL_FILTER and os.path.exists(anatomical_json_path):
            try:
                with open(anatomical_json_path, "r") as f:
                    anat_data = json.load(f)

                # Extract Cortex polygons
                cortex_polys = []
                for feat in anat_data:
                    props = feat.get("properties", {})
                    classification = props.get("classification", {})
                    if classification.get("name") == "Cortex":
                        cortex_polys.append(feat["geometry"]["coordinates"])

                if cortex_polys:
                    # Generate mask (1 inside Cortex, 0 outside)
                    # polygons_to_mask returns uint8
                    roi_mask = polygons_to_mask(cortex_polys, (h_img, w_img))
                    roi_mask_tensor = torch.from_numpy(roi_mask).float()
                    full_prob *= roi_mask_tensor
            except Exception as e:
                print(
                    f"Warning: Failed to apply anatomical filter for {image_path}: {e}"
                )

        # Threshold
        mask = (full_prob > Config.MASK_THRESHOLD).numpy().astype(np.uint8)

        return mask


def generate_submission():
    """
    Main function to generate the submission file.
    """
    set_seed(Config.SEED)

    # Load Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Setup Device
    device = torch.device(Config.DEVICE)

    # Load Model
    model = ResNet34UNetPlusPlus(in_channels=3, classes=1)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    submission_rows = []

    print(f"Generating predictions for {len(test_df)} images...")

    for _, row in test_df.iterrows():
        image_id = row["id"]
        image_path = row["image_path"]
        anat_path = row["anatomical_json_path"]

        # Run Inference
        pred_mask = predict_slide(model, image_path, anat_path, device)

        # Encode
        rle = rle_encode(pred_mask)
        submission_rows.append({"id": image_id, "predicted": rle})

        # Cleanup
        del pred_mask
        gc.collect()

    # Save Submission
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
