import os
import cv2
import torch
import rasterio
import numpy as np
import pandas as pd
import scipy.signal
from torch.cuda.amp import autocast
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed, rle_encode, get_tissue_mask
from library.model import build_model


def get_gaussian_window(size, sigma_scale=8):
    """
    Generates a 2D Gaussian window for weighting tile predictions.
    """
    x = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    gauss = np.exp(-0.5 * np.square(x) / np.square(size / sigma_scale))
    window = np.outer(gauss, gauss)
    return window.astype(np.float32)


def remove_small_objects(mask, min_size):
    """
    Removes connected components smaller than min_size.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # stats[:, 4] is the area
    # Label 0 is background, so we ignore it in the loop or check index
    # We create a new mask keeping only valid components
    new_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 1

    return new_mask


def predict_image(model, image_path, anatomical_json_path, image_id, device):
    """
    Performs sliding window inference on a single image.
    """
    # 1. Open Image to get dimensions
    with rasterio.open(image_path) as src:
        H, W = src.height, src.width
        # We don't read the whole image here to save memory, we read tiles later.

    # 2. Get Tissue Mask (ROI)
    # This uses the Fail-Open logic defined in utils
    tissue_mask = get_tissue_mask(
        image_id, W, H, anatomical_json_path, load_cached_data=True
    )

    # 3. Setup Sliding Window
    # Use Phase 2 tile size (High Precision)
    tile_size = Config.PHASE2["TILE_SIZE"]
    overlap = Config.INFERENCE_OVERLAP
    stride = int(tile_size * (1 - overlap))

    # Gaussian Window
    gaussian_weight = get_gaussian_window(tile_size)
    gaussian_weight = torch.from_numpy(gaussian_weight).to(device)

    # Accumulators (using float16/32 to fit in RAM)
    # 50k x 50k float32 is ~10GB. We have 220GB RAM, so this is safe.
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Preprocessing transform (Normalize only)
    transforms = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # 4. Iterate over tiles
    # Generate coordinates
    x_steps = [x for x in range(0, W - tile_size + stride, stride)]
    y_steps = [y for y in range(0, H - tile_size + stride, stride)]

    # Adjust last steps to ensure coverage of edges
    if x_steps[-1] + tile_size < W:
        x_steps.append(W - tile_size)
    if y_steps[-1] + tile_size < H:
        y_steps.append(H - tile_size)

    # Ensure unique coordinates to avoid redundant processing
    x_steps = sorted(list(set(x_steps)))
    y_steps = sorted(list(set(y_steps)))

    model.eval()

    # Open rasterio handle once
    with rasterio.open(image_path) as src:
        for y in y_steps:
            for x in x_steps:
                # ROI Check: If tile is completely outside tissue mask, skip
                # We check the tissue mask crop
                tissue_crop = tissue_mask[y : y + tile_size, x : x + tile_size]
                if np.sum(tissue_crop) == 0:
                    continue

                # Read Image Tile
                window = Window(x, y, tile_size, tile_size)
                # boundless=True handles padding if we were going off edge,
                # but our step logic keeps us inside.
                if src.count >= 3:
                    img = src.read(
                        [1, 2, 3], window=window, boundless=True, fill_value=0
                    )
                else:
                    img = src.read([1], window=window, boundless=True, fill_value=0)
                    img = np.repeat(img, 3, axis=0)

                img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C

                # Transform
                augmented = transforms(image=img)
                img_tensor = augmented["image"].unsqueeze(0).to(device)  # 1, C, H, W

                # Inference
                with torch.no_grad():
                    with autocast():
                        outputs = model(img_tensor)
                        # Handle Deep Supervision output (list)
                        if isinstance(outputs, list):
                            pred = outputs[0]
                        else:
                            pred = outputs

                        pred_sigmoid = torch.sigmoid(pred).squeeze().float()  # H, W

                # Accumulate
                # Move to CPU for accumulation to save GPU memory
                pred_np = (pred_sigmoid * gaussian_weight).cpu().numpy()
                weight_np = gaussian_weight.cpu().numpy()

                prob_map[y : y + tile_size, x : x + tile_size] += pred_np
                weight_map[y : y + tile_size, x : x + tile_size] += weight_np

    # 5. Normalize and Threshold
    # Avoid division by zero
    mask = np.zeros((H, W), dtype=np.uint8)
    valid_pixels = weight_map > 0

    prob_map[valid_pixels] /= weight_map[valid_pixels]
    mask[prob_map > Config.PREDICTION_THRESHOLD] = 1

    # 6. Post-Processing
    if Config.MIN_PIXEL_SIZE > 0:
        mask = remove_small_objects(mask, Config.MIN_PIXEL_SIZE)

    return mask


def run_inference():
    """
    Main execution function for inference.
    Generates predictions for the test set and saves submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Metadata
    # We use the test metadata generated by the metadata script
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    print(f"Loaded test metadata with {len(df_test)} images.")

    # 2. Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = build_model()
    model = model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    else:
        print(
            "WARNING: Model checkpoint not found. Using random weights (expect poor performance)."
        )

    # 3. Inference Loop
    submission_data = []

    for idx, row in df_test.iterrows():
        img_id = row["id"]

        # Construct full paths based on metadata
        # Metadata paths are relative to input dir (e.g., "test/image.tiff")
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Anatomical json might be NaN if not present
        anat_path = row.get("anatomical_json_path", None)
        if pd.isna(anat_path):
            anat_path = None

        print(f"Predicting image {idx+1}/{len(df_test)}: {img_id}")

        try:
            mask = predict_image(model, img_path, anat_path, img_id, device)
            rle = rle_encode(mask)
        except Exception as e:
            print(f"Error predicting {img_id}: {e}")
            # Fallback: empty prediction
            rle = ""

        submission_data.append({"id": img_id, "predicted": rle})

    # 4. Save Submission
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    submission_df = submission_df[["id", "predicted"]]

    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
