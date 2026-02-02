import os
import cv2
import torch
import rasterio
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config, seed_everything
from library.model import FPNResNet18
from library.utils import rle_encode, get_tissue_mask


class InferenceDataset(Dataset):
    """
    Dataset for sliding window inference on a WSI.
    Reads specific tiles defined by (x, y) coordinates.
    """

    def __init__(self, image_path, tile_coords, transform=None):
        self.image_path = image_path
        self.tile_coords = tile_coords  # List of (x, y) tuples
        self.transform = transform

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        x, y = self.tile_coords[idx]

        # Read the specific window
        window = rasterio.windows.Window(
            col_off=x, row_off=y, width=Config.TILE_SIZE, height=Config.TILE_SIZE
        )

        with rasterio.open(self.image_path) as src:
            # boundless=True pads with fill_value if window is out of bounds
            img = src.read(window=window, boundless=True, fill_value=0)

            # Handle channel dimensions
            if src.count == 1:
                img = np.repeat(img, 3, axis=0)
            elif src.count > 3:
                img = img[:3, :, :]

        # Convert to HWC for Albumentations
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, x, y


def get_inference_transforms():
    """
    Returns the normalization transforms for inference.
    """
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def predict_wsi(
    model, image_path, anat_path, device, threshold=0.5, batch_size=Config.BATCH_SIZE
):
    """
    Performs inference on a single Whole Slide Image (WSI).

    Args:
        model: Trained PyTorch model.
        image_path: Path to the TIFF image.
        anat_path: Path to the anatomical structure JSON.
        device: Torch device.
        threshold: Probability threshold for binary mask.
        batch_size: Batch size for inference.

    Returns:
        str: RLE encoded mask.
    """
    # 1. Get Image Dimensions
    with rasterio.open(image_path) as src:
        H, W = src.height, src.width

    # 2. Get Tissue Mask to define ROI
    # We use the utility which handles caching internally
    tissue_mask = get_tissue_mask(
        anat_path, (H, W), valid_classes=["Cortex", "Medulla"], load_cached_data=True
    )

    # 3. Generate Valid Tile Coordinates
    # We only predict on tiles that intersect with the tissue mask
    tile_coords = []

    # We iterate with the defined stride
    for y in range(0, H, Config.STRIDE):
        for x in range(0, W, Config.STRIDE):
            # Check intersection
            y_end = min(y + Config.TILE_SIZE, H)
            x_end = min(x + Config.TILE_SIZE, W)

            mask_crop = tissue_mask[y:y_end, x:x_end]

            # If there is any tissue in this crop, we process it
            # Using a very low threshold to be safe during inference
            if np.any(mask_crop):
                tile_coords.append((x, y))

    # If no tissue found, return empty RLE
    if not tile_coords:
        return ""

    # 4. Prepare Dataset and Loader
    dataset = InferenceDataset(
        image_path=image_path,
        tile_coords=tile_coords,
        transform=get_inference_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Inference Loop with Stitching
    # Allocate large arrays for probability accumulation
    # Using float32 for probabilities and uint8 for counts (max overlap won't exceed 255)
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint8)

    model.eval()

    with torch.no_grad():
        for images, xs, ys in loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Move to CPU
            probs = probs.squeeze(1).cpu().numpy()  # (B, H_tile, W_tile)
            xs = xs.numpy()
            ys = ys.numpy()

            # Stitch
            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                prob_tile = probs[i]

                # Calculate valid region for this tile (handle image edges)
                h_tile, w_tile = prob_tile.shape

                y_end = min(y + h_tile, H)
                x_end = min(x + w_tile, W)

                # Crop the tile prediction if it extends beyond image bounds
                # (Though rasterio padded reading usually handles this,
                # we need to map back to original H, W)
                h_valid = y_end - y
                w_valid = x_end - x

                prob_tile_valid = prob_tile[:h_valid, :w_valid]

                # Accumulate
                prob_map[y:y_end, x:x_end] += prob_tile_valid
                count_map[y:y_end, x:x_end] += 1

    # 6. Normalize and Threshold
    # Avoid division by zero
    mask = count_map > 0
    prob_map[mask] /= count_map[mask]

    binary_mask = (prob_map > threshold).astype(np.uint8)

    # 7. Encode
    rle = rle_encode(binary_mask)

    # Clean up memory
    del prob_map, count_map, binary_mask, tissue_mask
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    return rle


def generate_submission_csv(debug=False):
    """
    Generates the submission.csv file for the test set.
    """
    seed_everything(Config.SEED)

    # 1. Load Test Metadata
    test_df = pd.read_csv(Config.TEST_METADATA)

    if debug:
        test_df = test_df.head(1)

    print(f"Generating predictions for {len(test_df)} test images...")

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = FPNResNet18()

    # Load weights if available
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)

    # 3. Prediction Loop
    results = []

    for _, row in test_df.iterrows():
        image_id = row["id"]
        image_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        anat_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])

        print(f"Processing {image_id}...")

        try:
            rle = predict_wsi(
                model=model, image_path=image_path, anat_path=anat_path, device=device
            )
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            rle = ""

        results.append({"id": image_id, "predicted": rle})

    # 4. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure column order matches sample submission
    submission_df = submission_df[["id", "predicted"]]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
