import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.utils import set_seed, get_logger, rle_encode
from library.data_processing import (
    rasterize_json_polygons,
    get_tile_coordinates,
)
from library.dataset import HubmapDataset, get_transforms
from library.model import build_model
from library.training import get_gaussian_window

# Initialize Logger
logger = get_logger("Inference")


def predict_slide(
    model,
    image_metadata,
    tile_size=1024,
    overlap=0.5,
    batch_size=4,
    device="cuda",
    do_normalization=True,
):
    """
    Performs sliding-window inference on a single full-resolution image.

    Args:
        model (torch.nn.Module): Trained model.
        image_metadata (pd.Series): Row from the test metadata CSV containing paths and dims.
        tile_size (int): Size of the tiles.
        overlap (float): Overlap fraction.
        batch_size (int): Inference batch size.
        device (str): Device to run inference on.
        do_normalization (bool): Whether to apply Macenko normalization.

    Returns:
        str: RLE encoded prediction mask.
    """
    image_id = image_metadata["id"]
    # Paths are relative to input root in metadata, but we need to handle them correctly.
    # The metadata generator stored them as "test/filename.tiff" (relative to input/).
    # We prepend "./input" to access the file.
    image_path = os.path.join("./input", image_metadata["image_path"])
    anat_path = os.path.join("./input", image_metadata["anatomical_json_path"])

    # Dimensions
    h_img = int(image_metadata["height_pixels"])
    w_img = int(image_metadata["width_pixels"])

    logger.info(f"Processing {image_id} ({w_img}x{h_img})...")

    # 1. Generate Anatomical Mask (ROI)
    # We check Cortex first, then Medulla, similar to training logic
    roi_mask = rasterize_json_polygons(anat_path, (h_img, w_img), filter_name="Cortex")
    if roi_mask.sum() == 0:
        logger.info(f"No Cortex found for {image_id}, checking Medulla...")
        roi_mask = rasterize_json_polygons(
            anat_path, (h_img, w_img), filter_name="Medulla"
        )
        if roi_mask.sum() == 0:
            logger.warning(
                f"No anatomical structure found for {image_id}. Using whole image."
            )
            roi_mask = np.ones((h_img, w_img), dtype=np.uint8)

    # 2. Generate Tile Coordinates
    # We use a lower threshold for inference to ensure we cover edges of the tissue
    coords = get_tile_coordinates(roi_mask, tile_size, overlap, threshold=0.01)

    if not coords:
        logger.warning(f"No valid tiles found for {image_id}. Returning empty mask.")
        return ""

    # 3. Create Temporary Dataset for this Slide
    # We construct a DataFrame representing the tiles for this single image
    tile_data = []
    for x, y in coords:
        tile_data.append(
            {
                "id": image_id,
                "image_path": image_metadata["image_path"],  # relative to input/
                "x": x,
                "y": y,
                "w": tile_size,
                "h": tile_size,
            }
        )

    df_tiles = pd.DataFrame(tile_data)

    # Transform (Normalize only)
    test_transform = get_transforms(mode="test", img_size=tile_size)

    dataset = HubmapDataset(
        tile_df=df_tiles,
        metadata_df=None,
        transform=test_transform,
        do_normalization=do_normalization,
        mode="test",
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 4. Inference with Gaussian Stitching
    # Initialize buffers
    # Using float16 for probability buffer to save memory on large images
    prob_map = np.zeros((h_img, w_img), dtype=np.float16)
    weight_map = np.zeros((h_img, w_img), dtype=np.float16)

    gaussian_weight = get_gaussian_window(tile_size)

    model.eval()
    with torch.no_grad():
        # Iterate over batches
        # We need to track which tiles correspond to which batch
        # The loader yields (images, masks) - masks are dummy here
        # We iterate the dataframe index alongside

        tile_idx = 0
        for images, _ in loader:
            images = images.to(device)

            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs)  # (B, 1, H, W)
            preds_np = preds.squeeze(1).cpu().numpy()

            batch_len = images.size(0)

            for i in range(batch_len):
                # Get coordinates for this tile
                row = df_tiles.iloc[tile_idx]
                x, y = row["x"], row["y"]

                # Determine valid region (handle potential edge cropping if logic changed,
                # though HubmapDataset usually returns fixed size)
                h_pred, w_pred = preds_np[i].shape

                # Boundary checks
                y_end = min(y + h_pred, h_img)
                x_end = min(x + w_pred, w_img)

                h_eff = y_end - y
                w_eff = x_end - x

                if h_eff > 0 and w_eff > 0:
                    # Accumulate
                    prob_map[y:y_end, x:x_end] += (
                        preds_np[i, :h_eff, :w_eff] * gaussian_weight[:h_eff, :w_eff]
                    ).astype(np.float16)
                    weight_map[y:y_end, x:x_end] += gaussian_weight[
                        :h_eff, :w_eff
                    ].astype(np.float16)

                tile_idx += 1

    # 5. Finalize Mask
    # Avoid division by zero
    mask_valid = weight_map > 0
    prob_map[mask_valid] /= weight_map[mask_valid]

    # Threshold
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    # Encode
    rle = rle_encode(binary_mask)

    # Clean up
    del prob_map, weight_map, loader, dataset, df_tiles
    gc.collect()

    return rle


def run_inference(
    model_path,
    output_path="./submission/submission.csv",
    tile_size=1024,
    overlap=0.5,
    batch_size=4,
    do_normalization=True,
):
    """
    Main driver for inference pipeline.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load Metadata
    test_meta_path = "./metadata/test.csv"
    if not os.path.exists(test_meta_path):
        logger.error(f"Metadata file not found: {test_meta_path}")
        return

    df_test = pd.read_csv(test_meta_path)
    logger.info(f"Found {len(df_test)} test images.")

    # 2. Load Model
    logger.info(f"Loading model from {model_path}...")
    model = build_model(encoder_name="convnext_base", classes=1)

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        # Create a dummy file if model missing to prevent pipeline crash in some envs,
        # though this should fail in a real run.
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 3. Run Inference per Image
    submission_rows = []

    for idx, row in df_test.iterrows():
        try:
            rle = predict_slide(
                model=model,
                image_metadata=row,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
                device=device,
                do_normalization=do_normalization,
            )
            submission_rows.append({"id": row["id"], "predicted": rle})
        except Exception as e:
            logger.error(f"Error processing image {row['id']}: {e}")
            # Append empty prediction on failure
            submission_rows.append({"id": row["id"], "predicted": ""})

    # 4. Save Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
