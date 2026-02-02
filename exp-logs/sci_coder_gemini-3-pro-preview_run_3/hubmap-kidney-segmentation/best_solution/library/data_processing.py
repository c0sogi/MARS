import os
import json
import numpy as np
import pandas as pd
import rasterio
import cv2
from rasterio.windows import Window
from library.utils import CFG


def read_tiff(image_path):
    """
    Opens a TIFF file using rasterio.

    Args:
        image_path (str): Path to the TIFF file.

    Returns:
        rasterio.io.DatasetReader: Opened dataset handle.
    """
    return rasterio.open(image_path)


def read_tiff_region(src, x, y, w, h):
    """
    Reads a specific region from an opened rasterio dataset.

    Args:
        src: Opened rasterio dataset.
        x, y (int): Top-left coordinates.
        w, h (int): Width and height of the region.

    Returns:
        np.ndarray: Image data in (H, W, C) format.
    """
    # Rasterio reads as (C, H, W), we transpose to (H, W, C)
    window = Window(x, y, w, h)
    img = src.read(window=window)
    return np.moveaxis(img, 0, -1)


def parse_anatomical_json(json_path):
    """
    Parses the anatomical structure JSON to extract Cortex and Medulla polygons.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: List of polygons (list of points) for relevant structures.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    polygons = []
    for feature in data:
        # Check classification
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        name = classification.get("name", "")

        # We are interested in Cortex and Medulla
        if name in ["Cortex", "Medulla"]:
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])

            # Coordinates are often nested: [[[x,y], ...]]
            for poly_coords in coords:
                pts = np.array(poly_coords, dtype=np.int32)
                polygons.append(pts)

    return polygons


def rasterize_mask(polygons, shape):
    """
    Converts a list of polygons into a binary mask.

    Args:
        polygons (list): List of numpy arrays representing polygons.
        shape (tuple): (height, width) of the target mask.

    Returns:
        np.ndarray: Binary mask (uint8).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask


def get_anatomical_mask(image_id, json_path, shape, cache_dir, load_cached_data=True):
    """
    Retrieves the anatomical mask, either from cache or by generating it.

    Args:
        image_id (str): Unique image identifier.
        json_path (str): Path to the anatomical JSON.
        shape (tuple): (height, width) of the image.
        cache_dir (str): Directory to store cached masks.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Binary anatomical mask.
    """
    mask_dir = os.path.join(cache_dir, "tissue_masks_cache")
    os.makedirs(mask_dir, exist_ok=True)

    # Include shape in filename to avoid mismatches if metadata changes
    filename = f"{image_id}_{shape[0]}x{shape[1]}_Cortex_Medulla.npy"
    file_path = os.path.join(mask_dir, filename)

    if load_cached_data and os.path.exists(file_path):
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Failed to load cached mask for {image_id}: {e}. Regenerating.")

    # Generate mask
    if pd.isna(json_path) or not os.path.exists(json_path):
        # If no JSON, assume entire image is valid (or empty? Task implies filtering,
        # but for test images without anatomical structure, we might need full image.
        # However, task description says test sets include anatomical structure segmentations.)
        # If strictly missing, we return ones (process everything) or zeros.
        # Given the prompt context, anatomical structures are provided to help.
        # If missing, we default to ones (keep all tiles) to be safe.
        print(
            f"Warning: Anatomical JSON not found for {image_id}. Returning full mask."
        )
        mask = np.ones(shape, dtype=np.uint8)
    else:
        polygons = parse_anatomical_json(json_path)
        mask = rasterize_mask(polygons, shape)

    # Save to cache
    np.save(file_path, mask)
    return mask


def get_tile_coordinates(mask, tile_size, overlap, threshold=0.1):
    """
    Generates tile coordinates that intersect with the tissue mask.

    Args:
        mask (np.ndarray): Binary tissue mask.
        tile_size (int): Size of the square tile.
        overlap (int): Overlap between tiles in pixels.
        threshold (float): Minimum fraction of mask pixels required to keep tile.

    Returns:
        list: List of dicts {'x': int, 'y': int}.
    """
    h, w = mask.shape
    step = tile_size - overlap
    coordinates = []

    # If mask is empty, return empty list
    if mask.sum() == 0:
        return coordinates

    # Generate grid
    # We ensure we cover the edges by taking a ceiling division or checking bounds
    x_points = range(0, w, step)
    y_points = range(0, h, step)

    for y in y_points:
        for x in x_points:
            # Adjust if tile goes out of bounds (shift back)
            x_eff = min(x, w - tile_size)
            y_eff = min(y, h - tile_size)

            # Ensure non-negative (if image is smaller than tile_size)
            x_eff = max(0, x_eff)
            y_eff = max(0, y_eff)

            # Extract mask region
            # Handle case where image < tile_size
            eff_h = min(tile_size, h - y_eff)
            eff_w = min(tile_size, w - x_eff)

            mask_crop = mask[y_eff : y_eff + eff_h, x_eff : x_eff + eff_w]

            if mask_crop.mean() > threshold:
                coordinates.append({"x": int(x_eff), "y": int(y_eff)})

            # If we shifted back due to boundary, we might duplicate the last tile if we are not careful.
            # However, with range(0, w, step), the last point might trigger a shift.
            # A simple set of (x,y) could dedup, but let's stick to the grid.
            # The shift logic `min(x, w - tile_size)` ensures the last tile ends exactly at the edge.
            # If `x` was already such that `x + tile_size > w`, we shift.
            # We should break inner loops if we hit the boundary to avoid duplicates if step is small.
            # Simplified approach: Just standard sliding window, ignore partials or pad?
            # The prompt implies "ROI-constrained", usually we want fixed size inputs for CNNs.
            # The shift-back strategy is robust for fixed input size.

    # Remove duplicates that might occur due to boundary shifting
    unique_coords = [dict(t) for t in {tuple(d.items()) for d in coordinates}]
    return unique_coords


def prepare_data(
    metadata_df, tile_size, overlap, cache_dir, load_cached_data=True, split="train"
):
    """
    Main function to prepare tiled dataset dataframe.

    Args:
        metadata_df (pd.DataFrame): Metadata containing image paths.
        tile_size (int): Tile dimension.
        overlap (int): Overlap amount.
        cache_dir (str): Directory for caching.
        load_cached_data (bool): Use cache.
        split (str): 'train', 'val', or 'test' for naming cache files.

    Returns:
        pd.DataFrame: DataFrame containing tile coordinates and image IDs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tiles_{split}_{tile_size}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cached tiles parquet: {e}. Recomputing.")

    tile_data = []

    print(f"Preparing tiles for {split} set...")

    for _, row in metadata_df.iterrows():
        img_id = row["id"]

        # Construct full paths
        # Metadata paths are relative to input root (e.g. "train/id.tiff")
        img_path = os.path.join(CFG.input_root, row["image_path"])
        anat_path = os.path.join(CFG.input_root, row["anatomical_json_path"])

        if not os.path.exists(img_path):
            continue

        # Get image dimensions without loading data
        with rasterio.open(img_path) as src:
            h, w = src.height, src.width

        # Get anatomical mask
        mask = get_anatomical_mask(
            img_id, anat_path, (h, w), cache_dir, load_cached_data
        )

        # Get tiles
        # For test set, we might want a lower threshold to ensure we don't miss anything,
        # but the strategy says "strictly retain... using anatomical structure".
        coords = get_tile_coordinates(mask, tile_size, overlap, threshold=0.05)

        for coord in coords:
            tile_data.append(
                {
                    "id": img_id,
                    "image_path": row["image_path"],
                    "json_path": row.get("json_path", None),  # Might be NaN for test
                    "x": coord["x"],
                    "y": coord["y"],
                    "w": tile_size,
                    "h": tile_size,
                }
            )

    df_tiles = pd.DataFrame(tile_data)

    # Save cache
    if not df_tiles.empty:
        df_tiles.to_parquet(cache_path, index=False)

    return df_tiles
