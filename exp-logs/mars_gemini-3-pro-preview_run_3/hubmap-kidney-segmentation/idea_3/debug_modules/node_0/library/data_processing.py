import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import cv2
from sklearn.decomposition import TruncatedSVD
from library.utils import set_seed, get_logger

# Initialize Logger
logger = get_logger("DataProcessing")

# Default Reference Values for Macenko Normalization (derived from a standard H&E reference)
# Stain Matrix (He, Eos)
REF_HEREF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
# Max Concentrations
REF_MAXCREF = np.array([1.9705, 1.0308])


def read_tiff(image_path, window=None):
    """
    Reads a TIFF image or a specific window of it.

    Args:
        image_path (str): Path to the TIFF file.
        window (rasterio.windows.Window, optional): Window to read.

    Returns:
        np.ndarray: Image array (H, W, C) in RGB.
    """
    try:
        with rasterio.open(image_path) as src:
            if window:
                img = src.read(window=window)
            else:
                img = src.read()

            # Rasterio reads as (C, H, W), convert to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            return img
    except Exception as e:
        logger.error(f"Error reading TIFF {image_path}: {e}")
        return None


def rasterize_json_polygons(json_path, shape, filter_name=None):
    """
    Converts JSON polygon annotations into a binary mask.

    Args:
        json_path (str): Path to the JSON file.
        shape (tuple): Target shape (Height, Width).
        filter_name (str, optional): If provided, only rasterize objects with this classification name (e.g., 'Cortex').

    Returns:
        np.ndarray: Binary mask (uint8).
    """
    mask = np.zeros(shape, dtype=np.uint8)

    if not os.path.exists(json_path):
        logger.warning(f"JSON path not found: {json_path}. Returning empty mask.")
        return mask

    try:
        with open(json_path, "r") as f:
            annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {json_path}: {e}")
        return mask

    polygons = []
    for ann in annotations:
        # Check classification if filter provided
        if filter_name:
            props = ann.get("properties", {})
            classification = props.get("classification", {})
            name = classification.get("name", "")
            if name != filter_name:
                continue

        # Extract Geometry
        geom = ann.get("geometry", {})
        coords = geom.get("coordinates", [])

        # Coordinates can be nested differently depending on Polygon vs MultiPolygon
        # Standard HuBMAP format usually wraps coordinates in one extra list
        if len(coords) > 0:
            for poly_coords in coords:
                pts = np.array(poly_coords, dtype=np.int32)
                polygons.append(pts)

    if polygons:
        cv2.fillPoly(mask, polygons, 1)

    return mask


def macenko_normalize(
    img, alpha=1, beta=0.15, Io=240, target_HERef=None, target_maxCRef=None
):
    """
    Performs Macenko stain normalization on an RGB image.

    Args:
        img (np.ndarray): Input RGB image (H, W, 3).
        alpha (float): Percentile for normalization (default 1).
        beta (float): Transparency threshold (default 0.15).
        Io (int): Transmitted light intensity (default 240).
        target_HERef (np.ndarray): Reference stain matrix.
        target_maxCRef (np.ndarray): Reference max concentrations.

    Returns:
        np.ndarray: Normalized image.
    """
    if target_HERef is None:
        target_HERef = REF_HEREF
    if target_maxCRef is None:
        target_maxCRef = REF_MAXCREF

    h, w, c = img.shape
    img = img.reshape((-1, 3))

    # Calculate Optical Density
    OD = -np.log((img.astype(np.float32) + 1) / Io)

    # Remove data with too low OD (background)
    ODhat = OD[np.all(OD > beta, axis=1)]

    # If not enough tissue pixels, return original
    if ODhat.shape[0] < 100:
        return img.reshape((h, w, c))

    # Calculate eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    except np.linalg.LinAlgError:
        return img.reshape((h, w, c))

    # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
    # eigvecs returned in ascending order
    Tw = eigvecs[:, 1:3]
    projected = np.dot(ODhat, Tw)

    # Calculate angle of each point
    phi = np.arctan2(projected[:, 1], projected[:, 0])

    # Find min and max vectors and project back to OD space
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = np.dot(Tw, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Tw, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # Heuristic to ensure vector order (H&E)
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax]).T
    else:
        HE = np.array([vMax, vMin]).T

    # Rows correspond to channels (RGB), columns to stains (H, E)
    # Calculate concentrations
    # Y = C * HE^T -> C = Y * pinv(HE^T)
    Y = np.reshape(OD, (-1, 3))
    C = np.dot(Y, np.linalg.pinv(HE))

    # Normalize concentrations
    maxC = np.percentile(C, 99, axis=0)
    # Avoid division by zero
    maxC = np.maximum(maxC, 1e-5)

    C = C / maxC[None, :]
    C = C * target_maxCRef[None, :]

    # Reconstruct image
    # OD_norm = C * target_HERef^T
    Inorm = Io * np.exp(-np.dot(C, target_HERef.T))
    Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)

    return Inorm.reshape((h, w, 3))


def get_tile_coordinates(mask, tile_size, overlap, threshold=0.1):
    """
    Generates (x, y) coordinates for tiles that contain sufficient mask area.

    Args:
        mask (np.ndarray): Binary mask (H, W).
        tile_size (int): Size of the square tile.
        overlap (float): Overlap fraction (0.0 to 1.0).
        threshold (float): Minimum fraction of mask pixels required to keep tile.

    Returns:
        list: List of (x, y) top-left coordinates.
    """
    h, w = mask.shape
    step = int(tile_size * (1 - overlap))

    coordinates = []

    # If mask is empty, return empty list (or handle as needed)
    if mask.sum() == 0:
        return coordinates

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Adjust for boundary
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)

            # If we are at the edge and the tile is smaller than tile_size,
            # we can either skip or shift back. Shifting back ensures fixed size.
            if x + tile_size > w:
                x = max(0, w - tile_size)
                x_end = w
            if y + tile_size > h:
                y = max(0, h - tile_size)
                y_end = h

            # Extract mask patch
            mask_patch = mask[y:y_end, x:x_end]

            # Check tissue content
            if mask_patch.mean() >= threshold:
                coordinates.append((x, y))

            # If we shifted back for edge, we don't need to continue in this row/col loop for the very last bit
            if x == max(0, w - tile_size) and y == max(0, h - tile_size):
                break

    # Remove duplicates if shift-back caused any
    return sorted(list(set(coordinates)))


def process_dataset(
    metadata_path,
    output_dir,
    tile_size=1024,
    overlap=0.5,
    load_cached_data=True,
    tissue_threshold=0.05,
):
    """
    Main function to process the dataset and generate a tile index.

    Args:
        metadata_path (str): Path to metadata CSV.
        output_dir (str): Directory to save outputs (cache).
        tile_size (int): Tile size.
        overlap (float): Overlap ratio.
        load_cached_data (bool): Whether to use cached parquet/npy files.
        tissue_threshold (float): Minimum anatomical mask area to include a tile.

    Returns:
        pd.DataFrame: DataFrame containing tile information.
    """
    set_seed(42)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    mask_cache_dir = os.path.join(output_dir, "tissue_masks_cache")
    os.makedirs(mask_cache_dir, exist_ok=True)

    # Determine cache file name based on config
    meta_name = os.path.basename(metadata_path).replace(".csv", "")
    cache_file = os.path.join(output_dir, f"tiles_{meta_name}_{tile_size}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading cached tiles from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Processing dataset from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    tile_data = []

    # Input root directory (metadata paths are relative to input/)
    input_root = "./input"

    for idx, row in df.iterrows():
        image_id = row["id"]
        # Construct full paths
        # Metadata paths are relative to input root, e.g., "train/image.tiff"
        img_path = os.path.join(input_root, row["image_path"])
        anat_path = os.path.join(input_root, row["anatomical_json_path"])

        # Determine image shape
        # We can open the image to get shape, or use metadata if available.
        # Ideally open image to be safe.
        try:
            with rasterio.open(img_path) as src:
                h, w = src.height, src.width
        except Exception as e:
            logger.error(f"Skipping {image_id} due to read error: {e}")
            continue

        # Handle Anatomical Mask (Cortex)
        # Check cache
        mask_cache_path = os.path.join(mask_cache_dir, f"{image_id}_{w}x{h}_Cortex.npy")

        if load_cached_data and os.path.exists(mask_cache_path):
            cortex_mask = np.load(mask_cache_path)
        else:
            # Generate Cortex mask
            cortex_mask = rasterize_json_polygons(
                anat_path, (h, w), filter_name="Cortex"
            )
            # If Cortex is empty, maybe try to use the whole tissue or Medulla?
            # Prompt says "filtered by the anatomical 'Cortex' mask".
            # However, some images might be all Medulla or not labeled 'Cortex'.
            # Fallback: If cortex is empty, use Medulla or everything (assuming whole image is tissue).
            if cortex_mask.sum() == 0:
                logger.info(f"No Cortex found for {image_id}, checking Medulla...")
                medulla_mask = rasterize_json_polygons(
                    anat_path, (h, w), filter_name="Medulla"
                )
                if medulla_mask.sum() > 0:
                    cortex_mask = medulla_mask
                else:
                    # Fallback to a simple tissue detection or use whole image?
                    # For safety, let's assume whole image is valid if no annotation
                    logger.warning(
                        f"No anatomical structure found for {image_id}. Using whole image."
                    )
                    cortex_mask = np.ones((h, w), dtype=np.uint8)

            # Save to cache
            np.save(mask_cache_path, cortex_mask)

        # Get Tiles
        coords = get_tile_coordinates(
            cortex_mask, tile_size, overlap, threshold=tissue_threshold
        )

        for x, y in coords:
            tile_info = {
                "id": image_id,
                "image_path": row["image_path"],
                "json_path": row.get("json_path", ""),  # Might be empty for test
                "x": x,
                "y": y,
                "w": tile_size,
                "h": tile_size,
            }
            # Add other metadata columns if needed
            if "patient_number" in row:
                tile_info["patient_number"] = row["patient_number"]

            tile_data.append(tile_info)

        if (idx + 1) % 5 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} images.")

    # Create DataFrame
    tiles_df = pd.DataFrame(tile_data)

    # Save to cache
    if not tiles_df.empty:
        tiles_df.to_parquet(cache_file, index=False)
        logger.info(f"Saved {len(tiles_df)} tiles to {cache_file}")
    else:
        logger.warning("No tiles generated!")

    return tiles_df
