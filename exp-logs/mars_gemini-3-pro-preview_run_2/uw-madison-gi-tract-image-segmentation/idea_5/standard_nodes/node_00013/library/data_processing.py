import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import rle_decode


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)


def load_and_preprocess_image(file_path, target_size=None):
    """
    Loads an image from the input directory, normalizes it to [0, 1],
    and optionally resizes it.

    Args:
        file_path (str): Relative path to the image file.
        target_size (tuple, optional): (Height, Width) to resize the image to.

    Returns:
        np.ndarray: Preprocessed image (float32).
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    # Load image (handling 16-bit depth if present)
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        # In case of missing files or read errors
        raise FileNotFoundError(f"Failed to load image at {full_path}")

    # Convert to float32 for processing
    img = img.astype(np.float32)

    # Min-Max Normalization to [0, 1]
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    # Resize if required
    if target_size is not None:
        # cv2.resize expects (width, height)
        img = cv2.resize(
            img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
        )

    return img


def generate_search_vector(img, search_size):
    """
    Generates a flattened vector for similarity search by downsampling the image.

    Args:
        img (np.ndarray): Normalized image.
        search_size (tuple): (Height, Width) for the search vector representation.

    Returns:
        np.ndarray: Flattened vector.
    """
    # Resize to small dimensions for fast retrieval
    # cv2.resize expects (width, height)
    img_small = cv2.resize(
        img, (search_size[1], search_size[0]), interpolation=cv2.INTER_AREA
    )
    return img_small.flatten()


def prepare_atlas_data(load_cached_data=True, debug=False):
    """
    Prepares the Atlas Bank for the retrieval system.

    Processing steps:
    1. Loads training metadata.
    2. Calculates relative depth for each slice (slice_idx / total_slices_in_case).
    3. Generates search vectors for all training images.
    4. Decodes and resizes ground truth masks to the target configuration.
    5. Caches the resulting index, vectors, and masks to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from ./working/idea_5/.
        debug (bool): If True, processes only a small subset of data.

    Returns:
        tuple: (index_df, vectors, masks)
            - index_df (pd.DataFrame): Metadata for the atlas.
            - vectors (np.ndarray): Search vectors (N, D).
            - masks (np.ndarray): Binary masks (N, H, W, C).
    """
    set_seed()

    # Define cache paths
    index_path = Config.ATLAS_INDEX_PATH
    vectors_path = Config.ATLAS_VECTORS_PATH
    masks_path = Config.ATLAS_MASKS_PATH

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(index_path)
            and os.path.exists(vectors_path)
            and os.path.exists(masks_path)
        ):
            print("Loading Atlas data from cache...")
            index_df = pd.read_parquet(index_path)
            vectors = np.load(vectors_path)
            masks = np.load(masks_path)
            return index_df, vectors, masks
        else:
            print("Cache missing or incomplete. Processing data from scratch...")

    # 2. Load and Process Metadata
    print("Loading training metadata...")
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    if debug:
        # Select first 2 cases for debugging
        debug_cases = df["case"].unique()[:2]
        df = df[df["case"].isin(debug_cases)].copy()
        print(f"Debug mode enabled. Processing {len(df)} rows.")

    # Calculate Relative Depth
    # Find max slice number for each (case, day) group
    max_slices = df.groupby(["case", "day"])["slice"].max().reset_index()
    max_slices.rename(columns={"slice": "max_slice"}, inplace=True)

    df = pd.merge(df, max_slices, on=["case", "day"], how="left")
    df["relative_depth"] = df["slice"] / df["max_slice"]

    # Pivot Data: Convert from Long format (one row per class) to Wide (one row per image)
    # We need columns for each class segmentation
    pivot_df = df.pivot(
        index="id", columns="class", values="segmentation"
    ).reset_index()

    # Get image attributes (drop duplicates since they are repeated for each class in original df)
    # We keep: id, file_path, case, day, slice, relative_depth, img_width, img_height
    attr_cols = [
        "id",
        "file_path",
        "case",
        "day",
        "slice",
        "relative_depth",
        "img_width",
        "img_height",
    ]
    attributes = df[attr_cols].drop_duplicates()

    # Merge pivoted masks with attributes
    atlas_df = pd.merge(attributes, pivot_df, on="id", how="inner")

    # Sort by case, day, slice for organized indexing
    atlas_df = atlas_df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # 3. Initialize Arrays
    num_samples = len(atlas_df)
    search_dim = Config.SEARCH_SIZE[0] * Config.SEARCH_SIZE[1]

    # Vectors: (N, flattened_dim)
    vectors = np.zeros((num_samples, search_dim), dtype=np.float32)

    # Masks: (N, H, W, Num_Classes)
    masks = np.zeros(
        (num_samples, Config.IMG_SIZE[0], Config.IMG_SIZE[1], len(Config.CLASSES)),
        dtype=np.uint8,
    )

    print(f"Processing {num_samples} images for Atlas...")

    # 4. Iterate and Process
    for i, row in atlas_df.iterrows():
        # -- Image Processing --
        # Load image (normalized)
        img = load_and_preprocess_image(row["file_path"])

        # Generate search vector
        vectors[i] = generate_search_vector(img, Config.SEARCH_SIZE)

        # -- Mask Processing --
        original_h = row["img_height"]
        original_w = row["img_width"]

        for c_idx, class_name in enumerate(Config.CLASSES):
            rle = row[class_name] if class_name in row else None

            if pd.isna(rle) or rle == "":
                continue  # Mask is already zeros

            # Decode RLE to original size
            mask_2d = rle_decode(rle, (original_h, original_w))

            # Resize to standardized Atlas size
            # Use Nearest Neighbor to preserve binary values
            mask_resized = cv2.resize(
                mask_2d,
                (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            masks[i, :, :, c_idx] = mask_resized

    # 5. Save to Cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Remove RLE columns from dataframe to save space (we have binary masks now)
    cols_to_drop = [c for c in Config.CLASSES if c in atlas_df.columns]
    index_df_save = atlas_df.drop(columns=cols_to_drop)

    index_df_save.to_parquet(index_path, index=False)
    np.save(vectors_path, vectors)
    np.save(masks_path, masks)

    print("Atlas data preparation complete.")
    return index_df_save, vectors, masks
