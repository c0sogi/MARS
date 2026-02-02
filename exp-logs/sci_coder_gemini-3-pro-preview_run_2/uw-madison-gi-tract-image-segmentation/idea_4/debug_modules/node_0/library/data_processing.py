import os
import numpy as np
import pandas as pd
import cv2
from skimage.segmentation import slic
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor
import functools

from library import config, utils


def get_superpixels(img):
    """
    Generates superpixels for a given image using SLIC.

    Args:
        img (np.ndarray): Input image (normalized float32).

    Returns:
        np.ndarray: Integer mask of superpixel segments.
    """
    # SLIC parameters from config
    # channel_axis=None indicates the input is a 2D grayscale image
    segments = slic(
        img,
        n_segments=config.N_SEGMENTS,
        compactness=config.COMPACTNESS,
        sigma=config.SIGMA,
        start_label=1,
        channel_axis=None,
    )
    return segments


def extract_features_single(img_curr, img_prev, img_next, segments):
    """
    Extracts features for a single slice given its context and superpixel segments.

    Args:
        img_curr (np.ndarray): Current slice image.
        img_prev (np.ndarray): Previous slice image (context).
        img_next (np.ndarray): Next slice image (context).
        segments (np.ndarray): Superpixel integer mask.

    Returns:
        tuple: (pd.DataFrame features, np.ndarray unique_labels)
    """
    unique_labels = np.unique(segments)
    if len(unique_labels) == 0:
        return pd.DataFrame(), unique_labels

    # 1. Intensity Features (Current Slice)
    # Using ndimage for fast calculation of stats over regions
    mean_curr = ndimage.mean(img_curr, labels=segments, index=unique_labels)
    var_curr = ndimage.variance(img_curr, labels=segments, index=unique_labels)

    # 2. Context Features (2.5D)
    mean_prev = ndimage.mean(img_prev, labels=segments, index=unique_labels)
    mean_next = ndimage.mean(img_next, labels=segments, index=unique_labels)

    # 3. Spatial Features (Centroids)
    # center_of_mass returns list of (y, x) coordinates
    # We pass a dummy array of ones to get the geometric center of the segment shape
    coords = ndimage.center_of_mass(
        np.ones_like(segments), labels=segments, index=unique_labels
    )
    coords = np.array(coords)

    # Normalize coordinates by image dimensions
    h, w = img_curr.shape
    cent_y = coords[:, 0] / h
    cent_x = coords[:, 1] / w

    features = pd.DataFrame(
        {
            "mean": mean_curr,
            "std": np.sqrt(var_curr),
            "mean_prev": mean_prev,
            "mean_next": mean_next,
            "cent_y": cent_y,
            "cent_x": cent_x,
        }
    )

    return features, unique_labels


def _process_group(group_data, split):
    """
    Worker function to process a single Case/Day volume.
    Loads images, generates superpixels, and creates training samples.
    """
    (case, day), group_df = group_data

    # Sort slices to ensure correct volumetric context (i-1, i, i+1)
    group_df = group_df.sort_values("slice")
    slices = group_df["slice"].values

    volume_imgs = {}
    volume_masks = {}

    # --- Data Loading Phase ---
    for s in slices:
        # Get metadata for this slice
        # Note: metadata is long-format (one row per class), but file info is duplicated.
        # We take the first row for file info.
        row = group_df[group_df.slice == s].iloc[0]

        try:
            img = utils.load_image(row.file_path)
            volume_imgs[s] = img
        except Exception:
            continue

        # Load Masks (Only for training/validation)
        if split != "test":
            masks = {}
            for cls in config.CLASSES[1:]:  # Skip 'background'
                cls_row = group_df[(group_df.slice == s) & (group_df["class"] == cls)]
                if not cls_row.empty:
                    rle = cls_row.iloc[0]["segmentation"]
                    # Decode at original resolution
                    m = utils.rle_decode(rle, (row.img_height, row.img_width))
                    # Resize to target resolution using Nearest Neighbor to preserve binary nature
                    if (row.img_height, row.img_width) != config.IMG_SIZE:
                        m = cv2.resize(
                            m,
                            (config.IMG_SIZE[1], config.IMG_SIZE[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    masks[cls] = m
                else:
                    masks[cls] = np.zeros(config.IMG_SIZE, dtype=np.uint8)
            volume_masks[s] = masks

    # --- Processing Phase ---
    results = []

    for s in slices:
        if s not in volume_imgs:
            continue

        img_curr = volume_imgs[s]
        # Handle boundary conditions by replicating current slice
        img_prev = volume_imgs.get(s - 1, img_curr)
        img_next = volume_imgs.get(s + 1, img_curr)

        # 1. Generate Superpixels
        segments = get_superpixels(img_curr)

        # 2. Extract Features
        feats, u_labels = extract_features_single(
            img_curr, img_prev, img_next, segments
        )

        if feats.empty:
            continue

        # 3. Assign Labels (Training Only)
        if split != "test":
            masks = volume_masks[s]

            # Combine masks into a single map: 0=bg, 1=large, 2=small, 3=stomach
            combined_mask = np.zeros(config.IMG_SIZE, dtype=np.uint8)
            combined_mask[masks["large_bowel"] > 0] = 1
            combined_mask[masks["small_bowel"] > 0] = 2
            combined_mask[masks["stomach"] > 0] = 3

            # Determine label for each superpixel via Majority Vote
            # We count pixels of each class within each superpixel
            counts = []
            for cls_idx in range(len(config.CLASSES)):
                # Create binary mask for class cls_idx
                cls_mask = (combined_mask == cls_idx).astype(float)
                # Sum pixels of this class in each segment
                cls_count = ndimage.sum(cls_mask, labels=segments, index=u_labels)
                counts.append(cls_count)

            counts = np.stack(counts, axis=1)  # Shape: (N_segments, 4)
            labels = np.argmax(counts, axis=1)

            feats["label"] = labels

            # Add metadata for debugging or stratified splitting
            feats["case"] = case
            feats["day"] = day
            feats["slice"] = s

        results.append(feats)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def build_tabular_dataset(split="train", load_cached_data=True):
    """
    Constructs the tabular dataset for LightGBM.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Tabular dataset with features and labels.
    """
    cache_file = os.path.join(config.WORKING_DIR, f"tabular_{split}.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} dataset from {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Generating {split} dataset from scratch... This may take time.")

    # 2. Load Metadata
    meta = utils.load_metadata(split)

    # 3. Parallel Processing
    # Group by Case/Day to process volumes independently
    groups = list(meta.groupby(["case", "day"]))

    results = []
    # Use ProcessPoolExecutor to leverage multiple CPUs
    with ProcessPoolExecutor(max_workers=10) as executor:
        worker = functools.partial(_process_group, split=split)
        for res in executor.map(worker, groups):
            if not res.empty:
                results.append(res)

    if not results:
        raise ValueError(f"No data generated for split {split}")

    full_df = pd.concat(results, ignore_index=True)

    # 4. Balancing (Training Only)
    if split == "train":
        print("Balancing training dataset...")
        bg_mask = full_df["label"] == 0
        df_bg = full_df[bg_mask]
        df_fg = full_df[~bg_mask]

        # Downsample background class
        df_bg_sampled = df_bg.sample(
            frac=config.BACKGROUND_SAMPLE_RATE, random_state=config.SEED
        )

        # Combine and shuffle
        full_df = pd.concat([df_bg_sampled, df_fg], ignore_index=True)
        full_df = full_df.sample(frac=1.0, random_state=config.SEED).reset_index(
            drop=True
        )

    # 5. Save to Cache
    print(f"Saving {split} dataset to {cache_file}")
    full_df.to_parquet(cache_file)

    return full_df
