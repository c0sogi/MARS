import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import get_class_map


def prepare_classifier_data(split="train", load_cached=True):
    """
    Pre-processes the dataset for the classifier (Stage 2).
    Extracts character crops from the full pages, saves them to disk,
    and generates a cached .npy file compatible with KuzushijiCropDataset.

    This optimization significantly speeds up training by removing the need
    to load and crop large page images on the fly.

    Args:
        split (str): 'train' or 'val'.
        load_cached (bool): If True, attempts to load the pre-computed .npy file.

    Returns:
        list: A list of dictionaries, each containing metadata for a crop.
    """
    # Determine paths based on split
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_CLASSIFIER_TRAIN
        crop_dir = os.path.join(Config.WORKING_DIR, "crops", "train")
    else:
        metadata_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_CLASSIFIER_VAL
        crop_dir = os.path.join(Config.WORKING_DIR, "crops", "val")

    # 1. Try loading from cache
    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached classifier data for {split} from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True).tolist()
            # Verify that the first file exists to ensure cache validity
            if data and len(data) > 0 and os.path.exists(data[0]["image_path"]):
                return data
            else:
                print("Cached files not found on disk. Regenerating...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Regenerate Data
    print(f"Generating classifier crops for {split}...")

    # Ensure crop directory exists
    os.makedirs(crop_dir, exist_ok=True)

    # Load Class Map
    # We force load_cached=True for the class map to ensure consistency
    char_to_idx, _ = get_class_map(load_cached=True)

    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path, keep_default_na=False)

    data = []
    crop_size = Config.CLASSIFIER_IMG_SIZE

    # Iterate over all pages
    processed_count = 0
    skipped_count = 0

    for _, row in df.iterrows():
        image_id = row["image_id"]
        rel_path = row["file_path"]
        labels_str = row["labels"]

        if not labels_str:
            continue

        full_img_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load the full page image
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"Warning: Could not load image {full_img_path}")
            continue

        h_img, w_img = img.shape[:2]

        parts = labels_str.split()
        num_chars = len(parts) // 5

        for i in range(num_chars):
            code = parts[i * 5]

            # Skip if class not in map
            if code not in char_to_idx:
                continue

            try:
                x = int(parts[i * 5 + 1])
                y = int(parts[i * 5 + 2])
                w = int(parts[i * 5 + 3])
                h = int(parts[i * 5 + 4])

                # Handle boundaries (clamp to image dimensions)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)

                # Skip invalid crops (zero area)
                if x2 <= x1 or y2 <= y1:
                    skipped_count += 1
                    continue

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    skipped_count += 1
                    continue

                # Resize to target size (64x64)
                # We resize here to save disk space and loading time during training
                crop_resized = cv2.resize(
                    crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
                )

                # Save crop to disk
                # Filename: {image_id}_{index}_{class_idx}.jpg
                label_idx = char_to_idx[code]
                crop_filename = f"{image_id}_{i}_{label_idx}.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)

                cv2.imwrite(crop_path, crop_resized)

                # Add to data list
                # Note: We set bbox to [0, 0, crop_size, crop_size] because the image loaded
                # by the dataset will be the cropped image itself.
                data.append(
                    {
                        "image_path": crop_path,
                        "bbox": [0, 0, crop_size, crop_size],
                        "label_idx": label_idx,
                    }
                )

                processed_count += 1

            except ValueError:
                skipped_count += 1
                continue

    print(
        f"Processed {processed_count} crops. Skipped {skipped_count} invalid entries."
    )

    # Save the list to .npy cache
    print(f"Saving processed crop metadata to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, np.array(data, dtype=object))

    return data
