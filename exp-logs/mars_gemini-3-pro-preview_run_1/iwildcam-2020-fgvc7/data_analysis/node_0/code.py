import os
import json
import random
import numpy as np
import pandas as pd
import cv2
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
MEGADETECTOR_PATH = os.path.join(INPUT_DIR, "iwildcam2020_megadetector_results.json")
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_eda():
    set_seed(SEED)

    # --- 1. Load Data ---
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # --- 2. Target Variable Analysis ---
    print("TARGET VARIABLE ANALYSIS")
    target_col = "category_id"

    # Distribution
    class_counts = df[target_col].value_counts()
    n_classes = len(class_counts)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {n_classes}")

    # Imbalance
    max_class = class_counts.idxmax()
    max_count = class_counts.max()
    min_class = class_counts.idxmin()
    min_count = class_counts.min()

    print(
        f"Most Frequent Class: ID {max_class} (Count: {max_count}, {max_count/total_samples*100:.4f}%)"
    )
    print(
        f"Least Frequent Class: ID {min_class} (Count: {min_count}, {min_count/total_samples*100:.4f}%)"
    )
    print(f"Class Imbalance Ratio (Max/Min): {max_count/min_count:.4f}")

    # Specific check for Empty category (0)
    empty_count = class_counts.get(0, 0)
    print(
        f"Empty Images (Category 0): {empty_count} ({empty_count/total_samples*100:.4f}%)"
    )
    print("-" * 30)

    # --- 3. Input Data Analysis (Image) ---
    print("INPUT DATA ANALYSIS (IMAGE)")

    # Sampling for Image Analysis to keep within time limits
    # We use a stratified sample if possible, otherwise random
    sample_size_dims = 2000
    sample_size_pixels = 500

    if len(df) > sample_size_dims:
        sample_df = df.sample(n=sample_size_dims, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Subset for pixel stats
    pixel_sample_paths = set(
        sample_df.sample(n=min(len(sample_df), sample_size_pixels), random_state=SEED)[
            "file_path"
        ].values
    )

    print(
        f"Analyzing {len(sample_df)} images for dimensions and {len(pixel_sample_paths)} for pixel stats..."
    )

    for idx, row in sample_df.iterrows():
        # Construct full path
        # metadata file_path is like "train/..." or "test/..."
        # input dir is "./input"
        # so full path is "./input/train/..."
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(img_path):
            continue

        try:
            # Read image
            # cv2.imread returns BGR
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channels_list.append(c)

            # Calculate pixel stats for the smaller subset
            if row["file_path"] in pixel_sample_paths:
                # Convert to RGB for reporting
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize to 0-1 for calculation stability, then scale back or report as is
                # Here we report 0-255 stats
                img_data = img_rgb.astype(np.float64)

                pixel_sum += np.sum(img_data, axis=(0, 1))
                pixel_sq_sum += np.sum(img_data**2, axis=(0, 1))
                pixel_count += w * h

        except Exception as e:
            continue

    # Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(
        f"Widths: Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Heights: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels
    n_channels_counts = Counter(channels_list)
    print(f"Channel Counts: {dict(n_channels_counts)}")

    # Pixel Stats
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))
        print(
            f"Pixel Mean (RGB): R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}"
        )
        print(
            f"Pixel Std (RGB): R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}"
        )
    else:
        print("Pixel Stats: N/A (No images processed)")
    print("-" * 30)

    # --- 4. Feature/Signal Relationships ---
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 4.1 Meta-Feature: Image Size vs Target
    # We use the sampled dataframe which now has dimensions (we need to map them back)
    # Since we iterated, let's just create a small dataframe from the lists
    # Note: The lists align with the iterations.

    # We need to be careful about skipped images.
    # Re-collecting data into a structured format for correlation

    # Let's do a quick pass of the MegaDetector data first, as that's a strong signal
    print("Loading MegaDetector Results...")
    try:
        with open(MEGADETECTOR_PATH, "r") as f:
            md_data = json.load(f)

        # Create a map: image_id -> max_conf, num_detections
        md_map = {}
        for img_entry in md_data.get("images", []):
            img_id = img_entry["id"]
            detections = img_entry.get("detections", [])
            # Filter detections by confidence threshold (e.g., 0.5) to count "valid" animals
            valid_detections = [d for d in detections if d["conf"] > 0.5]
            max_conf = img_entry.get("max_detection_conf", 0.0)
            md_map[img_id] = {
                "num_detections": len(valid_detections),
                "max_conf": max_conf,
            }

        # Add MD features to our main sample_df
        # sample_df has 'image_id'
        sample_df["md_num_detections"] = sample_df["image_id"].map(
            lambda x: md_map.get(x, {}).get("num_detections", 0)
        )
        sample_df["md_max_conf"] = sample_df["image_id"].map(
            lambda x: md_map.get(x, {}).get("max_conf", 0.0)
        )

        # Analyze relationship between Category 0 (Empty) and MD Confidence
        empty_mask = sample_df["category_id"] == 0
        animal_mask = sample_df["category_id"] != 0

        mean_conf_empty = sample_df[empty_mask]["md_max_conf"].mean()
        mean_conf_animal = sample_df[animal_mask]["md_max_conf"].mean()

        print(f"Mean Max Confidence for Empty Images (Cat 0): {mean_conf_empty:.4f}")
        print(
            f"Mean Max Confidence for Animal Images (Cat != 0): {mean_conf_animal:.4f}"
        )

        mean_det_empty = sample_df[empty_mask]["md_num_detections"].mean()
        mean_det_animal = sample_df[animal_mask]["md_num_detections"].mean()

        print(f"Mean Num Detections (>0.5 conf) for Empty Images: {mean_det_empty:.4f}")
        print(
            f"Mean Num Detections (>0.5 conf) for Animal Images: {mean_det_animal:.4f}"
        )

        # Correlation
        # Create binary target: 0 if empty, 1 if animal
        sample_df["is_animal"] = (sample_df["category_id"] != 0).astype(int)
        corr_conf = sample_df["md_max_conf"].corr(sample_df["is_animal"])
        print(f"Correlation (Max Conf vs Is_Animal): {corr_conf:.4f}")

    except Exception as e:
        print(f"Could not analyze MegaDetector results: {e}")

    print("-" * 30)


if __name__ == "__main__":
    run_eda()
