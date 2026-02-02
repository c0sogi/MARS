import os
import pandas as pd
import numpy as np
import cv2
import json
import random
import sys

# --- Configuration ---
SEED = 42
METADATA_PATH = "./metadata/train.csv"
INPUT_ROOT = "./input"
LABEL_MAP_PATH = "./input/label_num_to_disease_map.json"
PIXEL_SAMPLE_SIZE = 1000  # Number of images to sample for pixel statistics


# --- Seeding ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- Helper Functions ---
def load_data():
    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(METADATA_PATH)

    # Resolve full paths
    # The metadata contains relative paths from input root (e.g., "train_images/img.jpg")
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_ROOT, x))

    # Load Label Map
    label_map = {}
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
            # Ensure keys are integers for mapping
            label_map = {int(k): v for k, v in label_map.items()}

    return df, label_map


def analyze_target(df, label_map):
    print("TARGET VARIABLE ANALYSIS")

    # Distribution
    counts = df["label"].value_counts().sort_index()
    total = len(df)

    print("Distribution:")
    for label, count in counts.items():
        name = label_map.get(label, f"Class {label}")
        ratio = count / total
        print(f"  ID {label} ({name}): {count} samples ({ratio:.4f})")

    # Imbalance
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"Class Balance Ratio (Max/Min): {imbalance_ratio:.4f}")

    if imbalance_ratio > 5:
        print("  Note: Dataset is heavily imbalanced.")
    elif imbalance_ratio > 1.5:
        print("  Note: Dataset is moderately imbalanced.")
    else:
        print("  Note: Dataset is relatively balanced.")


def analyze_images(df):
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []
    channels_map = {}  # count -> frequency

    # Pixel stats accumulators (R, G, B)
    # We will normalize to 0-1 for calculation
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_n_pixels = 0

    # Select sample for pixel stats to save time
    sample_indices = set(
        np.random.choice(df.index, min(len(df), PIXEL_SAMPLE_SIZE), replace=False)
    )

    # List to store meta-features for relationship analysis
    meta_features = []

    # Iterate through all images for metadata, sample for pixels
    # Using a loop is safer than bulk loading for memory
    for idx, row in df.iterrows():
        fpath = row["full_path"]

        if not os.path.exists(fpath):
            continue

        # File Size
        fsize = os.path.getsize(fpath)
        file_sizes.append(fsize)

        # Read Image
        # IMREAD_UNCHANGED to detect Alpha or Grayscale correctly
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # Dimensions
        if len(img.shape) == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)

        channels_map[c] = channels_map.get(c, 0) + 1

        # Pixel Stats (Sampled)
        if idx in sample_indices:
            # Normalize to 0-1
            img_norm = img.astype(np.float64) / 255.0

            # Handle channels for stats (assume RGB target)
            if c == 1:
                # Grayscale to RGB
                img_norm = np.stack([img_norm] * 3, axis=-1)
            elif c == 4:
                # Drop Alpha
                img_norm = img_norm[:, :, :3]
            elif c == 3:
                # BGR to RGB
                img_norm = img_norm[:, :, ::-1]

            # Reshape to (N, 3)
            pixels = img_norm.reshape(-1, 3)

            pixel_sum += pixels.sum(axis=0)
            pixel_sq_sum += (pixels**2).sum(axis=0)
            pixel_n_pixels += pixels.shape[0]

        # Store for later
        meta_features.append(
            {
                "label": row["label"],
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
                "file_size": fsize,
            }
        )

    # Report Dimensions
    w_arr = np.array(widths)
    h_arr = np.array(heights)
    ar_arr = np.array(aspect_ratios)

    print("Dimensions:")
    print(
        f"  Width:  Mean={w_arr.mean():.4f}, Std={w_arr.std():.4f}, Min={w_arr.min()}, Max={w_arr.max()}"
    )
    print(
        f"  Height: Mean={h_arr.mean():.4f}, Std={h_arr.std():.4f}, Min={h_arr.min()}, Max={h_arr.max()}"
    )
    print(f"  Aspect Ratio: Mean={ar_arr.mean():.4f}, Std={ar_arr.std():.4f}")

    # Report Channels
    print("Channels:")
    for c, count in channels_map.items():
        print(f"  {c} Channels: {count} images")

    # Report Pixel Stats
    if pixel_n_pixels > 0:
        rgb_mean = pixel_sum / pixel_n_pixels
        # Var = E[x^2] - (E[x])^2
        rgb_var = (pixel_sq_sum / pixel_n_pixels) - (rgb_mean**2)
        rgb_std = np.sqrt(np.maximum(rgb_var, 0))  # clip negative due to precision

        print("Pixel Stats (RGB, Normalized 0-1):")
        print(f"  Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
        print(f"  Std:  R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")

    return pd.DataFrame(meta_features)


def analyze_relationships(meta_df, label_map):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Group by label to see if image properties correlate with class
    print("Unstructured Relationships (Meta-features vs Target):")

    grouped = meta_df.groupby("label")[
        ["width", "height", "aspect_ratio", "file_size"]
    ].mean()

    # Add count to the group display
    grouped["count"] = meta_df.groupby("label")["file_size"].count()

    # Rename index
    grouped.index = grouped.index.map(lambda x: f"{x} ({label_map.get(x, '')[:10]}..)")

    # Print formatted
    print(grouped.to_string(float_format=lambda x: f"{x:.4f}"))

    # Check for correlation between file size and aspect ratio (redundancy check)
    corr = meta_df[["width", "height", "aspect_ratio", "file_size"]].corr()
    print("\nMeta-feature Correlation Matrix:")
    print(corr.to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    set_seed(SEED)

    # 1. Load
    df, label_map = load_data()

    # 2. Target Analysis
    analyze_target(df, label_map)

    # 3. Image Analysis
    meta_df = analyze_images(df)

    # 4. Relationships
    analyze_relationships(meta_df, label_map)


if __name__ == "__main__":
    main()
