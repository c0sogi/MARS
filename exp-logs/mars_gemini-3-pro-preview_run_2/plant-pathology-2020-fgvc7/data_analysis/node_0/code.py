import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter

# Constants
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train_metadata.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    df = pd.read_csv(METADATA_FILE)
    # Construct full path
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))
    return df


def analyze_targets(df):
    print("TARGET VARIABLE ANALYSIS")

    # The metadata contains a 'stratify_label' which represents the active class
    if "stratify_label" not in df.columns:
        print("Error: 'stratify_label' column missing from metadata.")
        return

    target_counts = df["stratify_label"].value_counts()
    total_samples = len(df)

    print("Distribution:")
    for label, count in target_counts.items():
        print(f"  {label}: {count}")

    print("\nImbalance/Skew:")
    print("  Class Balance Ratios (Proportion):")
    for label, count in target_counts.items():
        ratio = count / total_samples
        print(f"    {label}: {ratio:.4f}")

    # Check for extreme imbalance (e.g., any class < 1%)
    min_ratio = target_counts.min() / total_samples
    if min_ratio < 0.01:
        print(f"\n  Alert: Found rare class with frequency {min_ratio:.4f} < 1%")
    else:
        print("\n  No extremely rare classes (< 1%) detected.")
    print("-" * 30)


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE)")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []

    # For pixel stats (Global Mean/Std)
    # We will compute running sum and squared sum
    # To save time on very large datasets, one might sample, but for ~1300 images we can do all.
    pixel_sum = np.zeros(3)  # BGR in OpenCV
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0

    # We will also store meta-features for relationship analysis later
    meta_features = []

    # Iterate through images
    # Using a sample if dataset was huge, but here N=1310 is small enough to process fully within 1 hour.
    # We process all to be robust.

    for idx, row in df.iterrows():
        path = row["full_path"]
        if not os.path.exists(path):
            continue

        # Get file size
        f_size = os.path.getsize(path)
        file_sizes.append(f_size)

        # Read image
        img = cv2.imread(path)
        if img is None:
            continue

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)

        # Pixel stats accumulation
        # Normalize to 0-1 for calculation to avoid overflow with squares, then scale back or keep as is?
        # Standard practice is often reporting in 0-255 or 0-1. Let's do 0-255 stats.

        # Flatten spatial dimensions
        pixels = img.reshape(-1, 3)
        n_pixels = pixels.shape[0]

        pixel_sum += pixels.sum(axis=0)
        pixel_sq_sum += (pixels**2).sum(axis=0)
        total_pixels += n_pixels

        # Store for relationship analysis
        # Calculate mean intensity for this image
        mean_intensity = img.mean()
        meta_features.append(
            {
                "image_id": row["image_id"],
                "width": w,
                "height": h,
                "aspect_ratio": w / h,
                "mean_intensity": mean_intensity,
                "file_size_bytes": f_size,
            }
        )

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    ars = np.array(aspect_ratios)

    print("Dimensions:")
    print(
        f"  Widths: Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"  Heights: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"  Aspect Ratios: Mean={ars.mean():.4f}, Std={ars.std():.4f}, Min={ars.min():.4f}, Max={ars.max():.4f}"
    )

    # Channels Analysis
    c_counts = Counter(channels)
    print("\nChannels:")
    for c, count in c_counts.items():
        print(f"  {c} channels: {count} images")

    # Pixel Stats
    if total_pixels > 0:
        # Global Mean
        global_mean = pixel_sum / total_pixels
        # Global Std = sqrt(E[x^2] - (E[x])^2)
        global_sq_mean = pixel_sq_sum / total_pixels
        global_std = np.sqrt(global_sq_mean - global_mean**2)

        # OpenCV is BGR, converting to RGB for report
        rgb_mean = global_mean[::-1]
        rgb_std = global_std[::-1]

        print("\nPixel Stats (RGB, 0-255 scale):")
        print(f"  Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
        print(f"  Std:  R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")

    print("-" * 30)
    return pd.DataFrame(meta_features)


def analyze_relationships(df, meta_df):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Merge metadata features with target labels
    # df has 'image_id' and 'stratify_label'
    merged = pd.merge(df[["image_id", "stratify_label"]], meta_df, on="image_id")

    print("Unstructured (Meta-Feature) Relationships:")
    print(
        "Relationship between Image Metadata and Target Class (Mean values per class):"
    )

    # Group by label and calculate means of meta features
    numeric_cols = [
        "width",
        "height",
        "aspect_ratio",
        "mean_intensity",
        "file_size_bytes",
    ]
    grouped = merged.groupby("stratify_label")[numeric_cols].mean()

    # Print formatted table-like structure
    header = f"{'Class':<20} | {'Width':<10} | {'Height':<10} | {'AR':<10} | {'Intensity':<10} | {'FileSize':<10}"
    print(header)
    print("-" * len(header))

    for label, row in grouped.iterrows():
        print(
            f"{label:<20} | {row['width']:<10.4f} | {row['height']:<10.4f} | {row['aspect_ratio']:<10.4f} | {row['mean_intensity']:<10.4f} | {row['file_size_bytes']:<10.4f}"
        )

    print("\nObservation:")
    # Simple heuristic check
    ar_std = grouped["aspect_ratio"].std()
    if ar_std < 0.05:
        print("  Aspect ratios are consistent across classes.")
    else:
        print("  Aspect ratios vary noticeably between classes.")

    print("-" * 30)


def main():
    set_seed(SEED)

    try:
        df = load_data()

        # 1. Target Analysis
        analyze_targets(df)

        # 2. Image Analysis
        # This returns a dataframe of extracted meta features for step 3
        meta_features_df = analyze_images(df)

        # 3. Relationships
        if not meta_features_df.empty:
            analyze_relationships(df, meta_features_df)

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
