import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter

# ==========================================
# Configuration & Setup
# ==========================================
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)


def main():
    # ==========================================
    # 1. Data Loading
    # ==========================================
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Identify target column (using stratify_label generated in metadata for single-label analysis)
    # The original targets are one-hot/probabilistic, but stratify_label gives the dominant class.
    target_col = "stratify_label"

    if target_col not in df.columns:
        # Fallback if stratify_label is missing, though it should be there per description
        print(
            "Warning: 'stratify_label' not found. Attempting to reconstruct from columns."
        )
        possible_targets = ["healthy", "multiple_diseases", "rust", "scab"]
        existing_targets = [c for c in possible_targets if c in df.columns]
        if existing_targets:
            df[target_col] = df[existing_targets].idxmax(axis=1)
        else:
            print("Error: No target labels found.")
            return

    # Distribution
    class_counts = df[target_col].value_counts()
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {len(class_counts)}")
    print("\nClass Distribution:")
    for label, count in class_counts.items():
        ratio = count / total_samples
        print(f"  {label:<20}: {count:4d} ({ratio:.4f})")

    # Imbalance Check
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class if min_class > 0 else 0
    print(f"\nClass Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")
    if imbalance_ratio > 5:
        print("  -> Significant class imbalance detected.")
    elif imbalance_ratio > 2:
        print("  -> Moderate class imbalance detected.")
    else:
        print("  -> Classes are relatively balanced.")

    # ==========================================
    # 3. Input Data Analysis (Image Modality)
    # ==========================================
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # We will iterate through images to collect stats
    # Stats to collect: Width, Height, Aspect Ratio, Channel Counts, Pixel Sums (for mean/std)

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = Counter()

    # For global pixel stats (WELFORD's algorithm or simple accumulation)
    # Simple accumulation is fine for 1300 images.
    # We'll use running sum for mean, and running sum of squares for std.
    # Normalizing to [0, 1] for calculation.

    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    total_pixels = 0

    # To analyze meta-feature relationships later, we store per-image stats
    meta_features = []

    # Iterate
    # Note: paths in metadata are relative to input folder (e.g., "images/Train_0.jpg")
    valid_images_count = 0

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # Read image
        # cv2 reads in BGR format
        img = cv2.imread(full_path)
        if img is None:
            continue

        h, w, c = img.shape
        valid_images_count += 1

        widths.append(w)
        heights.append(h)
        ar = w / h
        aspect_ratios.append(ar)
        channel_counts[c] += 1

        # Convert to RGB for reporting standard stats
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # Accumulate for global stats
        # Sum over H and W
        ch_sums = img_rgb.sum(axis=(0, 1))  # [R_sum, G_sum, B_sum]
        ch_sq_sums = (img_rgb**2).sum(axis=(0, 1))

        sum_r += ch_sums[0]
        sum_g += ch_sums[1]
        sum_b += ch_sums[2]

        sum_sq_r += ch_sq_sums[0]
        sum_sq_g += ch_sq_sums[1]
        sum_sq_b += ch_sq_sums[2]

        total_pixels += h * w

        # Store for relationship analysis
        meta_features.append(
            {
                "target": row[target_col],
                "width": w,
                "height": h,
                "aspect_ratio": ar,
                "mean_intensity": img_rgb.mean(),
            }
        )

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Analyzed {valid_images_count} valid images.")

    print("\nImage Dimensions:")
    print(
        f"  Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"  Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"  Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    # Check for constant dimensions
    if widths.std() == 0 and heights.std() == 0:
        print("  -> All images have identical dimensions.")
    else:
        print("  -> Image dimensions vary.")

    # Channels
    print("\nChannel Distribution:")
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} images")

    # Pixel Stats
    if total_pixels > 0:
        mean_r = sum_r / total_pixels
        mean_g = sum_g / total_pixels
        mean_b = sum_b / total_pixels

        # Var = E[X^2] - (E[X])^2
        var_r = (sum_sq_r / total_pixels) - (mean_r**2)
        var_g = (sum_sq_g / total_pixels) - (mean_g**2)
        var_b = (sum_sq_b / total_pixels) - (mean_b**2)

        std_r = np.sqrt(max(0, var_r))
        std_g = np.sqrt(max(0, var_g))
        std_b = np.sqrt(max(0, var_b))

        print("\nGlobal Pixel Statistics (Normalized [0, 1]):")
        print(f"  Red   - Mean: {mean_r:.4f}, Std: {std_r:.4f}")
        print(f"  Green - Mean: {mean_g:.4f}, Std: {std_g:.4f}")
        print(f"  Blue  - Mean: {mean_b:.4f}, Std: {std_b:.4f}")
    else:
        print("\nGlobal Pixel Statistics: Unable to calculate (no pixels processed).")

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")
    print("Unstructured (Meta-Feature) Relationships:")

    if len(meta_features) > 0:
        meta_df = pd.DataFrame(meta_features)

        # Group by target and calculate mean of meta-features
        grouped = (
            meta_df.groupby("target")
            .agg(
                {
                    "width": "mean",
                    "height": "mean",
                    "aspect_ratio": "mean",
                    "mean_intensity": "mean",
                }
            )
            .reset_index()
        )

        print("\nAverage Meta-Features per Class:")
        print(
            f"{'Class':<20} | {'Width':<10} | {'Height':<10} | {'Aspect Ratio':<12} | {'Intensity':<10}"
        )
        print("-" * 75)
        for _, row in grouped.iterrows():
            print(
                f"{row['target']:<20} | {row['width']:<10.1f} | {row['height']:<10.1f} | {row['aspect_ratio']:<12.4f} | {row['mean_intensity']:<10.4f}"
            )

        # Check for significant differences (heuristic)
        print("\nObservations:")
        intensity_range = (
            grouped["mean_intensity"].max() - grouped["mean_intensity"].min()
        )
        if intensity_range > 0.1:
            print(
                f"  -> Significant variation in image brightness (intensity) across classes (Range: {intensity_range:.4f})."
            )
        else:
            print(
                f"  -> Image brightness is relatively consistent across classes (Range: {intensity_range:.4f})."
            )

        ar_range = grouped["aspect_ratio"].max() - grouped["aspect_ratio"].min()
        if ar_range > 0.1:
            print(f"  -> Aspect ratios vary significantly between classes.")
        else:
            print(f"  -> Aspect ratios are consistent across classes.")

    else:
        print("No meta-features extracted.")


if __name__ == "__main__":
    main()
