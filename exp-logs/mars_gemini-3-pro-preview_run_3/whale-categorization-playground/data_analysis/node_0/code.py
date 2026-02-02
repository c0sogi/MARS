import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
import warnings
from collections import Counter


# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Set seeds for reproducibility
    set_seed(42)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Paths
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # -------------------------------------------------------------------------
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # -------------------------------------------------------------------------
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    target_col = "Id"
    ids = df_train[target_col].values
    total_samples = len(ids)
    unique_classes = np.unique(ids)
    num_classes = len(unique_classes)

    class_counts = Counter(ids)
    sorted_counts = class_counts.most_common()

    # Imbalance Metrics
    counts = np.array(list(class_counts.values()))
    min_samples = np.min(counts)
    max_samples = np.max(counts)
    mean_samples = np.mean(counts)
    median_samples = np.median(counts)

    # Specific check for 'new_whale'
    new_whale_count = class_counts.get("new_whale", 0)
    new_whale_ratio = new_whale_count / total_samples

    # Top 5 classes excluding new_whale if possible
    top_5 = sorted_counts[:5]

    print(f"Total Samples: {total_samples}")
    print(f"Number of Unique Classes: {num_classes}")
    print(f"Class Balance Ratio (Max/Min): {max_samples/min_samples:.4f}")
    print(f"Samples per Class - Mean: {mean_samples:.4f}, Median: {median_samples:.4f}")
    print(
        f"Most Frequent Class ('new_whale'): {new_whale_count} samples ({new_whale_ratio*100:.4f}%)"
    )

    print("Top 5 Classes:")
    for cls, count in top_5:
        print(f"  {cls}: {count} ({count/total_samples*100:.4f}%)")

    # Singleton analysis
    singletons = sum(1 for c in counts if c == 1)
    print(
        f"Number of Singleton Classes (1 sample): {singletons} ({singletons/num_classes*100:.4f}% of classes)"
    )

    # -------------------------------------------------------------------------
    # SECTION 2: INPUT DATA ANALYSIS (IMAGE)
    # -------------------------------------------------------------------------
    print("\nSECTION 2: INPUT DATA ANALYSIS (IMAGE)")

    # We will iterate through images to collect stats
    # Stats to collect: Widths, Heights, Aspect Ratios, Channels
    # Pixel stats: Mean, Std (Channel-wise)

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = Counter()

    # Accumulators for pixel stats (assuming RGB for simplicity in reporting,
    # but we will handle grayscale by treating as 1 channel)
    # We will compute global mean/std over all pixels
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixel_count = 0

    # Limit processing for speed if necessary, but 6800 is manageable.
    # We'll process all to be robust.

    processed_count = 0
    missing_count = 0

    # Pre-construct full paths
    # file_path in metadata is relative (e.g. "train/img.jpg")
    # input dir is "./input"
    # full path = "./input/train/img.jpg"

    full_paths = [os.path.join(INPUT_DIR, p) for p in df_train["file_path"]]

    for idx, fpath in enumerate(full_paths):
        if not os.path.exists(fpath):
            missing_count += 1
            continue

        # Read image
        # IMREAD_UNCHANGED to detect if grayscale or alpha channel exists
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)

        if img is None:
            missing_count += 1
            continue

        h, w = img.shape[:2]

        # Check channels
        if len(img.shape) == 2:
            c = 1  # Grayscale
            # Convert to RGB for pixel stats consistency
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            c = img.shape[2]
            if c == 3:
                img_rgb = img  # BGR
            elif c == 4:
                # Drop alpha for stats
                img_rgb = img[:, :, :3]
            else:
                # Fallback
                img_rgb = img

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channel_counts[c] += 1

        # Pixel Stats Accumulation (Normalize 0-255 to 0-1 for calculation to avoid overflow, then scale back or report 0-255)
        # Using 0-255 scale for reporting
        img_flat = img_rgb.reshape(-1, 3).astype(np.float64)
        pixel_sum += img_flat.sum(axis=0)
        pixel_sq_sum += (img_flat**2).sum(axis=0)
        total_pixel_count += h * w

        processed_count += 1

    # Convert lists to arrays for stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Images Processed: {processed_count}")
    if missing_count > 0:
        print(f"Missing/Unreadable Images: {missing_count}")

    print("Dimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    print("Channels:")
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} images")

    # Calculate Global Pixel Stats (BGR order from OpenCV)
    if total_pixel_count > 0:
        global_mean = pixel_sum / total_pixel_count
        # Variance = E[X^2] - (E[X])^2
        global_var = (pixel_sq_sum / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)

        # OpenCV reads as BGR, usually we report RGB
        rgb_mean = global_mean[::-1]
        rgb_std = global_std[::-1]

        print("Pixel Statistics (RGB, 0-255 scale):")
        print(f"  Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
        print(f"  Std : R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")

    # -------------------------------------------------------------------------
    # SECTION 3: FEATURE/SIGNAL RELATIONSHIPS
    # -------------------------------------------------------------------------
    print("\nSECTION 3: FEATURE/SIGNAL RELATIONSHIPS")

    # Create a temporary dataframe for correlation analysis
    # We want to see if 'new_whale' class has different image characteristics than identified whales

    # Filter df_train to match the processed images order?
    # The loop order was based on df_train['file_path'], but we skipped missing.
    # Assuming no missing files based on previous metadata validation, we can align directly.
    # If missing > 0, we need to be careful.

    if processed_count == len(df_train):
        df_stats = df_train.copy()
        df_stats["width"] = widths
        df_stats["height"] = heights
        df_stats["aspect_ratio"] = aspect_ratios
        df_stats["is_new_whale"] = (df_stats["Id"] == "new_whale").astype(int)

        # 1. Correlation between Image Size (Area) and Class (New vs Known)
        df_stats["area"] = df_stats["width"] * df_stats["height"]

        # Point Biserial Correlation equivalent (Pearson between binary and continuous)
        corr_area = df_stats["area"].corr(df_stats["is_new_whale"])
        corr_ar = df_stats["aspect_ratio"].corr(df_stats["is_new_whale"])

        print("Relationship between Metadata and Target (is_new_whale):")
        print(f"  Correlation (Image Area vs is_new_whale): {corr_area:.4f}")
        print(f"  Correlation (Aspect Ratio vs is_new_whale): {corr_ar:.4f}")

        # Group stats
        grp = df_stats.groupby("is_new_whale")[["area", "aspect_ratio"]].mean()
        print("\n  Average Stats by Class Type:")
        print(
            f"    Known Whales - Avg Area: {grp.loc[0, 'area']:.4f}, Avg AR: {grp.loc[0, 'aspect_ratio']:.4f}"
        )
        print(
            f"    New Whales   - Avg Area: {grp.loc[1, 'area']:.4f}, Avg AR: {grp.loc[1, 'aspect_ratio']:.4f}"
        )

        if abs(corr_area) < 0.1 and abs(corr_ar) < 0.1:
            print(
                "\n  Insight: Image dimensions and aspect ratios are weakly correlated with the 'new_whale' label."
            )
            print(
                "           This suggests image quality/size is consistent across new and known whales."
            )
        else:
            print(
                "\n  Insight: Some correlation detected between image properties and label type."
            )

    else:
        print(
            "Skipping detailed relationship analysis due to mismatch in processed images and metadata."
        )


if __name__ == "__main__":
    main()
