import os
import sys
import numpy as np
import pandas as pd
import cv2
import random
import warnings
from collections import Counter

# Suppress warnings
warnings.filterwarnings("ignore")

# Set fixed random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def perform_eda():
    # Paths
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train_metadata.csv"

    # Load Metadata
    try:
        df = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print("Error: Metadata file not found.")
        return

    # --- 1. DATA INTEGRITY ---
    # Analysis is strictly performed on the training set loaded from metadata
    num_samples = len(df)

    # --- 2. PRE-PROCESSING & EXTRACTION ---
    # We need to iterate through the dataset to collect stats.
    # To save memory and time, we will process in a single pass where possible.

    # Target Stats Storage
    all_classes = []
    box_widths = []
    box_heights = []
    box_areas = []
    box_ratios = []
    anns_per_image = []

    # Image Stats Storage
    img_widths = []
    img_heights = []
    img_ratios = []
    img_areas = []

    # Pixel Stats Accumulators (for Welford's or simple sum aggregation)
    # Using simple sum aggregation for speed on 2k images
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    # Iterate through data
    # We will sample pixel stats from every image to be thorough,
    # but we must be careful with I/O time. 2245 images is manageable.

    for _, row in df.iterrows():
        # --- Process Labels ---
        label_str = row["labels"]
        if pd.isna(label_str) or label_str == "":
            anns_per_image.append(0)
        else:
            parts = label_str.strip().split(" ")
            # Format: Unicode X Y W H ...
            # Count annotations
            n_anns = len(parts) // 5
            anns_per_image.append(n_anns)

            for i in range(n_anns):
                base = i * 5
                try:
                    cls = parts[base]
                    # x = int(parts[base+1]) # Not needed for distribution stats
                    # y = int(parts[base+2])
                    w = int(parts[base + 3])
                    h = int(parts[base + 4])

                    all_classes.append(cls)
                    box_widths.append(w)
                    box_heights.append(h)
                    box_areas.append(w * h)
                    if h > 0:
                        box_ratios.append(w / h)
                    else:
                        box_ratios.append(0)
                except (ValueError, IndexError):
                    continue

        # --- Process Images ---
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        # Check existence
        if not os.path.exists(img_path):
            continue

        # Read Image
        # cv2.imread loads as BGR
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, c = img.shape
        img_widths.append(w)
        img_heights.append(h)
        img_areas.append(w * h)
        img_ratios.append(w / h)

        # Pixel Stats
        # Normalize to 0-1 for calculation to avoid overflow, then scale back or keep as is.
        # Keeping as 0-255 for standard reporting.
        img_float = img.astype(np.float64)

        # Sum per channel
        channel_sum += np.sum(img_float, axis=(0, 1))
        channel_sq_sum += np.sum(img_float**2, axis=(0, 1))
        total_pixel_count += w * h

    # --- 3. CALCULATE STATISTICS ---

    # Target Variable Analysis
    total_anns = len(all_classes)
    class_counts = Counter(all_classes)
    sorted_classes = class_counts.most_common()
    num_unique_classes = len(class_counts)

    # Box Stats
    bw = np.array(box_widths)
    bh = np.array(box_heights)
    ba = np.array(box_areas)
    br = np.array(box_ratios)

    # Image Stats
    iw = np.array(img_widths)
    ih = np.array(img_heights)
    ir = np.array(img_ratios)

    # Pixel Stats
    # BGR to RGB for reporting
    rgb_means = (channel_sum / total_pixel_count)[::-1]
    rgb_std = (
        np.sqrt(
            (channel_sq_sum / total_pixel_count)
            - (channel_sum / total_pixel_count) ** 2
        )
    )[::-1]

    # --- 4. OUTPUT GENERATION ---

    print("DATA INTEGRITY")
    print(f"Analysis performed on Training Set only.")
    print(f"Total Images Analyzed: {num_samples}")
    print("-" * 30)

    print("TARGET VARIABLE ANALYSIS")
    print("1. Classification (Character Labels):")
    print(f"Total Annotations: {total_anns}")
    print(f"Unique Classes (Cardinality): {num_unique_classes}")

    if num_unique_classes > 0:
        most_common = sorted_classes[0]
        least_common = sorted_classes[-1]
        print(
            f"Most Common Class: {most_common[0]} (Count: {most_common[1]}, Freq: {most_common[1]/total_anns:.4f})"
        )
        print(
            f"Least Common Class: {least_common[0]} (Count: {least_common[1]}, Freq: {least_common[1]/total_anns:.4f})"
        )

        # Class Balance Ratio (Max / Min)
        balance_ratio = most_common[1] / max(1, least_common[1])
        print(f"Class Imbalance Ratio (Max/Min): {balance_ratio:.4f}")

        # Rare labels (< 1%)
        threshold = total_anns * 0.01
        rare_count = sum(1 for c, count in class_counts.items() if count < threshold)
        print(
            f"Classes with < 1% Frequency: {rare_count} ({rare_count/num_unique_classes:.4f} of classes)"
        )

    print("\n2. Regression (Bounding Boxes):")
    if len(bw) > 0:
        print(
            f"Box Width:  Mean={bw.mean():.4f}, Std={bw.std():.4f}, Min={bw.min()}, Max={bw.max()}"
        )
        print(
            f"Box Height: Mean={bh.mean():.4f}, Std={bh.std():.4f}, Min={bh.min()}, Max={bh.max()}"
        )
        print(f"Box Area:   Mean={ba.mean():.4f}, Std={ba.std():.4f}")
        print(f"Box Aspect Ratio (W/H): Mean={br.mean():.4f}, Std={br.std():.4f}")

        # Skewness/Kurtosis for Area (Proxy for normality of object size)
        # Using simple calculation to avoid scipy dependency if not strictly needed, but scipy is allowed.
        # We'll stick to basic numpy for robustness.
        skew = np.mean((ba - np.mean(ba)) ** 3) / (np.std(ba) ** 3)
        kurt = np.mean((ba - np.mean(ba)) ** 4) / (np.std(ba) ** 4) - 3
        print(f"Box Area Skewness: {skew:.4f}")
        print(f"Box Area Kurtosis: {kurt:.4f}")
    else:
        print("No bounding boxes found.")
    print("-" * 30)

    print("INPUT DATA ANALYSIS (IMAGE DATA)")
    print(
        f"Dimensions (Width):  Mean={iw.mean():.4f}, Std={iw.std():.4f}, Min={iw.min()}, Max={iw.max()}"
    )
    print(
        f"Dimensions (Height): Mean={ih.mean():.4f}, Std={ih.std():.4f}, Min={ih.min()}, Max={ih.max()}"
    )
    print(f"Aspect Ratio (W/H):  Mean={ir.mean():.4f}, Std={ir.std():.4f}")

    # Check for grayscale vs RGB
    # We loaded as BGR. If std deviation between channels is 0, it's grayscale.
    # But we aggregated globally. We report global stats.
    print(
        f"Pixel Values (RGB) Mean: [{rgb_means[0]:.4f}, {rgb_means[1]:.4f}, {rgb_means[2]:.4f}]"
    )
    print(
        f"Pixel Values (RGB) Std:  [{rgb_std[0]:.4f}, {rgb_std[1]:.4f}, {rgb_std[2]:.4f}]"
    )
    print("-" * 30)

    print("FEATURE/SIGNAL RELATIONSHIPS")
    # Structured Relationships (Meta-features)
    # Create a small dataframe for correlation
    meta_df = pd.DataFrame(
        {
            "img_width": img_widths,
            "img_height": img_heights,
            "img_area": img_areas,
            "num_anns": anns_per_image,
        }
    )

    # Correlation Matrix
    corr = meta_df.corr(method="pearson")

    print("Correlation with Target (Number of Annotations):")
    print(f"  vs Image Width:  {corr.loc['img_width', 'num_anns']:.4f}")
    print(f"  vs Image Height: {corr.loc['img_height', 'num_anns']:.4f}")
    print(f"  vs Image Area:   {corr.loc['img_area', 'num_anns']:.4f}")

    # Redundancy Check
    print("\nMeta-Feature Redundancy (Correlation > 0.90):")
    redundant_pairs = []
    for c1 in meta_df.columns:
        for c2 in meta_df.columns:
            if c1 != c2 and c1 < c2:  # Avoid duplicates and self
                val = corr.loc[c1, c2]
                if abs(val) > 0.90:
                    redundant_pairs.append(f"{c1} & {c2} ({val:.4f})")

    if redundant_pairs:
        for pair in redundant_pairs:
            print(f"  {pair}")
    else:
        print("  No highly collinear meta-features found.")


if __name__ == "__main__":
    perform_eda()
