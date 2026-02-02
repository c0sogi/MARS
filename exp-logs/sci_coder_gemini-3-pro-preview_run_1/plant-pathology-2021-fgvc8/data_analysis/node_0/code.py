import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

# --- Configuration & Setup ---
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Suppress warnings
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")


def get_full_path(rel_path):
    return os.path.join(INPUT_DIR, rel_path)


def analyze_targets(df):
    print_section("TARGET VARIABLE ANALYSIS")

    # Parse labels (space delimited)
    df["label_list"] = df["labels"].apply(lambda x: x.split())

    # Flatten list to count individual labels
    all_labels = [label for sublist in df["label_list"] for label in sublist]
    label_counts = Counter(all_labels)
    total_images = len(df)

    print(f"Total Training Images: {total_images}")
    print(f"Total Unique Labels: {len(label_counts)}")

    # 1. Distribution & Imbalance
    print("\n--- Class Distribution & Balance ---")
    df_stats = pd.DataFrame.from_dict(label_counts, orient="index", columns=["count"])
    df_stats["frequency_ratio"] = df_stats["count"] / total_images
    df_stats = df_stats.sort_values("count", ascending=False)

    print(f"{'Label':<25} {'Count':<10} {'Freq Ratio'}")
    print("-" * 50)
    for label, row in df_stats.iterrows():
        print(f"{label:<25} {int(row['count']):<10} {row['frequency_ratio']:.4f}")

    # 2. Multi-label Analysis
    print("\n--- Multi-Label Analysis ---")
    df["num_labels"] = df["label_list"].apply(len)
    labels_per_image_counts = df["num_labels"].value_counts().sort_index()

    print("Labels per Image Distribution:")
    for num, count in labels_per_image_counts.items():
        ratio = count / total_images
        print(f"  {num} label(s): {count} images ({ratio:.4f})")

    return df


def analyze_images(df):
    print_section("IMAGE DATA ANALYSIS")

    # We will iterate through all images to get dimensions (fast with PIL)
    # We will use a sample to calculate pixel stats (slower)

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # For pixel stats (Welford's algorithm or simple accumulation)
    # We'll use accumulation on a sample for speed
    pixel_sample_size = min(2000, len(df))
    pixel_sample_indices = np.random.choice(df.index, pixel_sample_size, replace=False)

    r_sum, g_sum, b_sum = 0, 0, 0
    r_sq_sum, g_sq_sum, b_sq_sum = 0, 0, 0
    total_pixels = 0

    print(f"Analyzing dimensions for {len(df)} images...")
    print(f"Calculating pixel stats on a sample of {pixel_sample_size} images...")

    for idx, row in df.iterrows():
        try:
            full_path = get_full_path(row["file_path"])
            with Image.open(full_path) as img:
                # Dimensions
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h != 0 else 0)

                mode = img.mode
                if mode == "RGB":
                    c = 3
                elif mode == "L":
                    c = 1
                elif mode == "RGBA":
                    c = 4
                else:
                    c = len(img.getbands())
                channels.append(c)

                # Pixel Stats (only for sample)
                if idx in pixel_sample_indices:
                    # Convert to RGB for consistency in stats
                    img_rgb = img.convert("RGB")
                    arr = np.array(img_rgb) / 255.0

                    r_sum += np.sum(arr[:, :, 0])
                    g_sum += np.sum(arr[:, :, 1])
                    b_sum += np.sum(arr[:, :, 2])

                    r_sq_sum += np.sum(arr[:, :, 0] ** 2)
                    g_sq_sum += np.sum(arr[:, :, 1] ** 2)
                    b_sq_sum += np.sum(arr[:, :, 2] ** 2)

                    total_pixels += w * h

        except Exception as e:
            # In case of corrupt image, skip
            continue

    # 1. Dimensions
    print("\n--- Image Dimensions ---")
    w_series = pd.Series(widths)
    h_series = pd.Series(heights)
    ar_series = pd.Series(aspect_ratios)

    print(
        f"Width  - Mean: {w_series.mean():.4f}, Std: {w_series.std():.4f}, Min: {w_series.min()}, Max: {w_series.max()}"
    )
    print(
        f"Height - Mean: {h_series.mean():.4f}, Std: {h_series.std():.4f}, Min: {h_series.min()}, Max: {h_series.max()}"
    )
    print(f"Aspect Ratio - Mean: {ar_series.mean():.4f}, Std: {ar_series.std():.4f}")

    # 2. Channels
    print("\n--- Channel Distribution ---")
    chan_counts = Counter(channels)
    for c, count in chan_counts.items():
        print(f"  {c} Channels: {count} images")

    # 3. Pixel Stats
    print("\n--- Pixel Value Statistics (Normalized [0, 1]) ---")
    if total_pixels > 0:
        r_mean = r_sum / total_pixels
        g_mean = g_sum / total_pixels
        b_mean = b_sum / total_pixels

        r_std = np.sqrt((r_sq_sum / total_pixels) - (r_mean**2))
        g_std = np.sqrt((g_sq_sum / total_pixels) - (g_mean**2))
        b_std = np.sqrt((b_sq_sum / total_pixels) - (b_mean**2))

        print(f"Red   - Mean: {r_mean:.4f}, Std: {r_std:.4f}")
        print(f"Green - Mean: {g_mean:.4f}, Std: {g_std:.4f}")
        print(f"Blue  - Mean: {b_mean:.4f}, Std: {b_std:.4f}")
    else:
        print("Could not calculate pixel stats (no pixels processed).")

    # Add metadata to DF for relationship analysis
    df["width"] = w_series
    df["height"] = h_series
    df["aspect_ratio"] = ar_series

    return df


def analyze_relationships(df):
    print_section("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Label Co-occurrence
    print("\n--- Label Co-occurrence (Top Pairs) ---")
    # Get all pairs of labels within images that have > 1 label
    multi_label_df = df[df["num_labels"] > 1]
    pair_counts = defaultdict(int)

    for labels in multi_label_df["label_list"]:
        labels = sorted(labels)  # Sort to ensure (A, B) is same as (B, A)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pair_counts[(labels[i], labels[j])] += 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_pairs:
        print("No co-occurring labels found.")
    else:
        print(f"{'Label Pair':<40} {'Count'}")
        print("-" * 50)
        for (l1, l2), count in sorted_pairs[:10]:
            print(f"{f'{l1} & {l2}':<40} {count}")

    # 2. Metadata vs Target Correlation
    print("\n--- Metadata vs Target Correlation ---")
    # One-hot encode labels
    mlb = MultiLabelBinarizer()
    labels_onehot = mlb.fit_transform(df["label_list"])
    df_onehot = pd.DataFrame(labels_onehot, columns=mlb.classes_, index=df.index)

    # Combine with metadata
    meta_cols = ["width", "height", "aspect_ratio"]
    # Drop rows where metadata might be NaN (if image load failed)
    analysis_df = pd.concat([df[meta_cols], df_onehot], axis=1).dropna()

    if analysis_df.empty:
        print("Insufficient data for correlation analysis.")
        return

    print("Correlation (Pearson) between Image Metadata and Disease Presence:")
    print(f"{'Label':<25} {'Width':<10} {'Height':<10} {'Aspect Ratio'}")
    print("-" * 60)

    for label in mlb.classes_:
        corr_w = analysis_df[label].corr(analysis_df["width"])
        corr_h = analysis_df[label].corr(analysis_df["height"])
        corr_ar = analysis_df[label].corr(analysis_df["aspect_ratio"])
        print(f"{label:<25} {corr_w:.4f}     {corr_h:.4f}     {corr_ar:.4f}")

    # 3. Size vs Complexity
    # Check if 'complex' class images are larger/smaller on average
    if "complex" in mlb.classes_:
        print("\n--- 'Complex' Class Image Statistics ---")
        complex_imgs = analysis_df[analysis_df["complex"] == 1]
        non_complex_imgs = analysis_df[analysis_df["complex"] == 0]

        print(
            f"Complex Images (n={len(complex_imgs)}) - Mean Area: {(complex_imgs['width']*complex_imgs['height']).mean():.0f}"
        )
        print(
            f"Other Images   (n={len(non_complex_imgs)}) - Mean Area: {(non_complex_imgs['width']*non_complex_imgs['height']).mean():.0f}"
        )


def main():
    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # Run Analysis
    df_train = analyze_targets(df_train)
    df_train = analyze_images(df_train)
    analyze_relationships(df_train)

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
