import os
import pandas as pd
import numpy as np
import cv2
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import random

# ==========================================
# Configuration & Setup
# ==========================================
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 2000  # Number of images to sample for pixel/dimension analysis
NUM_WORKERS = 12  # Available vCPUs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)

# ==========================================
# Helper Functions
# ==========================================


def get_image_stats(row_tuple):
    """
    Worker function to process a single image.
    Returns a dictionary of stats or None if failure.
    row_tuple: (index, file_path)
    """
    idx, rel_path = row_tuple
    full_path = os.path.join(INPUT_DIR, rel_path)

    try:
        # cv2.imread loads as BGR
        img = cv2.imread(full_path)
        if img is None:
            return None

        height, width, channels = img.shape

        # Calculate pixel stats (normalize to 0-1 range for calculation then report)
        # Using float32 for precision
        img_float = img.astype(np.float32) / 255.0

        # Global mean/std for this image (across all pixels)
        # We also want per-channel, but for the summary report global is often sufficient
        # or we average the channels.

        mean_b = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_r = np.mean(img_float[:, :, 2])

        std_b = np.std(img_float[:, :, 0])
        std_g = np.std(img_float[:, :, 1])
        std_r = np.std(img_float[:, :, 2])

        return {
            "index": idx,
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height > 0 else 0,
            "channels": channels,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "std_r": std_r,
            "std_g": std_g,
            "std_b": std_b,
            "brightness": (mean_r + mean_g + mean_b) / 3.0,
        }
    except Exception:
        return None


# ==========================================
# Main Analysis
# ==========================================


def main():
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # ==========================================
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    # Parse labels (Space delimited)
    # Flatten all labels into a single list
    all_labels_list = []
    label_counts_per_row = []

    for labels_str in df["labels"]:
        lbls = labels_str.split()
        all_labels_list.extend(lbls)
        label_counts_per_row.append(len(lbls))

    total_samples = len(df)
    unique_labels = sorted(list(set(all_labels_list)))
    label_counts = Counter(all_labels_list)

    print(f"Total Samples: {total_samples}")
    print(f"Total Unique Labels: {len(unique_labels)}")
    print("\nLabel Distribution & Imbalance:")

    # Calculate ratios
    # In multi-label, sum of ratios > 1.0 is expected.
    # We report frequency relative to total samples.

    stats_data = []
    for lbl in unique_labels:
        count = label_counts[lbl]
        ratio = count / total_samples
        stats_data.append((lbl, count, ratio))

    # Sort by count descending
    stats_data.sort(key=lambda x: x[1], reverse=True)

    for lbl, count, ratio in stats_data:
        print(f"  - {lbl:<20}: Count = {count:<5}, Ratio = {ratio:.4f}")

    print("\nMulti-Label Statistics:")
    avg_labels = np.mean(label_counts_per_row)
    print(f"  - Average Labels per Image: {avg_labels:.4f}")

    # ==========================================
    # SECTION 2: INPUT DATA ANALYSIS (IMAGE)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Sample data for image analysis to save time
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        df_sample = df.copy()

    print(f"Analyzing {len(df_sample)} sampled images for pixel/dimension stats...")

    # Prepare arguments for parallel processing
    # passing (index, file_path)
    tasks = [(idx, row["file_path"]) for idx, row in df_sample.iterrows()]

    results = []
    # Process serially to avoid PicklingError in exec() environment
    temp_results = [get_image_stats(task) for task in tasks]

    # Filter out Nones
    results = [r for r in temp_results if r is not None]

    if not results:
        print("Error: No images could be processed.")
        return

    df_img_stats = pd.DataFrame(results)

    # Dimensions
    print("\nDimensions:")
    print(
        f"  - Width:  Mean = {df_img_stats['width'].mean():.4f}, Std = {df_img_stats['width'].std():.4f}, Min = {df_img_stats['width'].min()}, Max = {df_img_stats['width'].max()}"
    )
    print(
        f"  - Height: Mean = {df_img_stats['height'].mean():.4f}, Std = {df_img_stats['height'].std():.4f}, Min = {df_img_stats['height'].min()}, Max = {df_img_stats['height'].max()}"
    )
    print(
        f"  - Aspect Ratio: Mean = {df_img_stats['aspect_ratio'].mean():.4f}, Std = {df_img_stats['aspect_ratio'].std():.4f}"
    )

    # Channels
    channel_counts = df_img_stats["channels"].value_counts()
    print("\nChannels:")
    for ch, count in channel_counts.items():
        print(f"  - {ch} Channels: {count} images ({count/len(df_img_stats):.4f})")

    # Pixel Stats (Normalized 0-1)
    print("\nPixel Statistics (Normalized 0-1):")
    print(
        f"  - Mean R: {df_img_stats['mean_r'].mean():.4f}, Std R: {df_img_stats['std_r'].mean():.4f}"
    )
    print(
        f"  - Mean G: {df_img_stats['mean_g'].mean():.4f}, Std G: {df_img_stats['std_g'].mean():.4f}"
    )
    print(
        f"  - Mean B: {df_img_stats['mean_b'].mean():.4f}, Std B: {df_img_stats['std_b'].mean():.4f}"
    )

    # ==========================================
    # SECTION 3: FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("Analyzing relationship between Image Meta-Features and Disease Labels...")

    # We need to join the image stats back with the labels.
    # df_sample has the labels. df_img_stats has the stats and an 'index' column matching df_sample's index.

    # One-Hot Encode Labels for the sample
    # We create a binary matrix for the sampled data

    # Re-index df_img_stats to match df_sample via the 'index' column we stored
    df_img_stats.set_index("index", inplace=True)

    # Merge
    df_merged = df_sample.join(df_img_stats, how="inner")

    # Create binary columns for each unique label
    for lbl in unique_labels:
        df_merged[f"lbl_{lbl}"] = df_merged["labels"].apply(
            lambda x: 1 if lbl in x.split() else 0
        )

    # Meta features to correlate
    meta_features = ["width", "height", "aspect_ratio", "brightness"]

    print("\nCorrelation (Pearson) between Meta-Features and Target Presence:")
    print(
        f"{'Label':<20} | {'Width':<10} | {'Height':<10} | {'Aspect':<10} | {'Bright':<10}"
    )
    print("-" * 75)

    for lbl in unique_labels:
        target_col = f"lbl_{lbl}"
        corrs = []
        for meta in meta_features:
            if df_merged[target_col].std() == 0:
                corr = 0.0
            else:
                corr = df_merged[target_col].corr(df_merged[meta])
            corrs.append(corr)

        print(
            f"{lbl:<20} | {corrs[0]:<10.4f} | {corrs[1]:<10.4f} | {corrs[2]:<10.4f} | {corrs[3]:<10.4f}"
        )

    # Check for collinearity among meta-features
    print("\nMeta-Feature Redundancy (Correlation > 0.90):")
    corr_matrix = df_merged[meta_features].corr().abs()
    high_corr_pairs = []
    for i in range(len(meta_features)):
        for j in range(i + 1, len(meta_features)):
            if corr_matrix.iloc[i, j] > 0.90:
                high_corr_pairs.append(
                    (meta_features[i], meta_features[j], corr_matrix.iloc[i, j])
                )

    if high_corr_pairs:
        for f1, f2, val in high_corr_pairs:
            print(f"  - {f1} & {f2}: {val:.4f}")
    else:
        print("  - No highly collinear meta-features found.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
