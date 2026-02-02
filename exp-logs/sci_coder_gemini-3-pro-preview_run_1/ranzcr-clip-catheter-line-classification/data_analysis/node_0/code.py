import os
import pandas as pd
import numpy as np
import cv2
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_targets(df, target_cols):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # 1. Distribution / Class Balance
    print("Class Balance (Positive Label Frequency):")
    stats = []
    for col in target_cols:
        count = df[col].sum()
        ratio = count / len(df)
        stats.append((col, count, ratio))

    # Sort by frequency descending
    stats.sort(key=lambda x: x[2], reverse=True)

    for col, count, ratio in stats:
        print(f"  {col}: {count} ({ratio:.4f})")

    # 2. Multi-label analysis
    # How many labels does an average image have?
    df["label_count"] = df[target_cols].sum(axis=1)
    mean_labels = df["label_count"].mean()
    max_labels = df["label_count"].max()
    min_labels = df["label_count"].min()

    print(f"\nLabel Cardinality per Image:")
    print(f"  Mean Labels per Image: {mean_labels:.4f}")
    print(f"  Min Labels: {min_labels}")
    print(f"  Max Labels: {max_labels}")

    # Distribution of label counts
    counts = df["label_count"].value_counts().sort_index()
    print("  Distribution of Label Counts:")
    for k, v in counts.items():
        print(f"    {k} labels: {v} images ({v/len(df):.4f})")


def analyze_images(df, input_dir, sample_size=1000):
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    # Sample to save time
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42).copy()
        print(f"Analysis performed on a random sample of {sample_size} images.")
    else:
        sample_df = df.copy()
        print(f"Analysis performed on all {len(df)} images.")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # Pixel stats accumulators
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    missing_files = 0

    for idx, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        if not os.path.exists(full_path):
            missing_files += 1
            continue

        # Read image
        # cv2.imread loads as BGR by default.
        # We load unchanged to detect if it's truly grayscale or RGB
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            missing_files += 1
            continue

        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels.append(c)

        # For pixel stats, convert to float and normalize to 0-1 temporarily for stability if needed,
        # but here we keep 0-255 scale for reporting.
        # Flatten
        flat_img = img.flatten().astype(np.float64)
        pixel_sum += flat_img.sum()
        pixel_sq_sum += (flat_img**2).sum()
        pixel_count += len(flat_img)

    if missing_files > 0:
        print(f"Warning: {missing_files} files could not be loaded.")

    # Dimensions
    if widths:
        print("Image Dimensions:")
        print(
            f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(
            f"  Aspect Ratio (W/H): Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
        )

    # Channels
    if channels:
        unique_channels, counts = np.unique(channels, return_counts=True)
        print("Channel Distribution:")
        for uc, ucount in zip(unique_channels, counts):
            print(f"  {uc} channels: {ucount} images ({ucount/len(channels):.4f})")

    # Pixel Stats
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        global_var = (pixel_sq_sum / pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)

        print("Pixel Intensity Statistics (0-255 scale):")
        print(f"  Global Mean: {global_mean:.4f}")
        print(f"  Global Std Dev: {global_std:.4f}")

    return widths, heights, aspect_ratios


def analyze_relationships(df, target_cols, widths, heights, sample_indices):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # 1. Label Correlations (Co-occurrence)
    print("Label Co-occurrence (Top Correlations):")
    corr_matrix = df[target_cols].corr()

    # Extract upper triangle to avoid duplicates and self-correlation
    corrs = []
    for i in range(len(target_cols)):
        for j in range(i + 1, len(target_cols)):
            c1 = target_cols[i]
            c2 = target_cols[j]
            val = corr_matrix.iloc[i, j]
            corrs.append((c1, c2, val))

    # Sort by absolute correlation
    corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    for c1, c2, val in corrs[:5]:
        print(f"  {c1} vs {c2}: {val:.4f}")

    # 2. Meta-feature Relationships (Image Size vs Labels)
    # We need to map the sampled widths/heights back to the dataframe rows
    # Since we used sample_df in analyze_images, we can assume the order matches if we are careful.
    # However, to be robust, we'll just re-calculate for the whole DF or the sample DF if passed correctly.
    # In this script structure, we passed lists. Let's assume the lists correspond to the sampled rows.

    if widths and len(widths) == len(sample_indices):
        # Create a mini dataframe for the sampled data
        meta_df = df.loc[sample_indices].copy()
        meta_df["width"] = widths
        meta_df["height"] = heights
        meta_df["area"] = meta_df["width"] * meta_df["height"]

        print("\nImage Area vs Target Presence (Point-Biserial approx):")
        # Check if larger images are more likely to have certain catheters
        # We'll check against 'CVC - Normal' (most common) and 'ETT - Normal'

        check_cols = ["CVC - Normal", "ETT - Normal", "Swan Ganz Catheter Present"]
        for col in check_cols:
            if col in df.columns:
                pos_area = meta_df[meta_df[col] == 1]["area"].mean()
                neg_area = meta_df[meta_df[col] == 0]["area"].mean()

                # Handle cases where a split might be empty in the sample
                if pd.isna(pos_area):
                    pos_area = 0
                if pd.isna(neg_area):
                    neg_area = 0

                print(f"  {col}:")
                print(f"    Mean Area (Positive): {pos_area:.0f}")
                print(f"    Mean Area (Negative): {neg_area:.0f}")
                diff_pct = ((pos_area - neg_area) / (neg_area + 1e-6)) * 100
                print(f"    Difference: {diff_pct:.2f}%")


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train_metadata.csv"

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load Data
    df = pd.read_csv(METADATA_PATH)

    # Identify Target Columns
    # Excluding IDs and file_path
    non_target = ["StudyInstanceUID", "PatientID", "file_path"]
    target_cols = [c for c in df.columns if c not in non_target and c != "label_count"]

    # 1. Target Analysis
    analyze_targets(df, target_cols)

    # 2. Image Analysis
    # We sample indices first to maintain alignment for relationship analysis
    SAMPLE_SIZE = 1000
    if len(df) > SAMPLE_SIZE:
        sample_indices = df.sample(n=SAMPLE_SIZE, random_state=42).index
        sample_df = df.loc[sample_indices]
    else:
        sample_indices = df.index
        sample_df = df

    widths, heights, aspect_ratios = analyze_images(sample_df, INPUT_DIR, SAMPLE_SIZE)

    # 3. Relationships
    analyze_relationships(df, target_cols, widths, heights, sample_indices)


if __name__ == "__main__":
    main()
