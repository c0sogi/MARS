import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy.stats import pearsonr

# --- Configuration ---
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE = 2500  # Number of images to sample for pixel/dimension stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    # --- 1. Data Integrity ---
    print("=== DATA INTEGRITY ===")
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Training Data Loaded. Shape: {df.shape}")
    print("Analysis performed strictly on the training set provided in metadata.")

    # Identify target columns (exclude IDs and paths)
    non_target_cols = ["StudyInstanceUID", "PatientID", "file_path"]
    target_cols = [c for c in df.columns if c not in non_target_cols]

    # --- 2. Target Variable Analysis ---
    print("\n=== TARGET VARIABLE ANALYSIS ===")

    # Distribution and Imbalance
    print("--- Class Distribution ---")
    stats = []
    for col in target_cols:
        count = df[col].sum()
        ratio = df[col].mean()
        stats.append((col, count, ratio))

    # Sort by prevalence
    stats.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Label':<30} {'Count':<10} {'Prevalence':<10}")
    print("-" * 55)
    for label, count, ratio in stats:
        print(f"{label:<30} {count:<10} {ratio:.4f}")

    # Multi-label analysis
    print("\n--- Label Co-occurrence ---")
    df["label_sum"] = df[target_cols].sum(axis=1)
    label_sum_counts = df["label_sum"].value_counts().sort_index()
    print("Number of active labels per image:")
    for num_labels, count in label_sum_counts.items():
        print(f"  {num_labels} labels: {count} images ({count/len(df):.4f})")

    # --- 3. Image Data Analysis ---
    print("\n=== IMAGE DATA ANALYSIS ===")

    # Sample data for image analysis to save time
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        df_sample = df.copy()

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # Accumulators for pixel stats (Welford's algorithm or simple sum/sq_sum)
    # Using simple sum for batch estimation
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0.0

    valid_samples = 0

    # Store meta-features for relationship analysis later
    meta_features = {
        "width": [],
        "height": [],
        "aspect_ratio": [],
        "mean_intensity": [],
    }

    # Keep track of indices for the sample dataframe to align meta-features
    sample_indices = []

    for idx, row in df_sample.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(full_path):
            continue

        # Read image
        # cv2.imread loads as BGR by default.
        img = cv2.imread(full_path)

        if img is None:
            continue

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels.append(c)

        # Pixel stats (normalize to 0-1 for calculation)
        img_norm = img.astype(np.float32) / 255.0

        # Just take the first channel for stats if it's grayscale saved as RGB
        # Check if channels are identical (grayscale check)
        # Optimization: Just use mean of channels or single channel
        # X-rays are typically grayscale.
        flat_pixels = img_norm[:, :, 0].flatten()

        pixel_sum += flat_pixels.sum()
        pixel_sq_sum += (flat_pixels**2).sum()
        pixel_count += flat_pixels.size

        # Meta-features for this image
        mean_val = flat_pixels.mean()
        meta_features["width"].append(w)
        meta_features["height"].append(h)
        meta_features["aspect_ratio"].append(w / h)
        meta_features["mean_intensity"].append(mean_val)
        sample_indices.append(idx)

        valid_samples += 1

    # Dimensions
    print("--- Dimensions ---")
    widths = np.array(widths)
    heights = np.array(heights)
    ars = np.array(aspect_ratios)

    print(
        f"Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"Aspect Ratio: Mean={ars.mean():.4f}, Std={ars.std():.4f}, Min={ars.min():.4f}, Max={ars.max():.4f}"
    )

    # Channels
    print("\n--- Channels ---")
    unique_channels, counts = np.unique(channels, return_counts=True)
    for c, count in zip(unique_channels, counts):
        print(f"  {c} Channels: {count} images")

    # Pixel Stats
    print("\n--- Pixel Intensity (Normalized 0-1) ---")
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        global_var = (pixel_sq_sum / pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)
        print(f"Global Mean: {global_mean:.4f}")
        print(f"Global Std:  {global_std:.4f}")
    else:
        print("No pixels processed.")

    # --- 4. Feature/Signal Relationships ---
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # A. Target-Target Correlations
    print("--- Target Co-occurrence (Pearson Correlation > 0.3 or < -0.3) ---")
    corr_matrix = df[target_cols].corr()

    # Get upper triangle pairs
    correlated_pairs = []
    for i in range(len(target_cols)):
        for j in range(i + 1, len(target_cols)):
            c1 = target_cols[i]
            c2 = target_cols[j]
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.3:
                correlated_pairs.append((c1, c2, corr))

    correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if correlated_pairs:
        for c1, c2, corr in correlated_pairs:
            print(f"{c1} vs {c2}: {corr:.4f}")
    else:
        print("No strong correlations (>0.3) found between targets.")

    # B. Meta-Feature vs Target Relationships
    # Create a dataframe for the sampled images with their meta-features and targets
    print("\n--- Meta-Features vs Targets (Point-Biserial Correlation) ---")
    print("(Checking if image properties correlate with specific labels)")

    df_meta = pd.DataFrame(meta_features, index=sample_indices)
    # Join with targets
    df_meta = df_meta.join(df.loc[sample_indices, target_cols])

    meta_cols = ["width", "height", "aspect_ratio", "mean_intensity"]

    # Calculate correlation between continuous meta-features and binary targets
    significant_meta_corrs = []

    for t_col in target_cols:
        # Skip if target has 0 variance in sample
        if df_meta[t_col].nunique() < 2:
            continue

        for m_col in meta_cols:
            # Point-biserial correlation is mathematically equivalent to Pearson
            # when one variable is binary and other is continuous
            corr, _ = pearsonr(df_meta[m_col], df_meta[t_col])
            if abs(corr) > 0.1:  # Threshold for reporting
                significant_meta_corrs.append((m_col, t_col, corr))

    significant_meta_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    if significant_meta_corrs:
        print(f"Top correlations (|r| > 0.1):")
        for m_col, t_col, corr in significant_meta_corrs[:10]:  # Top 10
            print(f"{m_col} vs {t_col}: {corr:.4f}")
    else:
        print(
            "No significant correlations (|r| > 0.1) found between image metadata and targets."
        )


if __name__ == "__main__":
    main()
