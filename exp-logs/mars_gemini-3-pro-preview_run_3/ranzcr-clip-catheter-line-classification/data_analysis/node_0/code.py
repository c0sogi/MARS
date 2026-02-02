import os
import pandas as pd
import numpy as np
import cv2
import random

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE = 2000  # Number of images to sample for pixel/dimension analysis


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Main Analysis Script
# -----------------------------------------------------------------------------
def main():
    set_seed(SEED)

    # -------------------------------------------------------------------------
    # SECTION 1: DATA INTEGRITY
    # -------------------------------------------------------------------------
    print("SECTION 1: DATA INTEGRITY")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"{METADATA_PATH} not found. Ensure metadata is generated."
        )

    df = pd.read_csv(METADATA_PATH)

    # Identify Label Columns
    # Exclude IDs and file_path
    non_label_cols = ["StudyInstanceUID", "PatientID", "file_path"]
    label_cols = [c for c in df.columns if c not in non_label_cols]

    print(f"Dataset Loaded: {METADATA_PATH}")
    print(f"Total Training Samples: {len(df)}")
    print(f"Number of Target Labels: {len(label_cols)}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # SECTION 2: TARGET VARIABLE ANALYSIS
    # -------------------------------------------------------------------------
    print("SECTION 2: TARGET VARIABLE ANALYSIS")

    # 2.1 Distribution
    print("Subsection: Label Distribution")
    print(f"{'Label':<30} | {'Count':<8} | {'Ratio':<8}")
    print("-" * 55)

    for col in label_cols:
        count = df[col].sum()
        ratio = count / len(df)
        print(f"{col:<30} | {count:<8} | {ratio:.4f}")

    # 2.2 Co-occurrence / Correlation
    print("\nSubsection: Label Correlations (Top 5 Pairs)")
    # Calculate correlation matrix
    corr_matrix = df[label_cols].corr()

    # Unstack to get pairs
    corr_pairs = corr_matrix.abs().unstack()
    # Sort descending
    corr_pairs = corr_pairs.sort_values(ascending=False)
    # Remove self-correlation (== 1.0)
    corr_pairs = corr_pairs[corr_pairs < 1.0 - 1e-9]

    # Deduplicate pairs (A, B) and (B, A)
    seen = set()
    unique_pairs = []
    for idx, val in corr_pairs.items():
        a, b = idx
        pair_key = tuple(sorted((a, b)))
        if pair_key not in seen:
            seen.add(pair_key)
            # Get the original signed correlation
            orig_val = corr_matrix.loc[a, b]
            unique_pairs.append((a, b, orig_val))

    for i, (a, b, val) in enumerate(unique_pairs[:5]):
        print(f"{i+1}. {a} vs {b}: {val:.4f}")

    print("-" * 40)

    # -------------------------------------------------------------------------
    # SECTION 3: INPUT DATA ANALYSIS (IMAGE MODALITY)
    # -------------------------------------------------------------------------
    print("SECTION 3: INPUT DATA ANALYSIS (IMAGE MODALITY)")

    # Sample data for image analysis to ensure runtime < 1 hour
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    print(f"Analyzing a subset of {len(sample_df)} images for pixel/dimension stats...")

    # Accumulators
    meta_records = []

    # Pixel stats accumulators (using simple sum for readability and sufficient precision with float64)
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    total_pixel_count = 0

    missing_count = 0

    for _, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            missing_count += 1
            continue

        # Read image
        # IMREAD_UNCHANGED to preserve channels/depth
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            missing_count += 1
            continue

        # Dimensions
        if len(img.shape) == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape

        # Pixel Stats Update
        # Flatten to 1D
        img_flat = img.flatten().astype(np.float64)
        sum_pixels += np.sum(img_flat)
        sum_sq_pixels += np.sum(img_flat**2)
        total_pixel_count += len(img_flat)

        # Record Meta
        rec = {
            "width": w,
            "height": h,
            "aspect_ratio": w / h if h > 0 else 0,
            "channels": c,
        }
        # Attach labels for later correlation analysis
        for l in label_cols:
            rec[l] = row[l]

        meta_records.append(rec)

    if missing_count > 0:
        print(f"Warning: {missing_count} images were missing or unreadable.")

    # Create DataFrame from collected meta stats
    meta_df = pd.DataFrame(meta_records)

    if len(meta_df) == 0:
        print("Error: No images processed.")
        return

    # 3.1 Dimensions
    print("\nSubsection: Image Dimensions")
    w_stats = meta_df["width"].describe()
    h_stats = meta_df["height"].describe()
    ar_stats = meta_df["aspect_ratio"].describe()

    print(
        f"Width        - Mean: {w_stats['mean']:.4f}, Std: {w_stats['std']:.4f}, Min: {w_stats['min']:.0f}, Max: {w_stats['max']:.0f}"
    )
    print(
        f"Height       - Mean: {h_stats['mean']:.4f}, Std: {h_stats['std']:.4f}, Min: {h_stats['min']:.0f}, Max: {h_stats['max']:.0f}"
    )
    print(
        f"Aspect Ratio - Mean: {ar_stats['mean']:.4f}, Std: {ar_stats['std']:.4f}, Min: {ar_stats['min']:.4f}, Max: {ar_stats['max']:.4f}"
    )

    # 3.2 Channels
    print("\nSubsection: Channel Distribution")
    channel_counts = meta_df["channels"].value_counts()
    for ch, count in channel_counts.items():
        print(f"Channel {ch}: {count} images ({count/len(meta_df):.4f})")

    # 3.3 Pixel Stats
    print("\nSubsection: Global Pixel Statistics")
    if total_pixel_count > 0:
        global_mean = sum_pixels / total_pixel_count
        global_var = (sum_sq_pixels / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var) if global_var > 0 else 0.0

        print(f"Global Mean: {global_mean:.4f}")
        print(f"Global Std:  {global_std:.4f}")
    else:
        print("No pixel data available.")

    print("-" * 40)

    # -------------------------------------------------------------------------
    # SECTION 4: FEATURE/SIGNAL RELATIONSHIPS
    # -------------------------------------------------------------------------
    print("SECTION 4: FEATURE/SIGNAL RELATIONSHIPS")
    print("Subsection: Meta-Feature vs Target Correlations")
    print(
        "Analyzing Point-Biserial Correlation between Image Meta-Features and Binary Targets."
    )

    print(f"{'Label':<30} | {'Width':<10} | {'Height':<10} | {'Aspect Ratio':<12}")
    print("-" * 75)

    meta_features = ["width", "height", "aspect_ratio"]

    for label in label_cols:
        # Check if label exists in meta_df (it should) and has variance
        if label not in meta_df.columns or meta_df[label].nunique() < 2:
            continue

        corrs = []
        for feat in meta_features:
            # Correlation
            c = meta_df[feat].corr(meta_df[label])
            corrs.append(c)

        print(f"{label:<30} | {corrs[0]:.4f}     | {corrs[1]:.4f}     | {corrs[2]:.4f}")

    print("-" * 40)
    print("EDA Complete.")


if __name__ == "__main__":
    main()
