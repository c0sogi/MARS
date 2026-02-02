import os
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime

# ==========================================
# Configuration & Constants
# ==========================================
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train_metadata.csv"
SAMPLE_SIZE = 500  # Number of images to sample for heavy IO operations
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def get_file_path(rel_path):
    return os.path.join(INPUT_DIR, rel_path)


def perform_eda():
    print("Starting Exploratory Data Analysis...")
    start_time = time.time()

    # ==========================================
    # 1. Load Metadata
    # ==========================================
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)
    total_records = len(df)

    # Sample the dataframe for IO-heavy tasks
    sample_n = min(total_records, SAMPLE_SIZE)
    df_sample = df.sample(n=sample_n, random_state=SEED).copy()

    # ==========================================
    # 2. Target Variable Analysis (Masks)
    # ==========================================
    print(f"Analyzing Target Variable (Masks) on {sample_n} samples...")

    total_pixels = 0
    contrail_pixels = 0
    contrail_fractions = []
    empty_masks = 0

    # To correlate later
    df_sample["contrail_fraction"] = 0.0

    for idx, row in df_sample.iterrows():
        mask_path = get_file_path(row["human_pixel_masks"])
        try:
            # Mask shape: H x W x 1
            mask = np.load(mask_path)

            n_pixels = mask.size
            n_pos = np.sum(mask)

            total_pixels += n_pixels
            contrail_pixels += n_pos

            fraction = n_pos / n_pixels
            contrail_fractions.append(fraction)
            df_sample.loc[idx, "contrail_fraction"] = fraction

            if n_pos == 0:
                empty_masks += 1

        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}")

    global_pos_ratio = contrail_pixels / total_pixels if total_pixels > 0 else 0

    # ==========================================
    # 3. Input Data Analysis (Images)
    # ==========================================
    print(f"Analyzing Input Data (Satellite Bands) on {sample_n} samples...")

    # Stats accumulators per band
    # Bands are 08 to 16
    bands_indices = [f"{i:02d}" for i in range(8, 17)]
    band_stats = {
        b: {
            "sum": 0.0,
            "sq_sum": 0.0,
            "count": 0,
            "min": float("inf"),
            "max": float("-inf"),
        }
        for b in bands_indices
    }

    img_heights = []
    img_widths = []
    img_times = []  # Temporal depth

    # We will compute stats based on the sampled subset
    for idx, row in df_sample.iterrows():
        # Check dimensions using just one band (Band 11 is commonly used for Ash/Contrails)
        b11_path = get_file_path(row["band_11"])
        try:
            # Shape: H x W x T
            img = np.load(b11_path)
            h, w, t = img.shape
            img_heights.append(h)
            img_widths.append(w)
            img_times.append(t)
        except Exception as e:
            continue

        # Accumulate pixel stats for all bands
        # To save time, we calculate mean/std over the flattened array of the sample
        for b in bands_indices:
            col_name = f"band_{b}"
            path = get_file_path(row[col_name])
            try:
                data = np.load(path)
                # Flatten for stats
                flat = data.flatten()

                band_stats[b]["sum"] += np.sum(flat)
                band_stats[b]["sq_sum"] += np.sum(flat**2)
                band_stats[b]["count"] += flat.size
                band_stats[b]["min"] = min(band_stats[b]["min"], np.min(flat))
                band_stats[b]["max"] = max(band_stats[b]["max"], np.max(flat))
            except:
                pass

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    # 4.1 Metadata Analysis (Tabular)
    # Extract features from timestamp
    df_sample["datetime"] = pd.to_datetime(df_sample["timestamp"], unit="s")
    df_sample["hour"] = df_sample["datetime"].dt.hour
    df_sample["month"] = df_sample["datetime"].dt.month

    # Correlations
    # We look at relationships between metadata (Time, Location) and Target (Contrail Fraction)
    meta_cols = ["timestamp", "row_min", "col_min", "hour", "month"]
    correlations = {}
    for col in meta_cols:
        if col in df_sample.columns:
            corr = df_sample[col].corr(df_sample["contrail_fraction"])
            correlations[col] = corr

    # ==========================================
    # 5. Report Generation
    # ==========================================
    print("\n" + "=" * 40)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 40)

    # --- Data Integrity ---
    print("\nDATA INTEGRITY")
    print(f"Source: Training Set Only")
    print(f"Total Records Available: {total_records}")
    print(f"Records Analyzed: {sample_n}")

    # --- Target Variable Analysis ---
    print("\nTARGET VARIABLE ANALYSIS (Segmentation Masks)")
    print(
        f"Global Contrail Pixel Ratio: {global_pos_ratio:.6f} ({global_pos_ratio*100:.4f}%)"
    )
    print(
        f"Class Balance (Background:Contrail): {1/global_pos_ratio:.2f}:1"
        if global_pos_ratio > 0
        else "N/A"
    )
    print(f"Empty Masks in Sample: {empty_masks} ({empty_masks/sample_n*100:.2f}%)")

    # Distribution of coverage
    fractions = np.array(contrail_fractions)
    print(f"Contrail Coverage per Image - Mean: {np.mean(fractions):.6f}")
    print(f"Contrail Coverage per Image - Max:  {np.max(fractions):.6f}")
    print(f"Contrail Coverage per Image - Std:  {np.std(fractions):.6f}")

    # --- Input Data Analysis ---
    print("\nINPUT DATA ANALYSIS (Satellite Imagery)")

    # Dimensions
    avg_h = np.mean(img_heights) if img_heights else 0
    avg_w = np.mean(img_widths) if img_widths else 0
    avg_t = np.mean(img_times) if img_times else 0
    print(f"Image Dimensions (H x W): {avg_h:.0f} x {avg_w:.0f}")
    print(f"Temporal Depth (Frames per sample): {avg_t:.0f}")
    if len(set(img_heights)) > 1 or len(set(img_widths)) > 1:
        print("WARNING: Inconsistent image dimensions detected.")
    else:
        print("Dimensions are consistent across samples.")

    # Channel Stats
    print("\nChannel Statistics (Brightness Temperature K):")
    print(f"{'Band':<6} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 56)
    for b in bands_indices:
        stats = band_stats[b]
        if stats["count"] > 0:
            mean_val = stats["sum"] / stats["count"]
            variance = (stats["sq_sum"] / stats["count"]) - (mean_val**2)
            std_val = np.sqrt(variance) if variance > 0 else 0
            print(
                f"{b:<6} | {mean_val:<10.4f} | {std_val:<10.4f} | {stats['min']:<10.4f} | {stats['max']:<10.4f}"
            )

    # --- Feature Relationships ---
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("Correlation with Target (Contrail Fraction):")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"  {feat:<15}: {corr:.4f}")

    print("\nMetadata Summary:")
    print(
        f"  Date Range: {df_sample['datetime'].min()} to {df_sample['datetime'].max()}"
    )
    print(f"  Unique Record IDs: {df_sample['record_id'].nunique()}")

    # --- Missing Values ---
    print("\nMISSING VALUES (Metadata)")
    missing = df_sample.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found in metadata columns.")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df_sample)*100:.2f}%)")

    print("\n" + "=" * 40)
    print(f"EDA Completed in {time.time() - start_time:.2f} seconds.")
    print("=" * 40)


if __name__ == "__main__":
    perform_eda()
