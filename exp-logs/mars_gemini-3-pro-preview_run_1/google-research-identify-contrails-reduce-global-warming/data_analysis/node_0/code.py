import os
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def load_npy_file(relative_path):
    """Loads a .npy file from the input directory."""
    full_path = os.path.join(INPUT_DIR, relative_path)
    return np.load(full_path)


def get_file_stats(file_path):
    """Returns shape and dtype of a numpy file."""
    arr = load_npy_file(file_path)
    return arr.shape, arr.dtype


# ------------------------------------------------------------------------------
# Analysis Modules
# ------------------------------------------------------------------------------


def analyze_target(df, sample_size=500):
    """
    Analyzes the distribution of the target variable (masks).
    """
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Sample records
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    total_pixels = 0
    contrail_pixels = 0
    empty_masks_count = 0

    # Store coverage for meta-analysis later
    coverages = []

    for _, row in sample_df.iterrows():
        mask_path = row["human_pixel_masks"]
        try:
            mask = load_npy_file(mask_path)

            # Mask shape is H x W x 1
            n_pix = mask.size
            n_pos = np.sum(mask)

            total_pixels += n_pix
            contrail_pixels += n_pos

            if n_pos == 0:
                empty_masks_count += 1

            coverages.append(n_pos / n_pix)

        except Exception as e:
            coverages.append(np.nan)
            continue

    # Statistics
    pos_ratio = contrail_pixels / total_pixels if total_pixels > 0 else 0
    neg_ratio = 1 - pos_ratio
    empty_ratio = empty_masks_count / len(sample_df)

    print(f"Analysis based on {len(sample_df)} sampled masks.")
    print(f"Pixel Class Balance (Background): {neg_ratio:.4f}")
    print(f"Pixel Class Balance (Contrail):   {pos_ratio:.4f}")
    print(f"Ratio of Empty Masks (No Contrails): {empty_ratio:.4f}")

    # Add coverage to the dataframe for later correlation analysis
    sample_df["contrail_coverage"] = coverages
    return sample_df


def analyze_images(df, sample_size=100):
    """
    Analyzes the image data (bands).
    Calculates dimensions and pixel statistics (mean/std) per band.
    """
    print("\nIMAGE DATA ANALYSIS")
    print("-" * 30)

    # Use a smaller sample for image stats as reading 9 bands per record is heavy
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=SEED)
    else:
        sample_df = df

    print(f"Analysis based on {len(sample_df)} sampled records (all 9 bands).")

    # Check dimensions on the first record
    first_row = sample_df.iloc[0]
    band_08 = load_npy_file(first_row["band_08"])
    H, W, T = band_08.shape
    print(f"Image Dimensions (H, W): ({H}, {W})")
    print(f"Temporal Depth (T): {T}")
    print(f"Total Channels (9 Bands x {T} Timesteps): {9 * T}")

    # Initialize accumulators for Mean/Std calculation
    # We will calculate stats per Band ID (aggregating over H, W, T)
    band_ids = [f"band_{i:02d}" for i in range(8, 17)]
    stats = {bid: {"sum": 0.0, "sq_sum": 0.0, "count": 0} for bid in band_ids}

    for _, row in sample_df.iterrows():
        for bid in band_ids:
            try:
                path = row[bid]
                arr = load_npy_file(path).astype(np.float64)

                stats[bid]["sum"] += np.sum(arr)
                stats[bid]["sq_sum"] += np.sum(arr**2)
                stats[bid]["count"] += arr.size
            except Exception:
                continue

    print("\nPixel Statistics (Global Mean & Std per Band):")
    print(f"{'Band ID':<10} | {'Mean':<12} | {'Std Dev':<12}")
    print("-" * 40)

    for bid in band_ids:
        count = stats[bid]["count"]
        if count > 0:
            mean = stats[bid]["sum"] / count
            variance = (stats[bid]["sq_sum"] / count) - (mean**2)
            std = np.sqrt(variance) if variance > 0 else 0
            print(f"{bid:<10} | {mean:.4f}       | {std:.4f}")
        else:
            print(f"{bid:<10} | N/A          | N/A")


def analyze_meta_relationships(df):
    """
    Analyzes relationships between metadata and the target.
    Requires 'contrail_coverage' to be present in df (computed in analyze_target).
    """
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Filter out rows where coverage calculation failed
    valid_df = df.dropna(subset=["contrail_coverage"]).copy()

    if len(valid_df) < 2:
        print("Insufficient data for correlation analysis.")
        return

    # 1. Temporal Analysis
    # Convert timestamp to hour of day
    # Timestamps are Unix seconds.
    valid_df["datetime"] = pd.to_datetime(valid_df["timestamp"], unit="s")
    valid_df["hour_of_day"] = valid_df["datetime"].dt.hour

    # 2. Spatial Analysis
    # Use col_min and row_min as proxies for X/Y coordinates (Projected)
    # We assume 'col_min' ~ Longitude proxy, 'row_min' ~ Latitude proxy in the projection

    # Calculate Correlations
    # We correlate metadata with 'contrail_coverage'
    features_to_correlate = ["timestamp", "hour_of_day", "col_min", "row_min"]
    correlations = {}

    for feat in features_to_correlate:
        if feat in valid_df.columns:
            corr = valid_df[feat].corr(valid_df["contrail_coverage"])
            correlations[feat] = corr

    print("Correlation with Target (Contrail Coverage Fraction):")
    print(f"{'Feature':<20} | {'Pearson Corr':<12}")
    print("-" * 35)
    for feat, corr in correlations.items():
        print(f"{feat:<20} | {corr:.4f}")

    # Top feature importance (Lightweight Random Forest)
    # Using sklearn as it is standard and available
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer

        feature_cols = [
            "timestamp",
            "hour_of_day",
            "col_min",
            "row_min",
            "col_size",
            "row_size",
        ]
        # Select only available numeric columns
        X = valid_df[[c for c in feature_cols if c in valid_df.columns]]
        y = valid_df["contrail_coverage"]

        # Handle NaNs if any
        imputer = SimpleImputer()
        X_imputed = imputer.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=SEED)
        rf.fit(X_imputed, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop Meta-Feature Importance (Random Forest Regressor):")
        for i in range(min(3, len(indices))):
            idx = indices[i]
            print(f"{i+1}. {X.columns[idx]} (Imp: {importances[idx]:.4f})")

    except Exception as e:
        print(f"Could not calculate feature importance: {e}")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------


def main():
    # 1. Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # 2. Target Analysis (on a sample)
    # We use a sample of 1000 for mask analysis to get a good estimate of class balance
    df_with_target = analyze_target(df_train, sample_size=1000)

    # 3. Image Analysis (on a smaller sample)
    # We use a sample of 100 for heavy image I/O
    analyze_images(df_train, sample_size=100)

    # 4. Feature Relationships
    # Uses the dataframe from step 2 which has 'contrail_coverage'
    analyze_meta_relationships(df_with_target)


if __name__ == "__main__":
    main()
