import os
import numpy as np
import pandas as pd
import cv2
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import warnings


# ==========================================
# 1. Setup and Configuration
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Configuration
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train_meta.csv"
    SEED = 42

    # Suppress warnings
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Construct full paths
    # The metadata file_path is relative to input directory (e.g., "train/id.jpg")
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    print("==========================================")
    print("       EXPLORATORY DATA ANALYSIS          ")
    print("==========================================")

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("\nTARGET VARIABLE ANALYSIS")
    target_col = "Pawpularity"
    targets = df[target_col]

    mean_val = targets.mean()
    std_val = targets.std()
    min_val = targets.min()
    max_val = targets.max()
    skew_val = stats.skew(targets)
    kurt_val = stats.kurtosis(targets)

    print(f"Target: {target_col}")
    print(f"Count: {len(targets)}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")
    print(
        f"Skewness: {skew_val:.4f} (Positively skewed)"
        if skew_val > 0
        else f"Skewness: {skew_val:.4f} (Negatively skewed)"
    )
    print(f"Kurtosis: {kurt_val:.4f}")

    # ==========================================
    # 3. Input Data Analysis (Tabular)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TABULAR)")

    # Identify feature columns (excluding Id, target, file_path, full_path)
    exclude_cols = {"Id", "Pawpularity", "file_path", "full_path"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Missing Values
    missing_counts = df[feature_cols].isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Total Missing Values in Features: {total_missing}")
    if total_missing > 0:
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0])

    # Numerical/Categorical Stats
    # In this dataset, features are binary (0/1). We treat them as categorical but report frequencies.
    print(f"Number of Binary Features: {len(feature_cols)}")
    print("Feature Frequencies (Percentage of '1's):")
    for col in feature_cols:
        freq = df[col].mean() * 100
        print(f"  {col}: {freq:.2f}%")

    # ==========================================
    # 3. Input Data Analysis (Image)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # We will iterate to get dimensions.
    # For pixel stats, we sample to keep runtime low (e.g., 1000 images).

    widths = []
    heights = []
    aspect_ratios = []
    channels_dist = {}

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    sample_size_pixel_stats = 1000
    # Use a fixed sample for pixel stats
    sample_indices = np.random.choice(
        len(df), min(len(df), sample_size_pixel_stats), replace=False
    )
    sample_paths = set(df.iloc[sample_indices]["full_path"])

    # Iterate through all for dimensions, calculate pixels on sample
    # To optimize, we verify file existence first

    valid_image_count = 0

    for idx, row in df.iterrows():
        path = row["full_path"]
        if not os.path.exists(path):
            continue

        # Read image header only for dimensions if possible, but cv2 reads full.
        # Since we have time, reading full image is okay for 7k images.
        img = cv2.imread(path)

        if img is None:
            continue

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

        # Channels (OpenCV loads as BGR, shape is H,W,C)
        c_key = f"{c} Channels"
        channels_dist[c_key] = channels_dist.get(c_key, 0) + 1

        # Pixel stats (only for sample)
        if path in sample_paths:
            # Convert BGR to RGB for reporting
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to 0-1 for stats calculation
            img_norm = img_rgb / 255.0
            pixel_sum += np.sum(img_norm, axis=(0, 1))
            pixel_sq_sum += np.sum(img_norm**2, axis=(0, 1))
            pixel_count += h * w

        valid_image_count += 1

    # Dimension Stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Analyzed {valid_image_count} images.")
    print("Dimensions (Width):")
    print(
        f"  Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print("Dimensions (Height):")
    print(
        f"  Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print("Aspect Ratios (W/H):")
    print(f"  Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")

    print("Channel Distribution:")
    for k, v in channels_dist.items():
        print(f"  {k}: {v}")

    # Pixel Stats Calculation
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        # E[X^2] - (E[X])^2
        global_var = (pixel_sq_sum / pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)

        print(
            f"Pixel Stats (RGB, 0-1 scale) based on sample of {len(sample_paths)} images:"
        )
        print(
            f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
        )
        print(
            f"  Std:  R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
        )

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # 4.1 Structured Relationships (Correlation)
    print("Top Correlations with Target (Pearson):")
    # Compute correlation between binary features and target
    correlations = df[feature_cols].corrwith(df[target_col])
    sorted_corr = correlations.abs().sort_values(ascending=False)

    for feat in sorted_corr.index[:5]:
        corr_val = correlations[feat]
        print(f"  {feat}: {corr_val:.4f}")

    # 4.2 Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest Regressor):")
    X = df[feature_cols]
    y = df[target_col]

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(5, len(feature_cols))):
        feat_name = feature_cols[indices[i]]
        imp_val = importances[indices[i]]
        print(f"  {feat_name}: {imp_val:.4f}")

    # 4.3 Redundancy (Collinear Pairs)
    print("\nRedundancy Check (Correlation > 0.90):")
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if len(to_drop) == 0:
        print("  No highly collinear pairs found.")
    else:
        for col in to_drop:
            # Find the feature it correlates with
            correlated_feats = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} is highly correlated with: {correlated_feats}")

    # 4.4 Unstructured Relationships (Meta-features)
    # Add meta features to dataframe temporarily for correlation
    # We need to align the lists with the dataframe.
    # Note: The loop above iterated df rows in order, so lists are aligned if no skips.
    # We skipped missing images, but metadata generation ensures existence.
    # To be safe, we reconstruct a mini-df.

    # Assuming all images were found (verified in metadata step), lists align with df.
    if len(widths) == len(df):
        meta_df = pd.DataFrame(
            {
                "ImageWidth": widths,
                "ImageHeight": heights,
                "AspectRatio": aspect_ratios,
                "Target": df[target_col].values,
            }
        )

        print("\nMeta-Feature Correlations with Target:")
        meta_corr = meta_df.corr()["Target"].drop("Target")
        for idx, val in meta_corr.items():
            print(f"  {idx}: {val:.4f}")
    else:
        print(
            "\nSkipping Meta-Feature correlation due to image count mismatch (some images missing)."
        )


if __name__ == "__main__":
    main()
