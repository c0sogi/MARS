import os
import random
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_rle_area(rle_string):
    """Calculates the total number of masked pixels from RLE string."""
    if pd.isna(rle_string):
        return 0
    # RLE format: start length start length ...
    # We only need lengths (every second item starting from index 1)
    s = rle_string.split()
    lengths = s[1::2]
    return sum(map(int, lengths))


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    # Derive FTU Fraction
    df["mask_area"] = df["encoding"].apply(parse_rle_area)
    df["image_area"] = df["width_pixels"] * df["height_pixels"]
    df["ftu_fraction"] = df["mask_area"] / df["image_area"]

    target = df["ftu_fraction"]
    print("Target: FTU Area Fraction (Derived from Segmentation Mask)")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std:  {target.std():.4f}")
    print(f"Min:  {target.min():.4f}")
    print(f"Max:  {target.max():.4f}")
    print(f"Skewness: {skew(target):.4f}")
    print(f"Kurtosis: {kurtosis(target):.4f}")
    print("-" * 30)
    return df


def analyze_images(df, input_root):
    print("IMAGE DATA ANALYSIS")
    widths = df["width_pixels"]
    heights = df["height_pixels"]
    aspect_ratios = widths / heights

    print(
        f"Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}"
    )

    # Channel and Pixel Stats Analysis (Sampling)
    # We sample a few images and read a low-res overview to save time/memory
    sample_size = min(3, len(df))
    sample_indices = np.random.choice(df.index, sample_size, replace=False)

    channel_counts = []
    means = []
    stds = []

    print(f"Sampling {sample_size} images for pixel stats...")

    for idx in sample_indices:
        rel_path = df.loc[idx, "image_path"]
        full_path = os.path.join(input_root, rel_path)

        try:
            with rasterio.open(full_path) as src:
                channel_counts.append(src.count)
                # Read a decimated version (e.g., 1/32 scale) to compute stats efficiently
                # Calculate new shape
                h_new = max(1, src.height // 32)
                w_new = max(1, src.width // 32)
                out_shape = (src.count, h_new, w_new)

                data = src.read(out_shape=out_shape)

                means.append(np.mean(data))
                stds.append(np.std(data))
        except Exception as e:
            pass

    if channel_counts:
        print(f"Channel Counts (Sampled): {list(set(channel_counts))}")
        print(f"Pixel Global Mean (Sampled, Approx): {np.mean(means):.4f}")
        print(f"Pixel Global Std (Sampled, Approx):  {np.mean(stds):.4f}")
    else:
        print("Could not sample images for pixel stats.")
    print("-" * 30)


def analyze_tabular(df):
    print("TABULAR DATA ANALYSIS")

    # Numerical Columns
    num_cols = [
        "age",
        "weight_kilograms",
        "height_centimeters",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
    ]
    # Filter to existing columns
    num_cols = [c for c in num_cols if c in df.columns]

    for col in num_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        print(f"Column: {col}")
        print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
        print(f"  Min:  {series.min():.4f}, Max: {series.max():.4f}")
        print(f"  Outliers (IQR method): {outliers}")

    # Categorical Columns
    cat_cols = ["race", "ethnicity", "sex", "laterality"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"Column: {col}")
        print(f"  Cardinality: {unique_vals}")
        # Check rare categories (< 1%)
        counts = df[col].value_counts(normalize=True)
        rare = counts[counts < 0.01].index.tolist()
        if rare:
            print(f"  Rare categories (<1%): {rare}")
        else:
            print(f"  Rare categories (<1%): None")

    # Missing Values
    print("Missing Values:")
    missing = df.isnull().sum()
    missing_found = False
    for col, val in missing.items():
        if val > 0:
            print(f"  {col}: {val} ({val/len(df)*100:.2f}%)")
            missing_found = True
    if not missing_found:
        print("  None")
    print("-" * 30)


def analyze_relationships(df, target_col):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Prepare data for correlation and RF
    num_cols = [
        "age",
        "weight_kilograms",
        "height_centimeters",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    cat_cols = ["race", "ethnicity", "sex", "laterality"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # Create a working copy
    work_df = df[num_cols + cat_cols + [target_col]].copy()

    # Impute numerical
    imputer = SimpleImputer(strategy="mean")
    if num_cols:
        work_df[num_cols] = imputer.fit_transform(work_df[num_cols])

    # Encode categorical
    le = LabelEncoder()
    for col in cat_cols:
        # Fill nan with 'missing' before encoding
        work_df[col] = work_df[col].fillna("missing")
        work_df[col] = le.fit_transform(work_df[col].astype(str))

    # Correlation
    corr_matrix = work_df.corr()
    target_corr = corr_matrix[target_col].drop(target_col)

    print("Correlations with Target (FTU Fraction):")
    print(target_corr.sort_values(ascending=False))

    # Redundancy
    print("\nHighly Correlated Feature Pairs (>0.90):")
    cols = work_df.columns.drop(target_col)
    found_redundancy = False
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            val = corr_matrix.loc[c1, c2]
            if abs(val) > 0.9:
                print(f"  {c1} - {c2}: {val:.4f}")
                found_redundancy = True
    if not found_redundancy:
        print("  None")

    # Feature Importance (Random Forest)
    X = work_df.drop(columns=[target_col])
    y = work_df[target_col]

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("\nTop 5 Features (Random Forest Importance):")
    print(importances.head(5))
    print("-" * 30)


def main():
    set_seed()

    # 1. Load Data
    metadata_path = "./metadata/train.csv"
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)

    # 2. Target Variable Analysis
    df = analyze_target(df)

    # 3. Input Data Analysis (Image)
    analyze_images(df, "./input")

    # 4. Input Data Analysis (Tabular)
    analyze_tabular(df)

    # 5. Feature Relationships
    analyze_relationships(df, "ftu_fraction")


if __name__ == "__main__":
    main()
