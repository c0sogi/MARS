import os
import sys
import numpy as np
import pandas as pd
import json
import rasterio
import warnings
import random
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# --- Configuration & Setup ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def format_float(val):
    return f"{val:.4f}"


# --- Helper Functions ---


def calculate_rle_area(rle_string):
    """Calculates the total number of masked pixels from an RLE string."""
    if pd.isna(rle_string) or rle_string == "":
        return 0
    # RLE format: start length start length ...
    # We only need the lengths (every second value)
    try:
        rle_numbers = [int(x) for x in rle_string.split()]
        # Sum every second element starting from index 1
        return sum(rle_numbers[1::2])
    except:
        return 0


def analyze_json_objects(json_path):
    """Parses JSON to count objects and get their areas (approximated by polygon points)."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # data is a list of features
        num_objects = len(data)

        # Calculate rough area or just return counts.
        # Exact polygon area requires shapely, but we can just count points as a proxy for complexity
        # or just return the count of objects for the report.
        return num_objects
    except Exception:
        return 0


def get_image_stats(image_path):
    """Reads image metadata and computes stats on a downsampled version."""
    try:
        with rasterio.open(image_path) as src:
            width = src.width
            height = src.height
            count = src.count

            # Read a thumbnail (approx 512x512 max dimension) for stats
            scale = 512 / max(width, height)
            out_shape = (count, int(height * scale), int(width * scale))

            # Resampling.Bilinear is good, but Nearest is faster. Using default (Nearest) for speed.
            img_data = src.read(out_shape=out_shape)

            mean_val = np.mean(img_data)
            std_val = np.std(img_data)

            return {
                "width": width,
                "height": height,
                "channels": count,
                "mean": mean_val,
                "std": std_val,
                "aspect_ratio": width / height,
            }
    except Exception as e:
        return None


# --- Main Analysis Logic ---


def main():
    # 1. Load Data
    metadata_path = "./metadata/train_metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)

    # 1. Data Integrity
    print_section("1. Data Integrity")
    print(f"Analysis performed on Training Set only.")
    print(f"Number of samples: {len(df)}")
    print(f"Unique patients: {df['patient_number'].nunique()}")

    # 2. Target Variable Analysis
    print_section("2. Target Variable Analysis")

    # Calculate mask areas and coverage
    df["mask_pixels"] = df["encoding"].apply(calculate_rle_area)
    df["total_pixels"] = df["width_pixels"] * df["height_pixels"]
    df["mask_coverage"] = df["mask_pixels"] / df["total_pixels"]

    # Get object counts from JSON
    df["object_count"] = df["json_path"].apply(analyze_json_objects)

    print("Target: Glomerulus Segmentation Mask")
    print(f"Global Mean Mask Coverage: {format_float(df['mask_coverage'].mean())}")
    print(f"Global Std Mask Coverage: {format_float(df['mask_coverage'].std())}")
    print(f"Min Coverage: {format_float(df['mask_coverage'].min())}")
    print(f"Max Coverage: {format_float(df['mask_coverage'].max())}")
    print(f"Skewness of Coverage: {format_float(df['mask_coverage'].skew())}")
    print(f"Kurtosis of Coverage: {format_float(df['mask_coverage'].kurtosis())}")

    print("\nObject Counts (Glomeruli per image):")
    print(f"Mean Objects per Image: {format_float(df['object_count'].mean())}")
    print(f"Min Objects: {df['object_count'].min()}")
    print(f"Max Objects: {df['object_count'].max()}")

    # 3. Input Data Analysis (Image)
    print_section("3. Input Data Analysis (Image)")

    # We gather stats from the actual images
    img_stats_list = []
    for path in df["image_path"]:
        stats = get_image_stats(path)
        if stats:
            img_stats_list.append(stats)

    if img_stats_list:
        img_df = pd.DataFrame(img_stats_list)

        print("Dimensions:")
        print(
            f"Width Mean: {format_float(img_df['width'].mean())} (Min: {img_df['width'].min()}, Max: {img_df['width'].max()})"
        )
        print(
            f"Height Mean: {format_float(img_df['height'].mean())} (Min: {img_df['height'].min()}, Max: {img_df['height'].max()})"
        )
        print(f"Aspect Ratio Mean: {format_float(img_df['aspect_ratio'].mean())}")

        print("\nChannels:")
        print(f"Unique Channel Counts: {img_df['channels'].unique().tolist()}")

        print("\nPixel Intensity Statistics (Estimated from thumbnails):")
        print(f"Global Pixel Mean: {format_float(img_df['mean'].mean())}")
        print(f"Global Pixel Std: {format_float(img_df['std'].mean())}")
    else:
        print("Could not extract image statistics.")

    # 4. Input Data Analysis (Tabular Metadata)
    print_section("3. Input Data Analysis (Tabular Metadata)")

    # Filter for relevant metadata columns (excluding paths and IDs)
    meta_cols = [
        "age",
        "weight_kilograms",
        "bmi_kg/m^2",
        "laterality",
        "percent_cortex",
        "percent_medulla",
        "race",
        "sex",
    ]
    # Ensure columns exist
    meta_cols = [c for c in meta_cols if c in df.columns]

    # Numerical Analysis
    num_cols = df[meta_cols].select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print("Numerical Metadata:")
        for col in num_cols:
            col_data = df[col].dropna()
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            outliers = (
                (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
            ).sum()

            print(
                f"  {col}: Mean={format_float(col_data.mean())}, Std={format_float(col_data.std())}, "
                f"Min={format_float(col_data.min())}, Max={format_float(col_data.max())}, "
                f"Outliers={outliers}, Missing={df[col].isna().sum()}"
            )

    # Categorical Analysis
    cat_cols = df[meta_cols].select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print("\nCategorical Metadata:")
        for col in cat_cols:
            counts = df[col].value_counts()
            print(
                f"  {col}: {len(counts)} categories. Top: {counts.index[0]} ({counts.iloc[0]})"
            )
            if df[col].isna().sum() > 0:
                print(f"    Missing values: {df[col].isna().sum()}")

    # 5. Feature/Signal Relationships
    print_section("4. Feature/Signal Relationships")

    # Correlation Matrix (Numerical Metadata vs Target)
    # Target features: mask_coverage, object_count
    analysis_df = df[num_cols].copy()
    analysis_df["target_mask_coverage"] = df["mask_coverage"]
    analysis_df["target_object_count"] = df["object_count"]

    corr_matrix = analysis_df.corr(method="pearson")

    print("Correlations with Target (Mask Coverage):")
    target_corr = corr_matrix["target_mask_coverage"].drop(
        ["target_mask_coverage", "target_object_count"]
    )
    for idx, val in target_corr.items():
        print(f"  {idx}: {format_float(val)}")

    print("\nCorrelations with Target (Object Count):")
    obj_corr = corr_matrix["target_object_count"].drop(
        ["target_mask_coverage", "target_object_count"]
    )
    for idx, val in obj_corr.items():
        print(f"  {idx}: {format_float(val)}")

    # Feature Importance (Random Forest)
    # Prepare data
    rf_data = df[meta_cols].copy()

    # Handle missing values for RF
    for col in rf_data.columns:
        if rf_data[col].dtype == "object":
            rf_data[col] = rf_data[col].fillna("Missing")
            le = LabelEncoder()
            rf_data[col] = le.fit_transform(rf_data[col].astype(str))
        else:
            rf_data[col] = rf_data[col].fillna(rf_data[col].mean())

    X = rf_data
    y = df["mask_coverage"]

    if len(X) > 1:
        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        print("\nTop 5 Metadata Features predicting Mask Coverage (RF Importance):")
        for idx, val in importances.head(5).items():
            print(f"  {idx}: {format_float(val)}")
    else:
        print("\nNot enough data for Random Forest analysis.")

    # Redundancy Check
    print("\nRedundancy Check (Correlation > 0.90):")
    high_corr_pairs = []
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for col in upper.columns:
        for row in upper.index:
            if abs(upper.loc[row, col]) > 0.90:
                high_corr_pairs.append((row, col, upper.loc[row, col]))

    if high_corr_pairs:
        for p in high_corr_pairs:
            print(f"  {p[0]} & {p[1]}: {format_float(p[2])}")
    else:
        print("  No highly collinear pairs found.")


if __name__ == "__main__":
    main()
