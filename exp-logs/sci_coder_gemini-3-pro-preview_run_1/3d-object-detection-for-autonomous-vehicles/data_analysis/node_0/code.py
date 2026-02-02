import os
import json
import ast
import random
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis
from collections import Counter

# Constants
METADATA_PATH = "./metadata/train.csv"
SEED = 42
IMAGE_SAMPLE_SIZE = 500  # Number of images to sample for pixel stats to save time


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch is not strictly needed for this EDA script, but setting if environment has it
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data():
    """Loads metadata and parses complex columns."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    # Parse annotations (JSON) and image_paths (Stringified List)
    # We use a try-except block or apply directly if data quality is guaranteed by metadata script
    df["annotations"] = df["annotations"].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )
    df["image_paths"] = df["image_paths"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )

    return df


def analyze_target_variables(df):
    """
    Analyzes the annotations (targets).
    Targets include:
    - Classification: class_name
    - Regression: center_x, center_y, center_z, width, length, height, yaw
    """
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Flatten annotations into a single DataFrame for analysis
    all_anns = []
    for _, row in df.iterrows():
        for ann in row["annotations"]:
            all_anns.append(ann)

    if not all_anns:
        print("No annotations found in training set.")
        return None

    ann_df = pd.DataFrame(all_anns)

    # 1. Classification Analysis
    print("--- Classification (Object Classes) ---")
    class_counts = ann_df["class_name"].value_counts()
    total_objects = len(ann_df)
    print(f"Total Annotated Objects: {total_objects}")
    print(f"Number of Classes: {len(class_counts)}")
    print("Class Balance Ratios:")
    for cls, count in class_counts.items():
        ratio = count / total_objects
        print(f"  {cls}: {ratio:.4f} ({count})")

    # 2. Regression Analysis
    print("\n--- Regression (Bounding Box Parameters) ---")
    numeric_cols = [
        "center_x",
        "center_y",
        "center_z",
        "width",
        "length",
        "height",
        "yaw",
    ]

    for col in numeric_cols:
        data = ann_df[col]
        s = skew(data)
        k = kurtosis(data)
        print(f"Feature: {col}")
        print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
        print(f"  Min:  {data.min():.4f}, Max: {data.max():.4f}")
        print(f"  Skewness: {s:.4f}, Kurtosis: {k:.4f}")

    return ann_df


def analyze_image_data(df):
    """
    Analyzes image data properties.
    """
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Collect all image paths
    all_image_paths = []
    for paths in df["image_paths"]:
        all_image_paths.extend(paths)

    total_images = len(all_image_paths)
    print(f"Total Images Available: {total_images}")

    # Sample for analysis
    sample_paths = np.random.choice(
        all_image_paths, min(IMAGE_SAMPLE_SIZE, total_images), replace=False
    )
    print(f"Analyzing sample of {len(sample_paths)} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # For pixel stats (using Welford's algorithm or simple accumulation for mean/std)
    # Simple accumulation is sufficient for EDA on a sample
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_samples = 0

    for p in sample_paths:
        if not os.path.exists(p):
            continue

        # Read image
        img = cv2.imread(p)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)

        # Pixel stats
        # Normalize to 0-1 for standard reporting or keep 0-255. Usually 0-255 is clearer for raw EDA.
        # Let's report 0-255 stats.
        pixels = img.reshape(-1, 3)
        pixel_sum += pixels.sum(axis=0)
        pixel_sq_sum += (pixels**2).sum(axis=0)
        pixel_count += pixels.shape[0]

        valid_samples += 1

    if valid_samples == 0:
        print("No valid images found in sample.")
        return

    # Dimensions
    w_arr = np.array(widths)
    h_arr = np.array(heights)
    ar_arr = np.array(aspect_ratios)

    print("--- Dimensions ---")
    print(
        f"Width:  Mean={w_arr.mean():.4f}, Std={w_arr.std():.4f}, Min={w_arr.min()}, Max={w_arr.max()}"
    )
    print(
        f"Height: Mean={h_arr.mean():.4f}, Std={h_arr.std():.4f}, Min={h_arr.min()}, Max={h_arr.max()}"
    )
    print(f"Aspect Ratio: Mean={ar_arr.mean():.4f}, Std={ar_arr.std():.4f}")

    # Channels
    print("--- Channels ---")
    c_counts = Counter(channels)
    for c, count in c_counts.items():
        print(f"  {c} Channels: {count} images")

    # Pixel Stats
    rgb_mean = pixel_sum / pixel_count
    rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))

    print("--- Pixel Statistics (RGB, 0-255) ---")
    print(f"Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
    print(f"Std:  R={rgb_std[0]:.4f},  G={rgb_std[1]:.4f},  B={rgb_std[2]:.4f}")


def analyze_tabular_metadata(df):
    """
    Analyzes the metadata DataFrame as tabular data.
    """
    print("\n==== INPUT DATA ANALYSIS (TABULAR/METADATA) ====")

    # Missing Values
    print("--- Missing Values ---")
    missing = df.isnull().sum()
    total = len(df)
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({count/total*100:.4f}%)")
    if missing.sum() == 0:
        print("  No missing values in metadata columns.")

    # Object Counts per Sample
    df["num_objects"] = df["annotations"].apply(len)
    print("\n--- Object Counts per Sample ---")
    print(f"Mean Objects/Sample: {df['num_objects'].mean():.4f}")
    print(f"Std Objects/Sample:  {df['num_objects'].std():.4f}")
    print(f"Max Objects/Sample:  {df['num_objects'].max()}")
    print(f"Samples with 0 objects: {(df['num_objects'] == 0).sum()}")


def analyze_relationships(ann_df):
    """
    Analyzes relationships between features.
    """
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    if ann_df is None or len(ann_df) == 0:
        print("No annotation data available for relationship analysis.")
        return

    # 1. Structured Relationships (Correlations between box dimensions)
    print("--- Structured Relationships (Box Dimensions) ---")
    dims = ann_df[["width", "length", "height"]]
    corr = dims.corr(method="pearson")
    print("Correlation Matrix (Pearson):")
    print(corr.round(4))

    # Check for redundancy
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.9:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if high_corr:
        print("\nRedundant Features (>0.9 correlation):")
        for f1, f2, val in high_corr:
            print(f"  {f1} - {f2}: {val:.4f}")
    else:
        print("\nNo highly collinear pairs (>0.9) found in dimensions.")

    # 2. Unstructured/Meta Relationships
    print("\n--- Meta-Feature Relationships ---")

    # Relationship: Class vs Volume
    ann_df["volume"] = ann_df["width"] * ann_df["length"] * ann_df["height"]
    print("Average Volume per Class:")
    vol_by_class = (
        ann_df.groupby("class_name")["volume"].mean().sort_values(ascending=False)
    )
    for cls, mean_vol in vol_by_class.items():
        print(f"  {cls}: {mean_vol:.4f} m^3")

    # Relationship: Spatial Distribution (Center X vs Center Y)
    # This indicates where objects are relative to the ego vehicle
    # We can look at correlation or simple stats
    cx = ann_df["center_x"]
    cy = ann_df["center_y"]
    spatial_corr = cx.corr(cy)
    print(f"\nSpatial Correlation (Center X vs Center Y): {spatial_corr:.4f}")

    # Distance from origin (assuming ego is near 0,0, though in global coords this might differ.
    # Usually in these datasets coords are global. Let's check range.)
    # If coords are huge, they are global map coords. If small, they are ego-relative.
    # Based on standard datasets (NuScenes/Lyft), these are usually global map coordinates,
    # so distance from (0,0) is meaningless unless we transform to ego-frame.
    # However, we can check if they cluster.

    print(
        f"Coordinate Ranges -> X: [{cx.min():.2f}, {cx.max():.2f}], Y: [{cy.min():.2f}, {cy.max():.2f}]"
    )
    if (cx.max() - cx.min()) > 1000:
        print("  Note: Coordinates appear to be Global Map Coordinates (large range).")
    else:
        print("  Note: Coordinates appear to be Local/Ego Coordinates.")


def main():
    set_seeds(SEED)

    print("Starting Exploratory Data Analysis...")
    print(f"Data Source: {METADATA_PATH}")

    try:
        # 1. Data Integrity & Loading
        df = load_data()

        # 2. Target Variable Analysis
        ann_df = analyze_target_variables(df)

        # 3. Input Data Analysis (Tabular/Metadata)
        analyze_tabular_metadata(df)

        # 4. Input Data Analysis (Image)
        analyze_image_data(df)

        # 5. Feature Relationships
        analyze_relationships(ann_df)

        print("\nEDA Complete.")

    except Exception as e:
        print(f"\nAn error occurred during EDA: {e}")
        # Print traceback for debugging if needed, but keeping output clean as requested
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
