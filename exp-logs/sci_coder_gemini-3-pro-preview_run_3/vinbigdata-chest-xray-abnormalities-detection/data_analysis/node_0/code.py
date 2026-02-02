import pandas as pd
import numpy as np
import os
import random
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_ROOT = "./input"
SEED = 42

# Attempt to import pydicom for image reading
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("==== TARGET VARIABLE ANALYSIS ====")

    # 1. Distribution
    n_total = len(df)
    class_counts = df["class_id"].value_counts().sort_index()

    # Create a mapping for class names
    class_map = (
        df[["class_id", "class_name"]]
        .drop_duplicates()
        .set_index("class_id")["class_name"]
    )

    print(f"Total Annotations: {n_total}")
    print("\n--- Class Distribution ---")
    for cid in class_counts.index:
        count = class_counts[cid]
        pct = (count / n_total) * 100
        name = class_map.get(cid, "Unknown")
        print(f"Class {cid:2d} | {name:<20} | Count: {count:5d} | {pct:.4f}%")

    # 2. Imbalance
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f"\nClass Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Identify rare classes (< 1%)
    rare_classes = class_counts[class_counts / n_total < 0.01].index.tolist()
    rare_names = [class_map.get(c) for c in rare_classes]
    print(f"Rare Classes (<1% freq): {rare_names}")


def analyze_image_metadata(df):
    print("\n==== INPUT DATA ANALYSIS (IMAGE/BBOX) ====")

    # 1. Bounding Box Analysis (Proxy for Object Dimensions)
    # Filter out Class 14 (No finding) which uses 1-pixel placeholders
    df_findings = df[df["class_id"] != 14].copy()

    if len(df_findings) > 0:
        df_findings["width"] = df_findings["x_max"] - df_findings["x_min"]
        df_findings["height"] = df_findings["y_max"] - df_findings["y_min"]
        df_findings["area"] = df_findings["width"] * df_findings["height"]
        df_findings["aspect_ratio"] = df_findings["width"] / df_findings["height"]

        print("--- Bounding Box Statistics (Findings Only) ---")
        print(
            f"BBox Width  - Mean: {df_findings['width'].mean():.4f}, Std: {df_findings['width'].std():.4f}"
        )
        print(
            f"BBox Height - Mean: {df_findings['height'].mean():.4f}, Std: {df_findings['height'].std():.4f}"
        )
        print(
            f"BBox Area   - Mean: {df_findings['area'].mean():.4f}, Std: {df_findings['area'].std():.4f}"
        )
        print(f"Aspect Ratio- Mean: {df_findings['aspect_ratio'].mean():.4f}")

        # Outliers in Area (IQR Method)
        Q1 = df_findings["area"].quantile(0.25)
        Q3 = df_findings["area"].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (
            (df_findings["area"] < (Q1 - 1.5 * IQR))
            | (df_findings["area"] > (Q3 + 1.5 * IQR))
        ).sum()
        print(
            f"BBox Area Outliers (IQR method): {outliers} ({outliers/len(df_findings)*100:.2f}%)"
        )

    # 2. Pixel Analysis (Conditional)
    if HAS_PYDICOM:
        print("\n--- Pixel & Dimension Analysis (Sampled) ---")
        sample_size = 200
        unique_files = df["file_path"].unique()
        sampled_files = np.random.choice(
            unique_files, min(len(unique_files), sample_size), replace=False
        )

        heights, widths, means, stds = [], [], [], []

        for rel_path in sampled_files:
            try:
                path = os.path.join(INPUT_ROOT, rel_path)
                ds = pydicom.dcmread(path, stop_before_pixels=False)
                arr = ds.pixel_array.astype(np.float32)

                heights.append(ds.Rows)
                widths.append(ds.Columns)
                means.append(np.mean(arr))
                stds.append(np.std(arr))
            except Exception:
                continue

        if len(heights) > 0:
            print(f"Sample Size: {len(heights)}")
            print(
                f"Image Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}"
            )
            print(
                f"Image Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}"
            )
            print(f"Pixel Value  - Global Mean: {np.mean(means):.4f}")
            print(f"Pixel Value  - Global Std:  {np.mean(stds):.4f}")
    else:
        print("\n--- Pixel Analysis ---")
        print(
            "NOTE: 'pydicom' library not detected. Skipping raw DICOM pixel analysis."
        )


def analyze_relationships(df):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # 1. Structured: Correlations between BBox Area and Class
    df_findings = df[df["class_id"] != 14].copy()
    if len(df_findings) > 0:
        df_findings["area"] = (df_findings["x_max"] - df_findings["x_min"]) * (
            df_findings["y_max"] - df_findings["y_min"]
        )

        print("--- Top 5 Classes by Average BBox Area ---")
        avg_area = (
            df_findings.groupby("class_name")["area"]
            .mean()
            .sort_values(ascending=False)
        )
        for name, area in avg_area.head(5).items():
            print(f"{name:<20}: {area:.4f}")

    # 2. Meta-Feature: Annotations per Image
    print("\n--- Annotations per Image ---")
    counts = df.groupby("image_id").size()
    print(f"Mean Annotations/Image: {counts.mean():.4f}")
    print(f"Max Annotations/Image:  {counts.max()}")

    # 3. Exclusivity of 'No finding'
    print("\n--- Class Exclusivity ---")
    # Get images that have class 14
    no_finding_imgs = df[df["class_id"] == 14]["image_id"].unique()
    # Check if these images have any other class in the full dataframe
    subset = df[df["image_id"].isin(no_finding_imgs)]
    # Filter for rows that are NOT class 14
    contradictions = subset[subset["class_id"] != 14]

    if len(contradictions) == 0:
        print(
            "Class 14 ('No finding') is strictly exclusive (never appears with other classes)."
        )
    else:
        print(
            f"Class 14 co-occurs with other findings in {len(contradictions)} instances."
        )


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    analyze_target(df)
    analyze_image_metadata(df)
    analyze_relationships(df)


if __name__ == "__main__":
    main()
