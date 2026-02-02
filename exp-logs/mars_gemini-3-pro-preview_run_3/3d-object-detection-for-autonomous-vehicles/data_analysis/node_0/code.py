import os
import pandas as pd
import numpy as np
from collections import Counter
import warnings

# Configuration
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
RANDOM_SEED = 42
LIDAR_SAMPLE_SIZE = 200  # Number of lidar files to analyze for input stats

# Set seeds
np.random.seed(RANDOM_SEED)


def print_section(title):
    print(f"\n{'='*40}")
    print(f"{title.upper()}")
    print(f"{'='*40}")


def parse_predictions(df):
    """
    Parses the space-delimited label strings into a structured DataFrame of objects.
    Format: center_x center_y center_z width length height yaw class_name
    """
    all_objects = []

    # Pre-allocate lists for speed
    centers_x = []
    centers_y = []
    centers_z = []
    widths = []
    lengths = []
    heights = []
    yaws = []
    classes = []
    sample_tokens = []

    for idx, row in df.iterrows():
        label_str = row["label"]
        s_token = row["sample_token"]

        if pd.isna(label_str) or label_str == "":
            continue

        parts = str(label_str).strip().split()
        num_parts = len(parts)
        stride = 8

        if num_parts % stride != 0:
            continue

        num_objects = num_parts // stride

        for i in range(num_objects):
            offset = i * stride
            try:
                centers_x.append(float(parts[offset]))
                centers_y.append(float(parts[offset + 1]))
                centers_z.append(float(parts[offset + 2]))
                widths.append(float(parts[offset + 3]))
                lengths.append(float(parts[offset + 4]))
                heights.append(float(parts[offset + 5]))
                yaws.append(float(parts[offset + 6]))
                classes.append(parts[offset + 7])
                sample_tokens.append(s_token)
            except ValueError:
                continue

    objects_df = pd.DataFrame(
        {
            "sample_token": sample_tokens,
            "center_x": centers_x,
            "center_y": centers_y,
            "center_z": centers_z,
            "width": widths,
            "length": lengths,
            "height": heights,
            "yaw": yaws,
            "class_name": classes,
        }
    )

    return objects_df


def analyze_targets(objects_df, total_samples):
    print_section("Target Variable Analysis")

    # 1. Distribution
    print("--- Class Distribution ---")
    class_counts = objects_df["class_name"].value_counts()
    total_objs = len(objects_df)

    for cls, count in class_counts.items():
        ratio = count / total_objs
        print(f"{cls:<20} | Count: {count:<8} | Ratio: {ratio:.4f}")

    print(f"\nTotal Objects: {total_objs}")
    print(f"Avg Objects per Sample: {total_objs / total_samples:.4f}")

    # 2. Continuous Variable Analysis (Box Dimensions)
    print("\n--- Bounding Box Dimensions (Meters) ---")
    dims = ["width", "length", "height"]
    stats = objects_df[dims].describe().T[["mean", "std", "min", "max"]]

    # Add skewness
    skew = objects_df[dims].skew()
    stats["skew"] = skew

    # Formatting
    print(
        f"{'Dimension':<10} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Skew':<10}"
    )
    print("-" * 75)
    for idx, row in stats.iterrows():
        print(
            f"{idx:<10} | {row['mean']:<10.4f} | {row['std']:<10.4f} | {row['min']:<10.4f} | {row['max']:<10.4f} | {row['skew']:<10.4f}"
        )

    # 3. Spatial Distribution
    print("\n--- Spatial Distribution (Center Coordinates) ---")
    coords = ["center_x", "center_y", "center_z"]
    coord_stats = objects_df[coords].describe().T[["mean", "std", "min", "max"]]

    print(
        f"{'Coordinate':<10} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}"
    )
    print("-" * 65)
    for idx, row in coord_stats.iterrows():
        print(
            f"{idx:<10} | {row['mean']:<10.4f} | {row['std']:<10.4f} | {row['min']:<10.4f} | {row['max']:<10.4f}"
        )


def analyze_lidar_inputs(metadata_df):
    print_section("Input Data Analysis (Lidar)")

    # Sample files
    sample_df = metadata_df.sample(
        n=min(len(metadata_df), LIDAR_SAMPLE_SIZE), random_state=RANDOM_SEED
    )

    point_counts = []
    x_means, y_means, z_means = [], [], []
    intensities_mean = []

    missing_files = 0

    print(f"Analyzing {len(sample_df)} sampled Lidar files...")

    for _, row in sample_df.iterrows():
        path = os.path.join(INPUT_DIR, row["lidar_path"])
        if not os.path.exists(path):
            missing_files += 1
            continue

        try:
            # Standard Lidar format often float32, x,y,z,intensity (4 values) or x,y,z,intensity,ring (5 values)
            # We assume standard 4-feature or 5-feature packed binary
            raw_data = np.fromfile(path, dtype=np.float32)

            # Attempt to reshape. Common strides are 4 or 5.
            # If length is divisible by 5, try that, else 4.
            if len(raw_data) % 5 == 0:
                points = raw_data.reshape(-1, 5)
            elif len(raw_data) % 4 == 0:
                points = raw_data.reshape(-1, 4)
            else:
                # Fallback: treat as flat list of points (x,y,z) if divisible by 3?
                # Or just skip detailed stats if format is unknown
                continue

            point_counts.append(len(points))

            # Spatial stats (first 3 cols)
            x_means.append(points[:, 0].mean())
            y_means.append(points[:, 1].mean())
            z_means.append(points[:, 2].mean())

            # Intensity (4th col)
            if points.shape[1] >= 4:
                intensities_mean.append(points[:, 3].mean())

        except Exception:
            continue

    if missing_files > 0:
        print(f"WARNING: {missing_files} files were missing.")

    if not point_counts:
        print("No valid Lidar data could be parsed.")
        return

    # Report Stats
    pc_series = pd.Series(point_counts)
    print("\n--- Point Cloud Statistics ---")
    print(f"Avg Points per Scan: {pc_series.mean():.4f}")
    print(f"Std Points per Scan: {pc_series.std():.4f}")
    print(f"Min Points per Scan: {pc_series.min():.4f}")
    print(f"Max Points per Scan: {pc_series.max():.4f}")

    print("\n--- Global Signal Stats (Sampled) ---")
    print(f"Global Mean X: {np.mean(x_means):.4f}")
    print(f"Global Mean Y: {np.mean(y_means):.4f}")
    print(f"Global Mean Z: {np.mean(z_means):.4f}")
    if intensities_mean:
        print(f"Global Mean Intensity: {np.mean(intensities_mean):.4f}")


def analyze_relationships(objects_df, metadata_df):
    print_section("Feature/Signal Relationships")

    # 1. Correlation between dimensions
    print("--- Correlation: Box Dimensions ---")
    corr = objects_df[["width", "length", "height"]].corr()
    print(corr.round(4))

    # Check for collinearity
    print("\n--- Redundancy Check (>0.90 Correlation) ---")
    found_redundancy = False
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.90:
                print(
                    f"High Correlation: {corr.columns[i]} <-> {corr.columns[j]} ({corr.iloc[i, j]:.4f})"
                )
                found_redundancy = True
    if not found_redundancy:
        print("No highly collinear features found among box dimensions.")

    # 2. Class vs Dimensions (Importance)
    print("\n--- Average Dimensions by Class ---")
    class_dims = objects_df.groupby("class_name")[["width", "length", "height"]].mean()
    print(f"{'Class':<20} | {'Width':<10} | {'Length':<10} | {'Height':<10}")
    print("-" * 60)
    for cls, row in class_dims.iterrows():
        print(
            f"{cls:<20} | {row['width']:<10.4f} | {row['length']:<10.4f} | {row['height']:<10.4f}"
        )

    # 3. Meta-Feature Relationship: Object Count per Scene
    # We need to map samples back to scenes if possible, but here we check objects per sample
    print("\n--- Meta-Feature: Objects per Sample ---")
    objs_per_sample = objects_df.groupby("sample_token").size()
    # Include samples with 0 objects
    all_samples = set(metadata_df["sample_token"])
    samples_with_objs = set(objs_per_sample.index)
    zero_obj_count = len(all_samples - samples_with_objs)

    # Create a series including zeros
    counts_list = objs_per_sample.tolist() + [0] * zero_obj_count
    counts_series = pd.Series(counts_list)

    print(f"Mean Objects/Sample: {counts_series.mean():.4f}")
    print(f"Max Objects/Sample:  {counts_series.max():.4f}")
    print(
        f"Samples with 0 objs: {zero_obj_count} ({zero_obj_count/len(metadata_df):.2%})"
    )


def main():
    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Parse Targets
    objects_df = parse_predictions(df)

    # Run Analysis
    analyze_targets(objects_df, len(df))
    analyze_lidar_inputs(df)
    analyze_relationships(objects_df, df)


if __name__ == "__main__":
    main()
