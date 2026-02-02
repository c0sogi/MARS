import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import glob
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42

# Set Seeds
random.seed(SEED)
np.random.seed(SEED)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found at {METADATA_PATH}")
        return None

    df = pd.read_csv(METADATA_PATH)
    # Parse JSON strings
    df["annotations"] = df["annotations"].apply(json.loads)
    df["file_paths"] = df["file_paths"].apply(json.loads)
    return df


def analyze_targets(df):
    print_section("Target Variable Analysis")

    # Flatten annotations
    all_objects = []
    for anns in df["annotations"]:
        all_objects.extend(anns)

    if not all_objects:
        print("No annotations found in training data.")
        return None

    df_objs = pd.DataFrame(all_objects)

    # 1. Class Distribution
    print("--- Class Distribution ---")
    class_counts = df_objs["class_name"].value_counts()
    total_objs = len(df_objs)
    for cls, count in class_counts.items():
        ratio = count / total_objs
        print(f"{cls}: {count} ({ratio:.2%})")

    # 2. Bounding Box Dimensions
    print("\n--- Bounding Box Dimensions (Meters) ---")
    dims = ["width", "length", "height"]
    stats = df_objs[dims].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string(float_format="{:.4f}".format))

    # 3. Spatial Distribution
    print("\n--- Spatial Distribution (Center Coordinates) ---")
    coords = ["center_x", "center_y", "center_z"]
    stats_coords = df_objs[coords].describe().T[["mean", "std", "min", "max"]]
    print(stats_coords.to_string(float_format="{:.4f}".format))

    return df_objs


def analyze_images(df):
    print_section("Input Data Analysis: Images")

    # Collect all image paths
    img_paths = []
    for paths in df["file_paths"]:
        for sensor, rel_path in paths.items():
            if "CAM" in sensor or rel_path.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(INPUT_DIR, rel_path))

    if not img_paths:
        print("No image files found in metadata.")
        return

    # Sample for efficiency
    sample_size = min(200, len(img_paths))
    sampled_paths = random.sample(img_paths, sample_size)

    widths = []
    heights = []
    channels = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    print(f"Analyzing {sample_size} sampled images...")

    for p in sampled_paths:
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
        channels.append(c)

        # Accumulate for mean/std
        # Normalize to 0-1 for calculation usually, but standard is often 0-255 stats
        # We'll report 0-255 stats
        img_flat = img.reshape(-1, 3)
        pixel_sum += img_flat.sum(axis=0)
        pixel_sq_sum += (img_flat**2).sum(axis=0)
        pixel_count += img_flat.shape[0]

    if not widths:
        print("Could not read any images.")
        return

    # Dimensions
    print("\n--- Image Dimensions ---")
    print(
        f"Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )

    # Channels
    print("\n--- Channels ---")
    print(f"Channel Counts: {Counter(channels)}")

    # Pixel Stats
    if pixel_count > 0:
        mean = pixel_sum / pixel_count
        # std = sqrt(E[x^2] - (E[x])^2)
        std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))

        print("\n--- Pixel Intensity Stats (RGB, 0-255) ---")
        print(f"Mean: R={mean[0]:.4f}, G={mean[1]:.4f}, B={mean[2]:.4f}")
        print(f"Std:  R={std[0]:.4f}, G={std[1]:.4f}, B={std[2]:.4f}")


def analyze_lidar(df):
    print_section("Input Data Analysis: LiDAR")

    # Collect all lidar paths
    lidar_paths = []
    for paths in df["file_paths"]:
        for sensor, rel_path in paths.items():
            if "LIDAR" in sensor or rel_path.lower().endswith((".bin", ".pcd")):
                lidar_paths.append(os.path.join(INPUT_DIR, rel_path))

    if not lidar_paths:
        print("No LiDAR files found in metadata.")
        return

    # Sample for efficiency
    sample_size = min(100, len(lidar_paths))
    sampled_paths = random.sample(lidar_paths, sample_size)

    point_counts = []
    min_vals = []
    max_vals = []

    print(f"Analyzing {sample_size} sampled LiDAR files...")

    for p in sampled_paths:
        if not os.path.exists(p):
            continue

        try:
            # NuScenes/Lyft lidar data is typically float32
            # Usually (x, y, z, intensity) or (x, y, z, intensity, ring)
            # We will try to reshape to (-1, 5) first, then (-1, 4)
            raw_data = np.fromfile(p, dtype=np.float32)

            points = None
            # Heuristic to determine shape
            if raw_data.size % 5 == 0:
                points = raw_data.reshape(-1, 5)
            elif raw_data.size % 4 == 0:
                points = raw_data.reshape(-1, 4)
            elif raw_data.size % 3 == 0:
                points = raw_data.reshape(-1, 3)
            else:
                # Unknown format, skip
                continue

            point_counts.append(points.shape[0])

            # Spatial extent (x, y, z)
            xyz = points[:, :3]
            min_vals.append(xyz.min(axis=0))
            max_vals.append(xyz.max(axis=0))

        except Exception as e:
            continue

    if not point_counts:
        print("Could not read any LiDAR files.")
        return

    min_vals = np.array(min_vals)
    max_vals = np.array(max_vals)

    print("\n--- Point Cloud Statistics ---")
    print(
        f"Points per Scan: Mean={np.mean(point_counts):.4f}, Std={np.std(point_counts):.4f}"
    )
    print(f"Min Points: {np.min(point_counts)}, Max Points: {np.max(point_counts)}")

    print("\n--- Spatial Extent (Global Min/Max across sample) ---")
    # Min of mins, Max of maxs
    global_min = min_vals.min(axis=0)
    global_max = max_vals.max(axis=0)

    print(f"X Range: {global_min[0]:.4f} to {global_max[0]:.4f}")
    print(f"Y Range: {global_min[1]:.4f} to {global_max[1]:.4f}")
    print(f"Z Range: {global_min[2]:.4f} to {global_max[2]:.4f}")


def analyze_relationships(df, df_objs):
    print_section("Feature Relationships")

    if df_objs is None or df_objs.empty:
        print("No object data to analyze relationships.")
        return

    # 1. Object Density per Sample
    print("--- Object Density ---")
    objs_per_sample = df["annotations"].apply(len)
    print(
        f"Objects per Sample: Mean={objs_per_sample.mean():.4f}, Std={objs_per_sample.std():.4f}"
    )
    print(
        f"Zero-object Samples: {(objs_per_sample == 0).sum()} ({(objs_per_sample == 0).mean():.2%})"
    )

    # 2. Distance vs Class
    # Calculate distance from origin (0,0,0) - assuming ego is at origin or close to it in local coords
    # Note: center_x/y/z are world coordinates in some datasets, but often local in others.
    # Without ego_pose transformation, this is an approximation of spatial distribution in the map frame.
    # However, we can look at the spread.

    df_objs["distance"] = np.sqrt(df_objs["center_x"] ** 2 + df_objs["center_y"] ** 2)

    print("\n--- Mean Distance from Origin (0,0) by Class ---")
    # This gives an idea if certain classes appear further away or in specific map regions
    dist_by_class = df_objs.groupby("class_name")["distance"].mean().sort_values()
    for cls, dist in dist_by_class.items():
        print(f"{cls}: {dist:.4f}")

    # 3. Volume vs Class
    df_objs["volume"] = df_objs["width"] * df_objs["length"] * df_objs["height"]
    print("\n--- Mean Volume (m^3) by Class ---")
    vol_by_class = (
        df_objs.groupby("class_name")["volume"].mean().sort_values(ascending=False)
    )
    for cls, vol in vol_by_class.items():
        print(f"{cls}: {vol:.4f}")

    # 4. Correlation: Width vs Length
    corr = df_objs["width"].corr(df_objs["length"])
    print(f"\nCorrelation between Width and Length: {corr:.4f}")


def main():
    print("Starting Exploratory Data Analysis...")

    df_train = load_metadata()
    if df_train is None:
        return

    # Target Analysis
    df_objs = analyze_targets(df_train)

    # Input Analysis
    analyze_images(df_train)
    analyze_lidar(df_train)

    # Relationships
    analyze_relationships(df_train, df_objs)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
