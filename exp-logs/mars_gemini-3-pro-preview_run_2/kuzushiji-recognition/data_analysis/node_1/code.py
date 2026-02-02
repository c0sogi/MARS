import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter
from scipy.stats import skew, kurtosis

# Set constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
UNICODE_MAP_CSV = os.path.join(INPUT_DIR, "unicode_translation.csv")
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


def parse_labels(label_str):
    """
    Parses a label string into a list of dictionaries.
    Format: Unicode X Y W H ...
    """
    if pd.isna(label_str) or label_str == "":
        return []

    parts = label_str.split()
    # Each label has 5 components: char, x, y, w, h
    if len(parts) % 5 != 0:
        # In case of malformed string, handle gracefully or skip
        return []

    labels = []
    for i in range(0, len(parts), 5):
        try:
            char = parts[i]
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            w = int(parts[i + 3])
            h = int(parts[i + 4])
            labels.append({"char": char, "x": x, "y": y, "w": w, "h": h})
        except ValueError:
            continue
    return labels


def analyze_targets(df, unicode_map):
    """
    Analyzes the target variable: Class distribution and Bounding Boxes.
    """
    all_chars = []
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_aspect_ratios = []
    chars_per_image = []

    # Store normalized centroids for spatial analysis later
    normalized_centroids_x = []
    normalized_centroids_y = []

    # We need image dimensions for normalization, so we'll do a quick pass or look up if available.
    # Since we analyze images separately, we will store raw coords here and normalize during image loop
    # or just analyze raw distribution if image sizes are consistent.
    # However, image sizes vary. We will link this in the main loop for efficiency.

    # Pre-pass to parse all labels
    parsed_data = []

    for _, row in df.iterrows():
        labels = parse_labels(row.get("labels", ""))
        chars_per_image.append(len(labels))

        row_chars = []
        row_boxes = []

        for l in labels:
            all_chars.append(l["char"])
            row_chars.append(l["char"])

            w, h = l["w"], l["h"]
            bbox_widths.append(w)
            bbox_heights.append(h)
            bbox_areas.append(w * h)
            if h > 0:
                bbox_aspect_ratios.append(w / h)
            else:
                bbox_aspect_ratios.append(0)

            row_boxes.append((l["x"], l["y"], l["w"], l["h"]))

        parsed_data.append(
            {"image_id": row["image_id"], "num_chars": len(labels), "boxes": row_boxes}
        )

    # Class stats
    char_counts = Counter(all_chars)
    total_chars = len(all_chars)
    unique_classes = len(char_counts)

    # Top 5 classes
    top_5 = char_counts.most_common(5)
    top_5_formatted = []
    for char_code, count in top_5:
        char_name = unicode_map.get(char_code, "Unknown")
        top_5_formatted.append(
            f"{char_code} ({char_name}): {count} ({(count/total_chars)*100:.2f}%)"
        )

    # Imbalance
    counts = list(char_counts.values())
    if counts:
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
    else:
        imbalance_ratio = 0

    return {
        "total_chars": total_chars,
        "unique_classes": unique_classes,
        "top_5": top_5_formatted,
        "imbalance_ratio": imbalance_ratio,
        "bbox_stats": {
            "width_mean": np.mean(bbox_widths) if bbox_widths else 0,
            "width_std": np.std(bbox_widths) if bbox_widths else 0,
            "height_mean": np.mean(bbox_heights) if bbox_heights else 0,
            "height_std": np.std(bbox_heights) if bbox_heights else 0,
            "area_mean": np.mean(bbox_areas) if bbox_areas else 0,
            "ar_mean": np.mean(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
        },
        "counts_per_image": chars_per_image,
        "parsed_data": parsed_data,
    }


def analyze_images(df):
    """
    Analyzes image dimensions, channels, and pixel stats.
    Uses a sample for pixel stats to save time.
    """
    img_widths = []
    img_heights = []
    img_areas = []
    img_ars = []
    channel_counts = Counter()

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)  # Assume max 3 channels for accumulation
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Sample for pixel stats
    sample_size = 300
    sample_indices = set(random.sample(range(len(df)), min(sample_size, len(df))))

    # For correlation analysis later
    img_meta_list = []

    for idx, row in df.iterrows():
        # Path construction: metadata contains relative path from input dir
        # e.g. train_images/xxx.jpg.
        # We need to prepend INPUT_DIR ("./input")
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image header only if possible, but cv2.imread loads full image.
        # To optimize, we load full image but only compute complex stats on sample.
        # Given 2245 images, loading all is feasible (~1-2 mins).

        img = cv2.imread(full_path)
        if img is None:
            continue

        h, w, c = img.shape
        img_widths.append(w)
        img_heights.append(h)
        img_areas.append(w * h)
        img_ars.append(w / h)
        channel_counts[c] += 1

        img_meta_list.append(
            {"image_id": row["image_id"], "width": w, "height": h, "area": w * h}
        )

        # Pixel stats on sample
        if idx in sample_indices:
            # Normalize to 0-1 for stats
            img_norm = img.astype(np.float32) / 255.0
            # CV2 is BGR, we usually analyze as is or convert. Let's keep channel separation.
            # Accumulate
            # Reshape to (N, 3)
            pixels = img_norm.reshape(-1, c)
            pixel_sum[:c] += pixels.sum(axis=0)
            pixel_sq_sum[:c] += (pixels**2).sum(axis=0)
            pixel_count += pixels.shape[0]

    # Calculate global mean and std
    if pixel_count > 0:
        # Assuming 3 channels (BGR)
        global_mean = pixel_sum / pixel_count
        global_std = np.sqrt((pixel_sq_sum / pixel_count) - (global_mean**2))
    else:
        global_mean = np.zeros(3)
        global_std = np.zeros(3)

    return {
        "width_stats": (
            np.mean(img_widths),
            np.std(img_widths),
            np.min(img_widths),
            np.max(img_widths),
        ),
        "height_stats": (
            np.mean(img_heights),
            np.std(img_heights),
            np.min(img_heights),
            np.max(img_heights),
        ),
        "ar_stats": (np.mean(img_ars), np.std(img_ars)),
        "channels": dict(channel_counts),
        "pixel_mean": global_mean,  # BGR order
        "pixel_std": global_std,
        "img_meta": img_meta_list,
    }


def main():
    print("Starting EDA on Kuzushiji Dataset...")

    # 1. Load Data
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        unicode_df = pd.read_csv(UNICODE_MAP_CSV)
        # Create a map for unicode to char
        unicode_map = dict(zip(unicode_df["Unicode"], unicode_df["char"]))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Target Variable Analysis
    target_stats = analyze_targets(train_df, unicode_map)

    # 3. Input Data Analysis
    image_stats = analyze_images(train_df)

    # 4. Feature/Signal Relationships
    # Merge parsed data with image meta
    # parsed_data has num_chars, img_meta has width/height/area
    # Both are lists derived from iteration order of train_df (assuming no skips, but check IDs)

    img_meta_map = {item["image_id"]: item for item in image_stats["img_meta"]}
    parsed_data_map = {item["image_id"]: item for item in target_stats["parsed_data"]}

    # Align data
    aligned_data = []
    normalized_centers_x = []
    normalized_centers_y = []

    for img_id, meta in img_meta_map.items():
        if img_id in parsed_data_map:
            p_data = parsed_data_map[img_id]
            num_chars = p_data["num_chars"]
            aligned_data.append(
                {
                    "width": meta["width"],
                    "height": meta["height"],
                    "area": meta["area"],
                    "num_chars": num_chars,
                }
            )

            # Spatial analysis
            w, h = meta["width"], meta["height"]
            for box in p_data["boxes"]:
                bx, by, bw, bh = box
                cx = bx + bw / 2
                cy = by + bh / 2
                normalized_centers_x.append(cx / w)
                normalized_centers_y.append(cy / h)

    df_corr = pd.DataFrame(aligned_data)

    # Correlations
    if not df_corr.empty:
        corr_area_count = df_corr["area"].corr(df_corr["num_chars"])
        corr_width_count = df_corr["width"].corr(df_corr["num_chars"])
        corr_height_count = df_corr["height"].corr(df_corr["num_chars"])
    else:
        corr_area_count = 0
        corr_width_count = 0
        corr_height_count = 0

    # Spatial stats
    mean_norm_x = np.mean(normalized_centers_x) if normalized_centers_x else 0
    mean_norm_y = np.mean(normalized_centers_y) if normalized_centers_y else 0

    # 5. Output Report
    print("\nDATA INTEGRITY")
    print(f"Analysis performed on {len(train_df)} training samples.")
    print("Strictly using ./metadata/train.csv to prevent leakage.")

    print("\nTARGET VARIABLE ANALYSIS")
    print(f"Total Characters: {target_stats['total_chars']}")
    print(f"Unique Character Classes: {target_stats['unique_classes']}")
    print("Top 5 Frequent Classes:")
    for s in target_stats["top_5"]:
        print(f"  - {s}")
    print(f"Class Imbalance Ratio (Max/Min): {target_stats['imbalance_ratio']:.4f}")

    print("\nBounding Box Statistics:")
    print(
        f"  Mean Width: {target_stats['bbox_stats']['width_mean']:.4f} (Std: {target_stats['bbox_stats']['width_std']:.4f})"
    )
    print(
        f"  Mean Height: {target_stats['bbox_stats']['height_mean']:.4f} (Std: {target_stats['bbox_stats']['height_std']:.4f})"
    )
    print(f"  Mean Area: {target_stats['bbox_stats']['area_mean']:.4f}")
    print(f"  Mean Aspect Ratio (W/H): {target_stats['bbox_stats']['ar_mean']:.4f}")

    counts = target_stats["counts_per_image"]
    print(
        f"Characters per Page: Mean = {np.mean(counts):.4f}, Std = {np.std(counts):.4f}, Max = {np.max(counts)}"
    )

    print("\nINPUT DATA ANALYSIS (IMAGE)")
    w_stats = image_stats["width_stats"]
    h_stats = image_stats["height_stats"]
    print("Dimensions:")
    print(
        f"  Width: Mean={w_stats[0]:.4f}, Std={w_stats[1]:.4f}, Range=[{w_stats[2]}, {w_stats[3]}]"
    )
    print(
        f"  Height: Mean={h_stats[0]:.4f}, Std={h_stats[1]:.4f}, Range=[{h_stats[2]}, {h_stats[3]}]"
    )
    print(
        f"  Aspect Ratio: Mean={image_stats['ar_stats'][0]:.4f}, Std={image_stats['ar_stats'][1]:.4f}"
    )

    print("Channels:")
    for c, count in image_stats["channels"].items():
        type_str = "RGB" if c == 3 else ("Grayscale" if c == 1 else f"{c}-channel")
        print(f"  {type_str} ({c}): {count} images")

    print("Pixel Stats (Normalized 0-1, BGR Order):")
    # Handle case where mean might be size 1 or 3
    means = image_stats["pixel_mean"]
    stds = image_stats["pixel_std"]
    if len(means) == 3:
        print(f"  Mean: B={means[0]:.4f}, G={means[1]:.4f}, R={means[2]:.4f}")
        print(f"  Std:  B={stds[0]:.4f}, G={stds[1]:.4f}, R={stds[2]:.4f}")
    else:
        print(f"  Mean: {means[0]:.4f}")
        print(f"  Std:  {stds[0]:.4f}")

    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("Unstructured Relationships (Metadata vs Target):")
    print(f"  Correlation (Image Area vs Num Characters): {corr_area_count:.4f}")
    print(f"  Correlation (Image Width vs Num Characters): {corr_width_count:.4f}")
    print(f"  Correlation (Image Height vs Num Characters): {corr_height_count:.4f}")

    print("Spatial Distribution (Normalized Centroids):")
    print(f"  Mean X Position (0=Left, 1=Right): {mean_norm_x:.4f}")
    print(f"  Mean Y Position (0=Top, 1=Bottom): {mean_norm_y:.4f}")
    if mean_norm_x < 0.45:
        x_desc = "Left-skewed"
    elif mean_norm_x > 0.55:
        x_desc = "Right-skewed"
    else:
        x_desc = "Centered"

    if mean_norm_y < 0.45:
        y_desc = "Top-skewed"
    elif mean_norm_y > 0.55:
        y_desc = "Bottom-skewed"
    else:
        y_desc = "Centered"
    print(
        f"  Interpretation: Characters are roughly {x_desc} horizontally and {y_desc} vertically."
    )


if __name__ == "__main__":
    main()
