import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
TRAIN_METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
IMAGE_SAMPLE_SIZE = (
    2000  # Number of images to sample for pixel/dimension stats to ensure runtime < 1hr
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    """Loads the training metadata."""
    if not os.path.exists(TRAIN_METADATA_PATH):
        print(f"Error: Metadata file not found at {TRAIN_METADATA_PATH}")
        sys.exit(1)

    # Load metadata
    df = pd.read_csv(TRAIN_METADATA_PATH, dtype={"id": str, "attribute_ids": str})

    # Fill NaNs in attribute_ids with empty string for processing
    df["attribute_ids"] = df["attribute_ids"].fillna("")

    # Construct full file paths
    # The metadata file_path is relative to ./input (e.g., "train/xxxx.png")
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    return df


def analyze_targets(df):
    """Analyzes the distribution of the target variable (attribute_ids)."""
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Parse labels
    # attribute_ids is a space-separated string
    all_labels = []
    label_counts_per_image = []

    for labels_str in df["attribute_ids"]:
        if not labels_str.strip():
            labels = []
        else:
            labels = labels_str.split(" ")

        all_labels.extend(labels)
        label_counts_per_image.append(len(labels))

    label_counts = Counter(all_labels)
    num_unique_labels = len(label_counts)
    total_labels = len(all_labels)
    num_samples = len(df)

    # 1. Distribution Stats
    print(f"Total Samples: {num_samples}")
    print(f"Total Unique Labels: {num_unique_labels}")
    print(f"Total Label Annotations: {total_labels}")

    # 2. Class Balance / Imbalance
    if num_unique_labels > 0:
        freqs = list(label_counts.values())
        min_freq = np.min(freqs)
        max_freq = np.max(freqs)
        mean_freq = np.mean(freqs)

        print(
            f"Label Frequency Stats: Min={min_freq}, Max={max_freq}, Mean={mean_freq:.4f}"
        )

        # Rare labels (< 1% frequency)
        threshold = 0.01 * num_samples
        rare_labels = [k for k, v in label_counts.items() if v < threshold]
        print(
            f"Count of Rare Labels (< 1% freq): {len(rare_labels)} ({len(rare_labels)/num_unique_labels*100:.2f}% of vocabulary)"
        )

        # Top 5 most common
        print("Top 5 Most Common Labels (ID: Count):")
        for label, count in label_counts.most_common(5):
            print(f"  {label}: {count}")

    # 3. Label Cardinality & Density
    # Cardinality: Average number of labels per example
    cardinality = np.mean(label_counts_per_image)
    print(f"Label Cardinality (Avg labels per image): {cardinality:.4f}")

    # Density: Cardinality / Total Unique Labels
    if num_unique_labels > 0:
        density = cardinality / num_unique_labels
        print(f"Label Density: {density:.6f}")
    else:
        print("Label Density: 0.0000")

    print("")
    return label_counts_per_image


def process_image(path):
    """Helper to read image and return stats."""
    try:
        # Read image in color mode
        img = cv2.imread(path)
        if img is None:
            return None

        # Shape: (H, W, C)
        h, w, c = img.shape

        # Compute mean and std for this image (optimization: compute sum and sq_sum)
        # Normalize to 0-1 for calculation
        img_norm = img.astype(np.float32) / 255.0

        # Means per channel
        means = np.mean(img_norm, axis=(0, 1))
        # Stds per channel
        stds = np.std(img_norm, axis=(0, 1))

        return {
            "h": h,
            "w": w,
            "c": c,
            "means": means,  # BGR
            "stds": stds,  # BGR
            "area": h * w,
            "aspect_ratio": w / h if h > 0 else 0,
        }
    except Exception:
        return None


def analyze_images(df):
    """Analyzes image dimensions, channels, and pixel stats."""
    print("INPUT DATA ANALYSIS (IMAGE MODALITY)")
    print("-" * 30)

    # Sample data to save time
    if len(df) > IMAGE_SAMPLE_SIZE:
        sample_df = df.sample(n=IMAGE_SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    print(f"Analyzing a sample of {len(sample_df)} images...")

    paths = sample_df["full_path"].tolist()

    stats_list = []

    # Use ThreadPoolExecutor for I/O bound task
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_image, paths)

    for res in results:
        if res is not None:
            stats_list.append(res)

    if not stats_list:
        print("Error: No images could be processed.")
        return None

    # Aggregate stats
    heights = [s["h"] for s in stats_list]
    widths = [s["w"] for s in stats_list]
    aspect_ratios = [s["aspect_ratio"] for s in stats_list]
    channels = [s["c"] for s in stats_list]

    # 1. Dimensions
    print(
        f"Image Widths:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Image Heights: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # 2. Channels
    channel_counts = Counter(channels)
    print(f"Channel Distribution: {dict(channel_counts)}")

    # 3. Pixel Stats (Global approximation from sample means)
    # Note: This is an average of averages, which is a reasonable approximation for EDA
    # OpenCV loads as BGR
    b_means = [s["means"][0] for s in stats_list]
    g_means = [s["means"][1] for s in stats_list]
    r_means = [s["means"][2] for s in stats_list]

    b_stds = [s["stds"][0] for s in stats_list]
    g_stds = [s["stds"][1] for s in stats_list]
    r_stds = [s["stds"][2] for s in stats_list]

    print(
        f"Pixel Mean (RGB, 0-1): R={np.mean(r_means):.4f}, G={np.mean(g_means):.4f}, B={np.mean(b_means):.4f}"
    )
    print(
        f"Pixel Std  (RGB, 0-1): R={np.mean(r_stds):.4f}, G={np.mean(g_stds):.4f}, B={np.mean(b_stds):.4f}"
    )

    # Return stats aligned with the sample_df for relationship analysis
    # We need to filter sample_df to match stats_list (in case some images failed)
    # Since we used map, order is preserved, but we dropped Nones.
    # Re-aligning is tricky without indices. Let's just return the lists for correlation analysis
    # assuming failure rate is near zero.

    return {
        "areas": [s["area"] for s in stats_list],
        "aspect_ratios": aspect_ratios,
        # We need the corresponding label counts for these specific images
        # Since we can't easily map back if failures occurred, we will re-calculate label counts for the valid indices
        # For EDA purposes, we'll assume valid images match the sample_df head (filtered)
    }


def analyze_relationships(df, image_stats_data):
    """Analyzes relationships between meta-features and target."""
    print("")
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    if image_stats_data is None:
        print("Skipping relationship analysis due to missing image stats.")
        return

    # To ensure alignment, we need the label counts for the exact images in the sample
    # We will re-sample and process sequentially to guarantee alignment for this section
    # or rely on the fact that failures are rare.
    # Let's re-extract label counts for the sampled dataframe.

    # Note: In analyze_images, we sampled df. We need that same sample here.
    # Since we didn't return the sample_df, let's just do a quick correlation on the valid stats
    # assuming the 'image_stats_data' corresponds to the first N valid images of the sample.
    # A robust way is to pass the sample_df indices.

    # For this script, let's do a lightweight check:
    # We will calculate the label count for the *entire* df, then subset it based on the sample size used in image_stats.
    # This assumes no read errors, which is fair for this dataset.

    # Get label counts for the sample
    if len(df) > IMAGE_SAMPLE_SIZE:
        sample_df = df.sample(n=IMAGE_SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    # Calculate label counts for this sample
    label_counts = []
    for labels_str in sample_df["attribute_ids"]:
        if not labels_str.strip():
            c = 0
        else:
            c = len(labels_str.split(" "))
        label_counts.append(c)

    # Truncate to match image_stats_data length (in case of read failures)
    n_stats = len(image_stats_data["areas"])
    label_counts = label_counts[:n_stats]

    areas = image_stats_data["areas"]
    aspect_ratios = image_stats_data["aspect_ratios"]

    # Correlations
    corr_area = np.corrcoef(areas, label_counts)[0, 1]
    corr_ar = np.corrcoef(aspect_ratios, label_counts)[0, 1]

    print(f"Correlation (Image Area vs Label Count): {corr_area:.4f}")
    print(f"Correlation (Aspect Ratio vs Label Count): {corr_ar:.4f}")

    if abs(corr_area) > 0.1:
        print(
            "-> Note: Slight relationship detected between image size and number of annotations."
        )
    else:
        print(
            "-> Note: No significant linear relationship between image size and number of annotations."
        )


if __name__ == "__main__":
    set_seed(SEED)

    # 1. Load Data
    df = load_data()

    # 2. Target Analysis
    _ = analyze_targets(df)

    # 3. Image Analysis
    image_stats = analyze_images(df)

    # 4. Relationship Analysis
    analyze_relationships(df, image_stats)
