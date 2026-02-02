import os
import pandas as pd
import numpy as np
import cv2
import concurrent.futures
from collections import Counter
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for heavy pixel analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    """Loads the training metadata."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    # Handle potential NaN in attribute_ids by converting to empty string
    df["attribute_ids"] = df["attribute_ids"].fillna("")
    return df


def analyze_targets(df):
    """Analyzes the distribution of the target variable (multi-label attributes)."""
    print("2. TARGET VARIABLE ANALYSIS")

    # Parse attribute_ids
    # Convert space-separated string to list of integers
    all_labels = []
    label_counts_per_image = []

    for item in df["attribute_ids"]:
        if item.strip() == "":
            labels = []
        else:
            labels = [int(x) for x in item.split()]
        all_labels.extend(labels)
        label_counts_per_image.append(len(labels))

    label_counts_per_image = np.array(label_counts_per_image)
    total_images = len(df)
    unique_labels = set(all_labels)
    num_unique_labels = len(unique_labels)

    # Frequency distribution
    label_counter = Counter(all_labels)
    frequencies = list(label_counter.values())

    print(f"Task Type: Multi-label Classification")
    print(f"Total Samples: {total_images}")
    print(f"Total Unique Labels: {num_unique_labels}")

    # Label Cardinality (Labels per image)
    print(
        f"Labels per Image (Cardinality) - Mean: {np.mean(label_counts_per_image):.4f}"
    )
    print(f"Labels per Image (Cardinality) - Std: {np.std(label_counts_per_image):.4f}")
    print(f"Labels per Image (Cardinality) - Min: {np.min(label_counts_per_image):.4f}")
    print(f"Labels per Image (Cardinality) - Max: {np.max(label_counts_per_image):.4f}")

    # Class Balance
    if frequencies:
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        mean_freq = np.mean(frequencies)

        print(f"Label Frequency - Min: {min_freq} ({min_freq/total_images:.4f}%)")
        print(f"Label Frequency - Max: {max_freq} ({max_freq/total_images:.4f}%)")
        print(f"Label Frequency - Mean: {mean_freq:.4f}")

        # Top 5 and Bottom 5
        most_common = label_counter.most_common(5)
        least_common = label_counter.most_common()[:-6:-1]

        print(
            f"Top 5 Common Labels (ID: Count): {', '.join([f'{k}: {v}' for k, v in most_common])}"
        )
        print(
            f"Bottom 5 Rare Labels (ID: Count): {', '.join([f'{k}: {v}' for k, v in least_common])}"
        )

        # Imbalance Ratio
        print(f"Imbalance Ratio (Max/Min): {max_freq/min_freq:.4f}")
    else:
        print("No labels found.")

    return label_counts_per_image


def process_image(file_info):
    """
    Worker function to process a single image.
    Returns: (width, height, channels, mean_pixel_val, std_pixel_val, aspect_ratio)
    """
    idx, rel_path = file_info
    full_path = os.path.join(INPUT_DIR, rel_path)

    try:
        # Read image
        # IMREAD_UNCHANGED to detect alpha channels or grayscale correctly
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return None

        # Dimensions
        shape = img.shape
        h = shape[0]
        w = shape[1]

        if len(shape) == 2:
            c = 1  # Grayscale
        else:
            c = shape[2]

        # Aspect Ratio
        ar = w / h if h > 0 else 0

        # Pixel Stats (normalize to 0-1 for calculation)
        img_norm = img.astype(np.float32) / 255.0
        mean_val = np.mean(img_norm)
        std_val = np.std(img_norm)

        return (w, h, c, mean_val, std_val, ar)

    except Exception:
        return None


def analyze_images(df):
    """Analyzes image properties using a sample of the dataset."""
    print("\n3. INPUT DATA ANALYSIS (IMAGE)")

    # Sample data for efficiency
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    print(f"Analyzing a sample of {len(sample_df)} images for pixel statistics...")

    # Prepare inputs for parallel processing
    # df has 'file_path' column
    tasks = list(zip(sample_df.index, sample_df["file_path"]))

    widths = []
    heights = []
    channels = []
    means = []
    stds = []
    aspect_ratios = []

    # Use ProcessPoolExecutor for CPU-bound image processing
    # Adjust max_workers based on available vCPUs (12 available)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(process_image, tasks))

    # Filter out None results (failed reads)
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("Error: Could not read any images.")
        return None

    for w, h, c, m, s, ar in valid_results:
        widths.append(w)
        heights.append(h)
        channels.append(c)
        means.append(m)
        stds.append(s)
        aspect_ratios.append(ar)

    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    means = np.array(means)
    stds = np.array(stds)

    # Dimensions
    print("Dimensions:")
    print(
        f"Width - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )

    # Aspect Ratios
    print("Aspect Ratios (Width/Height):")
    print(f"Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}")

    # Channels
    c_counts = Counter(channels)
    print("Channels Distribution:")
    for c, count in c_counts.items():
        print(f"{c} Channels: {count} images ({count/len(valid_results)*100:.2f}%)")

    # Pixel Stats
    print("Pixel Statistics (Normalized 0-1):")
    print(f"Global Mean: {np.mean(means):.4f}")
    print(
        f"Global Std: {np.mean(stds):.4f}"
    )  # Average of stds gives a sense of contrast

    # Return meta-features for relationship analysis
    # We need to map these back to the sampled dataframe indices for correlation
    # Since we iterated over sample_df and results are ordered, we can assign them back
    # Note: We must handle skipped (None) images

    # Re-construct a dataframe for the valid samples
    valid_indices = [tasks[i][0] for i, r in enumerate(results) if r is not None]

    meta_df = pd.DataFrame(
        {
            "index": valid_indices,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "pixel_mean": means,
            "pixel_std": stds,
        }
    ).set_index("index")

    return meta_df


def analyze_relationships(df, meta_df, label_counts):
    """Analyzes relationships between image meta-features and target variables."""
    print("\n4. FEATURE/SIGNAL RELATIONSHIPS")

    # Join meta_df with original df to get label info for the sampled images
    # label_counts is an array aligned with df, so we add it to df first
    df["num_labels"] = label_counts

    # Inner join to keep only sampled images
    analysis_df = df.join(meta_df, how="inner")

    if analysis_df.empty:
        print("No overlapping data for relationship analysis.")
        return

    print("Unstructured (Meta-Feature) Relationships:")

    # 1. Correlation between Image Size (Area) and Number of Labels
    analysis_df["area"] = analysis_df["width"] * analysis_df["height"]

    corr_area_labels = analysis_df["area"].corr(analysis_df["num_labels"])
    print(f"Correlation (Image Area vs Num Labels): {corr_area_labels:.4f}")

    # 2. Correlation between Aspect Ratio and Number of Labels
    corr_ar_labels = analysis_df["aspect_ratio"].corr(analysis_df["num_labels"])
    print(f"Correlation (Aspect Ratio vs Num Labels): {corr_ar_labels:.4f}")

    # 3. Correlation between Pixel Complexity (Std) and Number of Labels
    # Higher std often implies more visual clutter/detail
    corr_std_labels = analysis_df["pixel_std"].corr(analysis_df["num_labels"])
    print(f"Correlation (Pixel Contrast/Std vs Num Labels): {corr_std_labels:.4f}")

    # 4. Check if larger images tend to have specific characteristics
    # Split into 'Large' and 'Small' based on median area
    median_area = analysis_df["area"].median()
    large_imgs = analysis_df[analysis_df["area"] > median_area]
    small_imgs = analysis_df[analysis_df["area"] <= median_area]

    print(
        f"Mean Labels in Large Images (>Median Area): {large_imgs['num_labels'].mean():.4f}"
    )
    print(
        f"Mean Labels in Small Images (<=Median Area): {small_imgs['num_labels'].mean():.4f}"
    )


def main():
    set_seed(SEED)

    try:
        # 1. Load Data
        train_df = load_data()

        # 2. Target Analysis
        label_counts = analyze_targets(train_df)

        # 3. Image Analysis (on sample)
        meta_df = analyze_images(train_df)

        # 4. Relationship Analysis
        if meta_df is not None:
            analyze_relationships(train_df, meta_df, label_counts)

        print("\nEDA Completed Successfully.")

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
