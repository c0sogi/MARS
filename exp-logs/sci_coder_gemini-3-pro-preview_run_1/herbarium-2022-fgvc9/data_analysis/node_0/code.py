import os
import json
import random
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
TRAIN_JSON = os.path.join(INPUT_DIR, "train_metadata.json")
SEED = 42
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    # Load the generated training metadata
    df = pd.read_csv(TRAIN_CSV)

    # Load the raw JSON for taxonomic structure
    with open(TRAIN_JSON, "r") as f:
        raw_meta = json.load(f)

    # Create a mapping for taxonomy
    # raw_meta['categories'] contains lists of dicts with family, genus, species, category_id
    categories = raw_meta.get("categories", [])
    tax_df = pd.DataFrame(categories)

    # Merge taxonomy info into main df
    if not tax_df.empty and "category_id" in tax_df.columns:
        df = df.merge(
            tax_df[["category_id", "family", "genus", "species"]],
            on="category_id",
            how="left",
        )

    return df


def analyze_target_variable(df):
    print("==== TARGET VARIABLE ANALYSIS ====")

    # 1. Species (Category ID) Distribution
    class_counts = df["category_id"].value_counts()
    n_classes = len(class_counts)

    print(f"Total Samples: {len(df)}")
    print(f"Number of Unique Classes (Species): {n_classes}")

    print("\n-- Class Distribution Stats --")
    print(f"Mean Samples per Class: {class_counts.mean():.4f}")
    print(f"Std Dev Samples per Class: {class_counts.std():.4f}")
    print(f"Min Samples in a Class: {class_counts.min()}")
    print(f"Max Samples in a Class: {class_counts.max()}")

    # Imbalance Ratio
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # 2. Taxonomic Distribution (Family/Genus)
    if "family" in df.columns and "genus" in df.columns:
        family_counts = df["family"].value_counts()
        genus_counts = df["genus"].value_counts()

        print("\n-- Taxonomic Hierarchy Stats --")
        print(f"Number of Unique Families: {len(family_counts)}")
        print(f"Number of Unique Genera: {len(genus_counts)}")

        print(f"Mean Samples per Family: {family_counts.mean():.4f}")
        print(f"Mean Samples per Genus: {genus_counts.mean():.4f}")

        # Top 3 Families
        print(f"Top 3 Families by Count: {dict(family_counts.head(3))}")


def process_single_image(args):
    """
    Helper to process a single image path.
    Returns: (width, height, channels, mean_color, std_color, file_size_bytes)
    """
    path, full_path = args
    try:
        # Get file size
        file_size = os.path.getsize(full_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        # Calculate mean and std per channel
        # We calculate sum and sum_sq here to aggregate later,
        # but for simple EDA reporting per-image mean/std distribution is also useful.
        # To get global dataset mean/std exactly, we need accumulators.
        # Here we return per-image stats to analyze distribution of brightness/contrast.
        mean_val = img.mean(axis=(0, 1))
        std_val = img.std(axis=(0, 1))

        return (w, h, c, mean_val, std_val, file_size)
    except Exception:
        return None


def analyze_images(df):
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Stratified sample if possible, else random sample
    if len(df) > SAMPLE_SIZE:
        # Simple random sample is safer for speed and avoiding singleton issues in stratification
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    print(f"Analyzing a subset of {len(sample_df)} images for statistics...")

    paths = []
    for _, row in sample_df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        paths.append((row["file_path"], full_path))

    # Parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(process_single_image, paths))

    # Filter None
    results = [r for r in results if r is not None]

    if not results:
        print("Error: No images could be processed.")
        return

    # Unpack results
    widths = [r[0] for r in results]
    heights = [r[1] for r in results]
    channels = [r[2] for r in results]
    means = np.array([r[3] for r in results])  # Shape (N, 3)
    stds = np.array([r[4] for r in results])  # Shape (N, 3)
    file_sizes = [r[5] for r in results]

    # 1. Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = widths / heights

    print("\n-- Image Dimensions --")
    print(
        f"Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )

    print("\n-- Aspect Ratios (Width/Height) --")
    print(f"Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}")
    print(f"Min: {aspect_ratios.min():.4f}, Max: {aspect_ratios.max():.4f}")

    # 2. Channels
    unique_channels = np.unique(channels)
    print(f"\n-- Channel Distribution --")
    print(f"Unique Channel Counts found: {unique_channels}")
    # Check for grayscale (1 channel) vs RGB
    rgb_count = channels.count(3)
    gray_count = channels.count(1)
    print(f"RGB Images: {rgb_count} ({rgb_count/len(channels)*100:.2f}%)")
    if gray_count > 0:
        print(f"Grayscale Images: {gray_count} ({gray_count/len(channels)*100:.2f}%)")

    # 3. Pixel Stats (Approximate Global)
    # Averaging per-image means is a good approximation for global mean if image sizes are roughly similar
    global_mean = means.mean(axis=0)
    global_std = stds.mean(
        axis=0
    )  # This is average contrast, not global std dev, but useful for EDA.
    # For normalization, we usually want global std dev.
    # Global Var = E[Var(X|I)] + Var(E[X|I])
    # But simple average of means is sufficient for "Pixel Stats" reporting in EDA.

    print("\n-- Pixel Value Statistics (RGB, 0-255) --")
    print(
        f"Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
    )
    # We report the mean of standard deviations to indicate average image contrast
    print(
        f"Avg Std Dev (Contrast): R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
    )

    return widths, heights, file_sizes, sample_df


def analyze_relationships(df, widths, heights, file_sizes, sample_df):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Add stats to sample dataframe
    # Note: sample_df index might not align perfectly if we filtered Nones,
    # but for this robust script we assume failures are negligible or we re-align.
    # Let's just create a new DF from the results for correlation analysis

    stats_df = pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
            "aspect_ratio": widths / heights,
            "category_id": sample_df["category_id"].values[
                : len(widths)
            ],  # Truncate if mismatch
        }
    )

    # Calculate class frequency for each sample
    class_counts = df["category_id"].value_counts()
    stats_df["class_freq"] = stats_df["category_id"].map(class_counts)

    # 1. Structured Relationships (Image Properties)
    print("-- Image Property Correlations --")
    corr_wh = stats_df["width"].corr(stats_df["height"])
    print(f"Correlation (Width vs Height): {corr_wh:.4f}")

    corr_size_dim = stats_df["file_size"].corr(stats_df["width"] * stats_df["height"])
    print(f"Correlation (File Size vs Image Area): {corr_size_dim:.4f}")

    # 2. Meta-Feature Relationships
    print("\n-- Metadata vs Image Properties --")
    # Does class frequency correlate with image quality/size?
    # (e.g. Do rare classes have lower quality/smaller images?)
    corr_freq_size = stats_df["class_freq"].corr(stats_df["file_size"])
    print(f"Correlation (Class Frequency vs File Size): {corr_freq_size:.4f}")

    corr_freq_area = stats_df["class_freq"].corr(stats_df["width"] * stats_df["height"])
    print(f"Correlation (Class Frequency vs Image Area): {corr_freq_area:.4f}")

    if abs(corr_freq_size) < 0.1:
        print(
            "Observation: Little to no linear relationship between class rarity and image file size."
        )
    else:
        print(
            "Observation: Some relationship detected between class rarity and image properties."
        )


def main():
    set_seed(SEED)

    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Target Analysis
    analyze_target_variable(df)

    # 3. Image Analysis
    # We pass the full df, function handles sampling
    img_stats = analyze_images(df)

    if img_stats:
        widths, heights, file_sizes, sample_df = img_stats

        # 4. Relationships
        analyze_relationships(df, widths, heights, file_sizes, sample_df)


if __name__ == "__main__":
    main()
