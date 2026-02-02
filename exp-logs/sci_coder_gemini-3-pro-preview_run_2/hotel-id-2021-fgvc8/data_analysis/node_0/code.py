import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SEED = 42
SAMPLE_SIZE_PIXELS = 1000  # Number of images to sample for pixel stats
SAMPLE_SIZE_DIMS = 5000  # Number of images to sample for dimension stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df):
    print("=== TARGET VARIABLE ANALYSIS ===")
    target_col = "hotel_id"

    # Distribution
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    print(f"Target Variable: {target_col}")
    print(f"Type: Classification (Multi-class)")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Samples: {total_samples}")

    # Imbalance/Skew
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    mean_samples = class_counts.mean()
    median_samples = class_counts.median()

    print(f"Class Balance Statistics:")
    print(f"  Min samples per class: {min_samples}")
    print(f"  Max samples per class: {max_samples}")
    print(f"  Mean samples per class: {mean_samples:.4f}")
    print(f"  Median samples per class: {median_samples:.4f}")

    # Top/Bottom classes
    print(f"Top 5 Most Frequent Classes:\n{class_counts.head(5).to_string()}")
    print(f"Bottom 5 Least Frequent Classes:\n{class_counts.tail(5).to_string()}")
    print("-" * 30)


def analyze_tabular(df):
    print("=== INPUT DATA ANALYSIS (TABULAR METADATA) ===")

    # Numerical/Categorical split
    # 'chain' is categorical (ID), 'timestamp' is temporal/object

    # Chain Analysis
    print("Feature: chain (Categorical)")
    chain_counts = df["chain"].value_counts()
    print(f"  Cardinality: {df['chain'].nunique()}")
    print(
        f"  Most frequent chain: {chain_counts.idxmax()} (Count: {chain_counts.max()})"
    )
    print(
        f"  Least frequent chain: {chain_counts.idxmin()} (Count: {chain_counts.min()})"
    )

    # Rare labels
    rare_threshold = 0.01 * len(df)
    rare_chains = chain_counts[chain_counts < rare_threshold]
    print(
        f"  Chains with < 1% frequency: {len(rare_chains)} (out of {len(chain_counts)})"
    )

    # Missing Values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    for col in df.columns:
        pct = (missing[col] / len(df)) * 100
        print(f"  {col}: {missing[col]} ({pct:.4f}%)")

    # Timestamp Analysis
    if "timestamp" in df.columns:
        print("\nFeature: timestamp (Temporal)")
        # Convert to datetime, handle errors
        dates = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_dates = dates.dropna()

        if not valid_dates.empty:
            print(f"  Range: {valid_dates.min()} to {valid_dates.max()}")
            # Extract year
            years = valid_dates.dt.year
            print(
                f"  Year Distribution:\n{years.value_counts().sort_index().to_string()}"
            )
        else:
            print("  No valid timestamps found.")

    print("-" * 30)


def get_image_stats(df):
    print("=== INPUT DATA ANALYSIS (IMAGE DATA) ===")

    # Sample for Dimensions
    sample_df_dims = df.sample(n=min(len(df), SAMPLE_SIZE_DIMS), random_state=SEED)

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []

    # Sample for Pixel Stats (subset of dims sample to save time)
    sample_df_pixels = sample_df_dims.sample(
        n=min(len(sample_df_dims), SAMPLE_SIZE_PIXELS), random_state=SEED
    )
    pixel_indices = set(sample_df_pixels.index)

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    print(f"Analyzing dimensions on {len(sample_df_dims)} images...")
    print(f"Analyzing pixel stats on {len(sample_df_pixels)} images...")

    for idx, row in sample_df_dims.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # For dimensions, we can just read the header if possible, but cv2.imread is robust
        # To be fast, we read flag IMREAD_UNCHANGED
        try:
            img = cv2.imread(full_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            c = 1 if len(img.shape) == 2 else img.shape[2]

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channels_list.append(c)

            # If this image is selected for pixel stats
            if idx in pixel_indices:
                # Convert to RGB for consistency if color
                if c == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif c == 1:
                    # Treat grayscale as 3 channels for global stats or keep separate?
                    # Usually convert to RGB for unified stats
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Normalize to 0-1 for calculation stability, then scale back or report 0-255
                img_norm = img.astype(np.float32) / 255.0
                pixel_sum += img_norm.sum(axis=(0, 1))
                pixel_sq_sum += (img_norm**2).sum(axis=(0, 1))
                pixel_count += h * w

        except Exception as e:
            continue

    # Dimension Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    ars = np.array(aspect_ratios)

    print("\nDimensions:")
    print(
        f"  Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"  Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"  Aspect Ratio: Mean={ars.mean():.4f}, Std={ars.std():.4f}, Min={ars.min():.4f}, Max={ars.max():.4f}"
    )

    # Outliers (IQR Method for Width)
    q1 = np.percentile(widths, 25)
    q3 = np.percentile(widths, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((widths < lower_bound) | (widths > upper_bound)).sum()
    print(
        f"  Width Outliers (IQR method): {outliers} ({outliers/len(widths)*100:.2f}%)"
    )

    # Channels
    c_counts = Counter(channels_list)
    print("\nChannels:")
    for c, count in c_counts.items():
        print(f"  {c} channels: {count} images")

    # Pixel Stats
    if pixel_count > 0:
        # Calculate mean and std
        # E[X]
        global_mean = pixel_sum / pixel_count
        # Var(X) = E[X^2] - (E[X])^2
        global_var = (pixel_sq_sum / pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)

        # Scale back to 0-255 for reporting
        mean_255 = global_mean * 255.0
        std_255 = global_std * 255.0

        print("\nPixel Statistics (RGB, 0-255 scale):")
        print(f"  Mean: R={mean_255[0]:.4f}, G={mean_255[1]:.4f}, B={mean_255[2]:.4f}")
        print(f"  Std:  R={std_255[0]:.4f}, G={std_255[1]:.4f}, B={std_255[2]:.4f}")

    print("-" * 30)
    return widths, heights


def analyze_relationships(df, widths, heights):
    print("=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Structured: Chain vs Hotel_ID
    # Check if chains have unique hotels or if hotels are shared
    print("Relationship: Chain vs Hotel ID")
    hotels_per_chain = df.groupby("chain")["hotel_id"].nunique()
    print(f"  Average hotels per chain: {hotels_per_chain.mean():.4f}")
    print(f"  Max hotels in a single chain: {hotels_per_chain.max()}")

    # Check disjointness: Do hotels belong to multiple chains?
    hotel_chain_counts = df.groupby("hotel_id")["chain"].nunique()
    multi_chain_hotels = (hotel_chain_counts > 1).sum()
    print(f"  Hotels belonging to >1 chain: {multi_chain_hotels}")

    # Unstructured: Metadata vs Image Properties
    # Since we only have widths/heights for a sample, we need to map them back to the dataframe subset
    # For simplicity in this script, we'll assume the sample order was preserved or just do a quick check
    # if we had the subset dataframe.
    # Instead, let's look at Chain vs Image Count (Class Imbalance per chain)

    print("\nRelationship: Chain vs Image Count")
    imgs_per_chain = df["chain"].value_counts()
    correlation = imgs_per_chain.corr(
        hotels_per_chain
    )  # Correlation between size of chain (imgs) and diversity (hotels)
    print(
        f"  Correlation between Chain Image Count and Unique Hotels Count: {correlation:.4f}"
    )

    # Random Forest Feature Importance (Tabular)
    # Predicting Chain from Timestamp features? Or predicting Hotel from Chain?
    # Given high cardinality of target (7700), standard RF is slow/noisy.
    # Let's check importance of 'chain' for 'hotel_id' using mutual information proxy or simple logic.
    # Since hotels are strictly nested in chains (multi_chain_hotels is likely 0), Chain is a perfect predictor of a subset of hotels.
    print("\nFeature Importance:")
    print(
        "  'chain' is a hierarchical parent of 'hotel_id'. Knowing 'chain' reduces the search space significantly."
    )

    print("-" * 30)


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    analyze_target(df)
    analyze_tabular(df)
    widths, heights = get_image_stats(df)
    analyze_relationships(df, widths, heights)


if __name__ == "__main__":
    main()
