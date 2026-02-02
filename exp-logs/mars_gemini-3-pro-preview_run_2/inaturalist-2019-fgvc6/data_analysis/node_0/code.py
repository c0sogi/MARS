import pandas as pd
import numpy as np
import cv2
import os
import random
from pathlib import Path
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def get_super_category(file_path):
    # Path format: train_val2019/{SuperCategory}/{Category}/{ImageId}.jpg
    # or similar. We try to extract the top-level folder after the dataset root.
    parts = file_path.split("/")
    if len(parts) > 2:
        return parts[1]  # Assuming train_val2019/SuperCategory/...
    return "Unknown"


def analyze_targets(df):
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    target_col = "category_id"
    counts = df[target_col].value_counts()
    n_classes = len(counts)

    print(f"Total Samples: {len(df)}")
    print(f"Number of Classes: {n_classes}")

    # Distribution stats
    mean_count = counts.mean()
    std_count = counts.std()
    min_count = counts.min()
    max_count = counts.max()

    print(f"Class Count Mean: {mean_count:.4f}")
    print(f"Class Count Std: {std_count:.4f}")
    print(f"Class Count Min: {min_count}")
    print(f"Class Count Max: {max_count}")

    # Imbalance
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top 5 and Bottom 5 classes
    print("\nTop 5 Most Frequent Classes:")
    print(counts.head(5).to_string())
    print("\nTop 5 Least Frequent Classes:")
    print(counts.tail(5).to_string())

    # Check for skewness in class distribution (counts)
    skewness = counts.skew()
    print(f"\nSkewness of Class Counts: {skewness:.4f}")
    if skewness > 1:
        print("Observation: The class distribution is highly skewed.")
    elif skewness < -1:
        print("Observation: The class distribution is highly skewed (negative).")
    else:
        print("Observation: The class distribution is moderately skewed or symmetric.")


def analyze_images(df, input_dir, sample_size=3000):
    print("\nSECTION 2: INPUT DATA ANALYSIS (IMAGE)")

    # Sampling for efficiency
    if len(df) > sample_size:
        sampled_df = df.sample(n=sample_size, random_state=42)
    else:
        sampled_df = df

    print(f"Analyzing a random sample of {len(sampled_df)} images for stats...")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # Pixel stats accumulators
    # We will compute mean/std per channel (B, G, R)
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    missing_files = 0

    for _, row in sampled_df.iterrows():
        file_path = input_dir / row["file_name"]

        # cv2.imread loads as BGR
        img = cv2.imread(str(file_path))

        if img is None:
            missing_files += 1
            continue

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) > 2 else 1

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channel_counts.append(c)

        # Pixel stats
        if c == 3:
            # Flatten H*W, 3
            pixels = img.reshape(-1, 3) / 255.0
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels**2).sum(axis=0)
            total_pixel_count += h * w
        elif c == 1:
            # Treat grayscale as repeating 3 channels for global stats or handle separately
            # Here we assume RGB model target, so we treat as 3 channels
            pixels = img.reshape(-1) / 255.0
            s = pixels.sum()
            sq = (pixels**2).sum()
            channel_sum += np.array([s, s, s])
            channel_sq_sum += np.array([sq, sq, sq])
            total_pixel_count += h * w

    if total_pixel_count > 0:
        global_mean = channel_sum / total_pixel_count
        # E[X^2] - (E[X])^2
        global_std = np.sqrt((channel_sq_sum / total_pixel_count) - (global_mean**2))

        # Convert BGR to RGB for reporting
        rgb_mean = global_mean[::-1]
        rgb_std = global_std[::-1]
    else:
        rgb_mean = np.zeros(3)
        rgb_std = np.zeros(3)

    # Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Missing/Unreadable Files in Sample: {missing_files}")

    print("\n--- Dimensions ---")
    print(
        f"Width: Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}, Min={aspect_ratios.min():.4f}, Max={aspect_ratios.max():.4f}"
    )

    # Aspect Ratio Distribution buckets
    ar_bins = [0, 0.9, 1.1, 100]
    ar_labels = ["Portrait (<0.9)", "Square (0.9-1.1)", "Landscape (>1.1)"]
    ar_cats = pd.cut(aspect_ratios, bins=ar_bins, labels=ar_labels)
    print("\nAspect Ratio Distribution:")
    print(ar_cats.value_counts(normalize=True).to_string())

    print("\n--- Channels ---")
    unique_channels, counts_channels = np.unique(channel_counts, return_counts=True)
    for ch, cnt in zip(unique_channels, counts_channels):
        print(f"Channels {ch}: {cnt} images ({cnt/len(channel_counts)*100:.2f}%)")

    print("\n--- Pixel Stats (Normalized 0-1, RGB Order) ---")
    print(f"Mean: R={rgb_mean[0]:.4f}, G={rgb_mean[1]:.4f}, B={rgb_mean[2]:.4f}")
    print(f"Std : R={rgb_std[0]:.4f}, G={rgb_std[1]:.4f}, B={rgb_std[2]:.4f}")

    return sampled_df.copy(), widths, heights, aspect_ratios


def analyze_relationships(df, sampled_df, widths, heights, ars):
    print("\nSECTION 3: FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Super Category Analysis (Meta-feature)
    # Extract super category from file path
    df["super_category"] = df["file_name"].apply(get_super_category)

    print("--- Super Category Distribution ---")
    super_counts = df["super_category"].value_counts()
    print(super_counts.to_string())

    # 2. Relationship between Image Metadata and Target
    # We use the sampled dataframe which corresponds to the widths/heights/ars arrays
    # Add the extracted stats to the sampled_df
    sampled_df["width"] = widths
    sampled_df["height"] = heights
    sampled_df["aspect_ratio"] = ars
    sampled_df["super_category"] = sampled_df["file_name"].apply(get_super_category)

    print("\n--- Meta-Feature vs Target Relationships ---")

    # Does Aspect Ratio vary by Super Category?
    # Group by super category and get mean AR
    ar_by_super = sampled_df.groupby("super_category")["aspect_ratio"].agg(
        ["mean", "std", "count"]
    )
    print("Aspect Ratio by Super Category:")
    print(ar_by_super)

    # ANOVA Test: Is there a significant difference in Aspect Ratio between Super Categories?
    # Filter out groups with < 2 samples for ANOVA
    groups = []
    group_names = []
    for name, group in sampled_df.groupby("super_category"):
        if len(group) > 5:
            groups.append(group["aspect_ratio"].values)
            group_names.append(name)

    if len(groups) > 1:
        f_val, p_val = stats.f_oneway(*groups)
        print(f"\nANOVA One-way Test (Aspect Ratio ~ Super Category):")
        print(f"F-statistic: {f_val:.4f}, p-value: {p_val:.4f}")
        if p_val < 0.05:
            print(
                "Result: Significant difference in Aspect Ratios between super categories."
            )
        else:
            print(
                "Result: No significant difference in Aspect Ratios between super categories."
            )

    # Correlation between Image Size (Pixels) and Category ID?
    # (Checking if 'later' categories in the ID list have larger/smaller images - unlikely but checks structure)
    sampled_df["num_pixels"] = sampled_df["width"] * sampled_df["height"]
    corr_size_cat, p_corr = stats.pearsonr(
        sampled_df["category_id"], sampled_df["num_pixels"]
    )
    print(
        f"\nCorrelation between Category ID and Image Size (Pixels): {corr_size_cat:.4f} (p={p_corr:.4f})"
    )

    # Check if image size varies by super category
    size_by_super = sampled_df.groupby("super_category")["num_pixels"].mean()
    print("\nAverage Image Size (pixels) by Super Category:")
    print(size_by_super.sort_values(ascending=False))


def main():
    set_seed()

    input_dir = Path("./input")
    metadata_path = Path("./metadata/train.csv")

    if not metadata_path.exists():
        print("Error: Metadata file not found.")
        return

    print("Loading Metadata...")
    df = pd.read_csv(metadata_path)

    # Run Analyses
    analyze_targets(df)

    # Image analysis returns sampled data and extracted features for relationship analysis
    sampled_df, widths, heights, ars = analyze_images(df, input_dir, sample_size=3000)

    analyze_relationships(df, sampled_df, widths, heights, ars)

    print("\nEDA Completed.")


if __name__ == "__main__":
    main()
