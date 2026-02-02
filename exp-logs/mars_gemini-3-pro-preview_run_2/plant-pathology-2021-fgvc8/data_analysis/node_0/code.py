import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations
from collections import Counter

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SAMPLE_SIZE = 2000  # Number of images to sample for pixel/dimension analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    df = pd.read_csv(METADATA_PATH)
    # Construct full paths relative to the working directory context
    # The metadata file_path is relative to ./input
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))
    return df


def analyze_targets(df):
    print("2. TARGET VARIABLE ANALYSIS")

    # Parse labels (space delimited)
    df["label_list"] = df["labels"].apply(lambda x: x.split())

    # Use MultiLabelBinarizer to get a binary matrix
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["label_list"])
    classes = mlb.classes_

    # 1. Distribution of individual labels
    label_counts = np.sum(y, axis=0)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Unique Classes: {len(classes)}")
    print("\nClass Distribution (Frequency & Percentage):")

    # Create a dataframe for sorting
    class_stats = pd.DataFrame(
        {
            "Label": classes,
            "Count": label_counts,
            "Frequency": label_counts / total_samples,
        }
    ).sort_values("Count", ascending=False)

    for _, row in class_stats.iterrows():
        print(f"  {row['Label']:<20}: {int(row['Count']):>5} ({row['Frequency']:.4f})")

    # 2. Imbalance
    max_count = class_stats["Count"].max()
    min_count = class_stats["Count"].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"\nClass Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # 3. Label Cardinality (Labels per image)
    df["label_count"] = df["label_list"].apply(len)
    cardinality_counts = df["label_count"].value_counts().sort_index()

    print("\nLabels per Image (Cardinality):")
    for k, v in cardinality_counts.items():
        print(f"  {k} label(s): {v:>5} samples ({v/total_samples:.4f})")

    avg_labels = df["label_count"].mean()
    print(f"  Average labels per image: {avg_labels:.4f}")

    return mlb, y, class_stats


def analyze_images(df):
    print("\n3. INPUT DATA ANALYSIS (IMAGE)")

    # Sample data for image analysis to save time
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    print(f"Analyzing a sample of {len(sample_df)} images...")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []

    # Accumulators for pixel stats (Welford's or simple sum for mean)
    # Using simple sum for mean, and sum of squares for std
    # To avoid overflow, use float64
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    valid_images = 0

    for _, row in sample_df.iterrows():
        path = row["full_path"]

        # File size
        try:
            f_size = os.path.getsize(path)
            file_sizes.append(f_size)
        except OSError:
            continue

        # Read Image
        img = cv2.imread(path)
        if img is None:
            continue

        # OpenCV loads as BGR, convert to RGB for standard reporting
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channels.append(c)

        # Pixel stats
        # Flatten spatial dimensions: (H*W, 3)
        pixels = img.reshape(-1, 3) / 255.0
        pixel_sum += pixels.sum(axis=0)
        pixel_sq_sum += (pixels**2).sum(axis=0)
        pixel_count += pixels.shape[0]

        valid_images += 1

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print("\nImage Dimensions:")
    print(
        f"  Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"  Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    # Channel Analysis
    unique_channels = np.unique(channels)
    print(f"\nChannels Distribution: {unique_channels}")
    if len(unique_channels) == 1 and unique_channels[0] == 3:
        print("  All images are RGB (3 channels).")

    # Pixel Stats
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        # std = sqrt(E[x^2] - (E[x])^2)
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))

        print("\nPixel Value Statistics (Normalized 0-1):")
        print(
            f"  Mean (R, G, B): [{rgb_mean[0]:.4f}, {rgb_mean[1]:.4f}, {rgb_mean[2]:.4f}]"
        )
        print(
            f"  Std  (R, G, B): [{rgb_std[0]:.4f}, {rgb_std[1]:.4f}, {rgb_std[2]:.4f}]"
        )

    return sample_df, file_sizes, widths, heights


def analyze_relationships(df, mlb, y, sample_df, file_sizes, widths, heights):
    print("\n4. FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Co-occurrence Matrix (Label Correlations)
    print("Label Co-occurrence (Top Pairs):")

    # Calculate co-occurrence matrix: Y.T @ Y
    co_occurrence = np.dot(y.T, y)

    # Set diagonal to 0 to ignore self-occurrence for finding pairs
    np.fill_diagonal(co_occurrence, 0)

    classes = mlb.classes_
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            count = co_occurrence[i, j]
            if count > 0:
                pairs.append((classes[i], classes[j], count))

    # Sort by count
    pairs.sort(key=lambda x: x[2], reverse=True)

    for c1, c2, count in pairs[:5]:
        print(f"  {c1} + {c2}: {count} occurrences")

    if not pairs:
        print("  No multi-label co-occurrences found.")

    # 2. Meta-feature vs Target Relationships
    # We use the sampled dataframe for this to match the image stats we collected
    # We need to map the sample_df back to the targets

    # Create a subset of targets corresponding to the sample
    sample_indices = sample_df.index
    y_sample = y[sample_indices]

    # Meta-features
    meta_features = pd.DataFrame(
        {
            "file_size": file_sizes,
            "width": widths,
            "height": heights,
            "aspect_ratio": widths / heights,
        }
    )

    print("\nMeta-Feature vs Target Correlations (Point-Biserial):")
    print("  (Correlation between image property and presence of disease)")

    # Calculate correlation for each class
    # We iterate over classes and calculate correlation with meta-features
    results = []

    for i, class_name in enumerate(classes):
        # Binary vector for this class
        class_vec = y_sample[:, i]

        # Skip if class variance is 0 in sample
        if np.std(class_vec) == 0:
            continue

        row = {"Class": class_name}
        for col in meta_features.columns:
            # Pearson correlation between binary class and continuous meta-feature
            corr = np.corrcoef(class_vec, meta_features[col])[0, 1]
            row[col] = corr
        results.append(row)

    corr_df = pd.DataFrame(results).set_index("Class")

    # Print top correlations (magnitude)
    # Stack and sort
    if not corr_df.empty:
        stacked_corr = corr_df.stack().reset_index()
        stacked_corr.columns = ["Class", "Feature", "Correlation"]
        stacked_corr["AbsCorr"] = stacked_corr["Correlation"].abs()
        top_corrs = stacked_corr.sort_values("AbsCorr", ascending=False).head(5)

        for _, row in top_corrs.iterrows():
            print(
                f"  {row['Class']:<20} vs {row['Feature']:<12}: {row['Correlation']:.4f}"
            )
    else:
        print("  Insufficient variance in sample to calculate correlations.")


def main():
    set_seed(SEED)

    # 1. Load Data
    df = load_data()

    # 2. Target Analysis
    mlb, y, class_stats = analyze_targets(df)

    # 3. Image Analysis
    sample_df, file_sizes, widths, heights = analyze_images(df)

    # 4. Relationships
    analyze_relationships(df, mlb, y, sample_df, file_sizes, widths, heights)


if __name__ == "__main__":
    main()
