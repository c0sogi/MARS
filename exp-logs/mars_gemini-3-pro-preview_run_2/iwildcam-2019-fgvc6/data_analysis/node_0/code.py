import os
import pandas as pd
import numpy as np
import cv2
import concurrent.futures
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
RANDOM_SEED = 42
SAMPLE_SIZE_IMAGES = 10000  # Number of images to sample for pixel stats to fit in time
N_WORKERS = 12


def set_seed(seed):
    np.random.seed(seed)


def analyze_target(df):
    print("==== TARGET VARIABLE ANALYSIS ====")
    target_col = "Category"

    # Distribution
    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print(f"Target Column: {target_col}")
    print(f"Number of Classes: {len(counts)}")
    print(
        f"Most Frequent Class: {counts.idxmax()} (Count: {counts.max()}, {proportions.max()*100:.2f}%)"
    )
    print(
        f"Least Frequent Class: {counts.idxmin()} (Count: {counts.min()}, {proportions.min()*100:.2f}%)"
    )

    # Imbalance
    imbalance_ratio = counts.max() / counts.min()
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top 5 Classes
    print("\nTop 5 Classes Distribution:")
    for cat, prop in proportions.head(5).items():
        print(f"  Class {cat}: {prop:.4f}")
    print("-" * 30)


def analyze_tabular_metadata(df):
    print("==== TABULAR DATA ANALYSIS (METADATA) ====")

    # Identify columns
    # We know Id, Category, file_path are specific. Let's look at others.
    exclude_cols = ["Id", "Category", "file_path", "file_name"]
    potential_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Metadata Columns Available: {potential_cols}")

    for col in potential_cols:
        # Check if categorical or numerical
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numerical analysis
            stats = df[col].describe()
            print(f"\nColumn: {col} (Numerical)")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")

            # Outliers (IQR)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (
                (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            ).sum()
            print(f"  Outliers (IQR method): {outliers}")

        else:
            # Categorical analysis
            unique_count = df[col].nunique()
            print(f"\nColumn: {col} (Categorical)")
            print(f"  Cardinality: {unique_count}")

            # Flag high cardinality
            if unique_count > 50:
                print(f"  [FLAG] High Cardinality (> 50 categories)")

            # Missing values
            nans = df[col].isna().sum()
            nan_pct = (nans / len(df)) * 100
            print(f"  Missing Values: {nans} ({nan_pct:.4f}%)")

    # Global Missing Value Check
    print("\nGlobal Missing Values:")
    missing = df.isna().sum()
    print(missing[missing > 0])
    print("-" * 30)


def process_image(file_info):
    """
    Helper function to process a single image.
    Returns: (width, height, channels, mean_channels, std_channels, is_valid)
    """
    rel_path, _ = file_info
    full_path = os.path.join(INPUT_DIR, rel_path)

    try:
        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Dimensions
        h, w, c = img.shape

        # Pixel Stats (Normalize to 0-1 for calculation)
        img_norm = img.astype(np.float32) / 255.0

        # Calculate mean and var per channel for this image
        # We return sum and sum_sq to aggregate globally later,
        # but for simplicity in this EDA script (approximate global stats),
        # we will return the mean/std of this specific image to distribution analysis.
        # Actually, to get accurate Global Mean/Std, we should return sum and count.

        mean_ch = np.mean(img_norm, axis=(0, 1))  # BGR
        std_ch = np.std(img_norm, axis=(0, 1))  # BGR

        return (w, h, c, mean_ch, std_ch, True)

    except Exception:
        return None


def analyze_images(df):
    print("==== IMAGE DATA ANALYSIS ====")

    # Sampling
    if len(df) > SAMPLE_SIZE_IMAGES:
        sample_df = df.sample(n=SAMPLE_SIZE_IMAGES, random_state=RANDOM_SEED)
    else:
        sample_df = df

    print(f"Analyzing sample of {len(sample_df)} images...")

    paths = sample_df["file_path"].tolist()
    # Dummy second arg for map
    inputs = [(p, 0) for p in paths]

    widths = []
    heights = []
    channels = []
    means = []  # List of arrays
    stds = []  # List of arrays

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(process_image, inputs))

    # Filter None
    valid_results = [r for r in results if r is not None]

    for w, h, c, m, s, _ in valid_results:
        widths.append(w)
        heights.append(h)
        channels.append(c)
        means.append(m)
        stds.append(s)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = widths / heights

    # Dimensions
    print("\nDimensions:")
    print(
        f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels
    print("\nChannels:")
    c_counts = Counter(channels)
    for c_num, count in c_counts.items():
        print(f"  {c_num} Channels: {count} images")

    # Pixel Stats (Global approximation based on sample means)
    # Note: OpenCV loads as BGR
    means = np.array(means)
    stds = np.array(stds)

    global_mean = np.mean(means, axis=0)  # Average of averages
    global_std = np.sqrt(
        np.mean(stds**2 + means**2, axis=0) - global_mean**2
    )  # Approximate global std

    print("\nPixel Statistics (Normalized 0-1, BGR Order):")
    print(
        f"  Mean: B={global_mean[0]:.4f}, G={global_mean[1]:.4f}, R={global_mean[2]:.4f}"
    )
    print(
        f"  Std:  B={global_std[0]:.4f},  G={global_std[1]:.4f},  R={global_std[2]:.4f}"
    )

    # Return stats for relationship analysis
    return valid_results, sample_df.iloc[: len(valid_results)]


def analyze_relationships(df, image_stats_results, image_df_sample):
    print("-" * 30)
    print("==== FEATURE/SIGNAL RELATIONSHIPS ====")

    target_col = "Category"

    # 1. Categorical Metadata vs Target (Mutual Information)
    # Check for 'location'
    if "location" in df.columns:
        # Encode location
        loc_codes = df["location"].astype("category").cat.codes
        target_codes = df[target_col]

        mi = normalized_mutual_info_score(target_codes, loc_codes)
        print(f"\nMutual Information (Location vs Category): {mi:.4f}")
        print("  (Higher value indicates Location is strongly predictive of Species)")

        # Check specific strong correlations (e.g., does a location only have 1 species?)
        loc_counts = df.groupby("location")[target_col].nunique()
        single_species_locs = (loc_counts == 1).sum()
        print(
            f"  Locations with only 1 unique species: {single_species_locs} / {df['location'].nunique()}"
        )

    # 2. Image Features vs Target
    # We need to align the image stats with the sampled dataframe
    # image_stats_results corresponds to image_df_sample rows (assuming no failures, or we filter)

    # Extract areas
    areas = [r[0] * r[1] for r in image_stats_results]  # w * h
    image_df_sample = image_df_sample.copy()
    image_df_sample["img_area"] = areas

    # Correlation between Area and Category (Categorical)
    # We can look at mean area per category
    print("\nImage Area vs Category:")
    mean_area_per_cat = image_df_sample.groupby(target_col)["img_area"].mean()
    print("  Mean Image Area (pixels) for Top 3 Frequent Classes:")
    top_classes = df[target_col].value_counts().head(3).index
    for c in top_classes:
        if c in mean_area_per_cat:
            print(f"    Class {c}: {mean_area_per_cat[c]:.2f}")

    # Check if empty images (Class 0) are significantly different in brightness
    # Extract brightness (mean of means)
    brightness = [np.mean(r[3]) for r in image_stats_results]
    image_df_sample["brightness"] = brightness

    mean_bright_per_cat = image_df_sample.groupby(target_col)["brightness"].mean()
    print("\nImage Brightness vs Category:")
    if 0 in mean_bright_per_cat:
        print(f"    Class 0 (Empty): {mean_bright_per_cat[0]:.4f}")

    non_empty_bright = image_df_sample[image_df_sample[target_col] != 0][
        "brightness"
    ].mean()
    print(f"    Non-Empty Avg:   {non_empty_bright:.4f}")


def main():
    set_seed(RANDOM_SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    analyze_target(df)

    # 2. Tabular/Metadata Analysis
    analyze_tabular_metadata(df)

    # 3. Image Analysis
    image_results, image_sample_df = analyze_images(df)

    # 4. Relationships
    analyze_relationships(df, image_results, image_sample_df)


if __name__ == "__main__":
    main()
