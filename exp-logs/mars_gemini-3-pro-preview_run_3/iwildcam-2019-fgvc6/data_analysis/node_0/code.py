import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis

# Configuration
INPUT_ROOT = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis


def set_seeds(seed):
    np.random.seed(seed)
    # torch is not used here, but good practice if extended
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_data():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    return df


def process_image(row_tuple):
    """
    Worker function to process a single image.
    Returns a dictionary of stats or None if failure.
    """
    idx, row = row_tuple
    # Construct full path. Metadata file_path is relative to input root (e.g., train_images/id.jpg)
    full_path = os.path.join(INPUT_ROOT, row["file_path"])

    try:
        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Dimensions
        h, w, c = img.shape

        # Pixel stats (Global for the image)
        # Convert to float for precision
        img_float = img.astype(np.float32) / 255.0
        mean_val = np.mean(img_float)
        std_val = np.std(img_float)

        return {
            "Id": row["Id"],
            "Category": row["Category"],
            "Height": h,
            "Width": w,
            "Channels": c,
            "AspectRatio": w / h if h > 0 else 0,
            "PixelMean": mean_val,
            "PixelStd": std_val,
        }
    except Exception:
        return None


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("========================")

    # Distribution
    counts = df["Category"].value_counts()
    total = len(df)

    print(f"Total Samples: {total}")
    print(f"Number of Classes: {df['Category'].nunique()}")

    # Imbalance
    # Calculate ratios
    ratios = counts / total
    print("\nClass Distribution (Top 5):")
    for cat, ratio in ratios.head(5).items():
        print(f"Class {cat}: {ratio:.4f} ({counts[cat]} samples)")

    min_class = counts.idxmin()
    max_class = counts.idxmax()
    imbalance_ratio = counts[max_class] / counts[min_class]

    print(f"\nMost Frequent Class: {max_class} ({counts[max_class]} samples)")
    print(f"Least Frequent Class: {min_class} ({counts[min_class]} samples)")
    print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")
    print("-" * 30)


def analyze_images(df):
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("===========================")

    # Stratified Sampling to ensure we cover classes, but fallback to random if classes are too small
    if len(df) > SAMPLE_SIZE:
        try:
            # Attempt stratified sample
            sample_df, _ = train_test_split(
                df, train_size=SAMPLE_SIZE, stratify=df["Category"], random_state=SEED
            )
        except ValueError:
            # Fallback for rare classes
            sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        sample_df = df

    print(f"Analyzing {len(sample_df)} sampled images for detailed statistics...")

    # Prepare data for multiprocessing
    rows = list(sample_df.iterrows())

    # Use multiprocessing to speed up IO
    num_cores = min(12, multiprocessing.cpu_count())
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_image, rows)

    # Filter Nones
    results = [r for r in results if r is not None]
    stats_df = pd.DataFrame(results)

    if stats_df.empty:
        print("Error: No images could be processed.")
        return stats_df

    # Dimensions
    print("\nDimensions:")
    print(
        f"Width:  Mean={stats_df['Width'].mean():.4f}, Std={stats_df['Width'].std():.4f}, Min={stats_df['Width'].min()}, Max={stats_df['Width'].max()}"
    )
    print(
        f"Height: Mean={stats_df['Height'].mean():.4f}, Std={stats_df['Height'].std():.4f}, Min={stats_df['Height'].min()}, Max={stats_df['Height'].max()}"
    )

    # Aspect Ratios
    print("\nAspect Ratios:")
    print(
        f"Mean={stats_df['AspectRatio'].mean():.4f}, Std={stats_df['AspectRatio'].std():.4f}"
    )

    # Channels
    print("\nChannels:")
    channel_counts = stats_df["Channels"].value_counts()
    for c, count in channel_counts.items():
        print(f"{c} Channels: {count} images ({count/len(stats_df):.2%})")

    # Pixel Stats
    print("\nPixel Statistics (Normalized 0-1):")
    print(f"Global Mean Intensity: {stats_df['PixelMean'].mean():.4f}")
    print(f"Global Std Deviation:  {stats_df['PixelStd'].mean():.4f}")

    return stats_df


def analyze_relationships(stats_df):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("============================")

    if stats_df.empty:
        print("No image stats available for relationship analysis.")
        return

    # Structured Relationships (Meta-features)
    # Features: Width, Height, AspectRatio, PixelMean, PixelStd
    features = ["Width", "Height", "AspectRatio", "PixelMean", "PixelStd"]
    X = stats_df[features].fillna(0)
    y = stats_df["Category"]

    # Correlation
    print("Structured Relationships (Meta-features Correlation):")
    corr_matrix = X.corr(method="pearson")

    # Check for redundancy (Collinear pairs > 0.90)
    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                redundant_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    if redundant_pairs:
        print("Redundant Features (Correlation > 0.90):")
        for f1, f2, val in redundant_pairs:
            print(f"  {f1} - {f2}: {val:.4f}")
    else:
        print("No highly collinear meta-features found (> 0.90).")

    # Feature Importance (Random Forest)
    print("\nMeta-Feature Importance (Random Forest):")
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    # Unstructured Relationships
    print("\nUnstructured (Meta-Feature) Relationships:")
    # Check correlation between meta-features and Target (Category ID is nominal, but we can check if specific features vary by class)
    # We will look at the correlation between features and the 'Empty' class (Category 0) vs others,
    # as 'Empty' is the dominant class.

    stats_df["Is_Empty"] = (stats_df["Category"] == 0).astype(int)

    print("Correlation with 'Empty' Class (Category 0):")
    for feat in features:
        corr = stats_df[feat].corr(stats_df["Is_Empty"])
        print(f"  {feat} vs Is_Empty: {corr:.4f}")

    print("\nObservation:")
    if abs(stats_df["PixelMean"].corr(stats_df["Is_Empty"])) > 0.1:
        print(
            "  There is a noticeable correlation between image brightness and the image being empty."
        )
    else:
        print(
            "  Image brightness does not strongly correlate with the image being empty."
        )


def main():
    set_seeds(SEED)

    try:
        df = load_data()

        # 1. Target Variable Analysis
        analyze_target(df)

        # 2. Image Data Analysis (with sampling)
        stats_df = analyze_images(df)

        # 3. Feature Relationships
        analyze_relationships(stats_df)

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
