import os
import cv2
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Constants
INPUT_DIR = Path("./input")
METADATA_PATH = Path("./metadata/train.csv")
SAMPLE_SIZE = 5000  # Number of images to sample for pixel/dimension analysis
NUM_WORKERS = 12  # Usage of available vCPUs


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_target(df):
    print_section("Target Variable Analysis")

    target_col = "category_id"
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")

    # Imbalance Analysis
    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()
    imbalance_ratio = max_count / min_count

    print(f"Class Counts - Min: {min_count}, Max: {max_count}, Mean: {mean_count:.4f}")
    print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Rare classes (< 1%)
    # In a dataset with 1000 classes, 1% is high, so we look for very rare absolute counts or relative to mean
    threshold_pct = 0.01
    rare_classes = class_counts[class_counts < (total_samples * threshold_pct)]
    print(
        f"Classes with < 1% frequency: {len(rare_classes)} ({len(rare_classes)/num_classes*100:.2f}%)"
    )

    # Top 5 classes
    print("\nTop 5 Frequent Classes:")
    print(class_counts.head(5).to_string())


def get_image_stats(row):
    """
    Reads an image and returns metadata and pixel stats.
    Returns None if read fails.
    """
    file_path = INPUT_DIR / row["file_path"]
    try:
        # Check file size
        file_size = os.path.getsize(file_path)

        # Read image
        img = cv2.imread(str(file_path))
        if img is None:
            return None

        # Dimensions (H, W, C)
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        # Convert BGR to RGB if color
        if c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Pixel stats (normalize to 0-1 for calculation)
        img_norm = img.astype(np.float32) / 255.0

        # Global mean/std per channel
        if c == 3:
            means = img_norm.mean(axis=(0, 1))
            stds = img_norm.std(axis=(0, 1))
        else:
            means = [img_norm.mean()]
            stds = [img_norm.std()]

        return {
            "width": w,
            "height": h,
            "channels": c,
            "aspect_ratio": w / h if h > 0 else 0,
            "file_size_bytes": file_size,
            "mean_r": means[0] if c == 3 else means[0],
            "mean_g": means[1] if c == 3 else 0,
            "mean_b": means[2] if c == 3 else 0,
            "std_r": stds[0] if c == 3 else stds[0],
            "std_g": stds[1] if c == 3 else 0,
            "std_b": stds[2] if c == 3 else 0,
            "category_id": row["category_id"],
        }
    except Exception:
        return None


def analyze_images(df):
    print_section("Input Data Analysis (Image Modality)")

    # Stratified Sample if possible, else random
    if len(df) > SAMPLE_SIZE:
        try:
            # Attempt stratified sampling, fallback to random if classes are too small
            sample_df = df.groupby("category_id", group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), max(1, int(SAMPLE_SIZE / df["category_id"].nunique())))
                )
            )
            # If stratification yields too few, just random sample the rest
            if len(sample_df) < SAMPLE_SIZE:
                remaining = SAMPLE_SIZE - len(sample_df)
                rest_df = df.drop(sample_df.index)
                if not rest_df.empty:
                    sample_df = pd.concat(
                        [
                            sample_df,
                            rest_df.sample(
                                n=min(len(rest_df), remaining), random_state=RANDOM_SEED
                            ),
                        ]
                    )
        except Exception:
            sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        sample_df = df

    print(f"Analyzing a sample of {len(sample_df)} images...")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Map rows to futures
        futures = [
            executor.submit(get_image_stats, row) for _, row in sample_df.iterrows()
        ]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)

    stats_df = pd.DataFrame(results)

    if stats_df.empty:
        print("Error: No images could be processed.")
        return pd.DataFrame()

    # Dimensions
    print("\n--- Dimensions ---")
    print(
        f"Width  - Mean: {stats_df['width'].mean():.4f}, Std: {stats_df['width'].std():.4f}, Min: {stats_df['width'].min()}, Max: {stats_df['width'].max()}"
    )
    print(
        f"Height - Mean: {stats_df['height'].mean():.4f}, Std: {stats_df['height'].std():.4f}, Min: {stats_df['height'].min()}, Max: {stats_df['height'].max()}"
    )
    print(
        f"Aspect Ratio - Mean: {stats_df['aspect_ratio'].mean():.4f}, Std: {stats_df['aspect_ratio'].std():.4f}"
    )

    # Channels
    print("\n--- Channels ---")
    channel_counts = stats_df["channels"].value_counts()
    for ch, count in channel_counts.items():
        print(f"Channel {ch}: {count} images ({count/len(stats_df)*100:.2f}%)")

    # Pixel Stats (Global)
    print("\n--- Pixel Statistics (Normalized 0-1) ---")
    # Filter for RGB images for accurate RGB stats
    rgb_df = stats_df[stats_df["channels"] == 3]
    if not rgb_df.empty:
        print(
            f"Mean (R, G, B): ({rgb_df['mean_r'].mean():.4f}, {rgb_df['mean_g'].mean():.4f}, {rgb_df['mean_b'].mean():.4f})"
        )
        print(
            f"Std  (R, G, B): ({rgb_df['std_r'].mean():.4f}, {rgb_df['std_g'].mean():.4f}, {rgb_df['std_b'].mean():.4f})"
        )
    else:
        print(f"Mean (Gray): {stats_df['mean_r'].mean():.4f}")
        print(f"Std  (Gray): {stats_df['std_r'].mean():.4f}")

    return stats_df


def analyze_relationships(stats_df):
    print_section("Feature/Signal Relationships (Meta-Features)")

    if stats_df.empty:
        print("Skipping relationship analysis due to empty stats.")
        return

    # Prepare data for relationship analysis
    # Features: Width, Height, Aspect Ratio, File Size
    # Target: category_id

    features = ["width", "height", "aspect_ratio", "file_size_bytes"]
    X = stats_df[features].fillna(0)
    y = stats_df["category_id"]

    # Encode target if necessary (it's already int, but for safety)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 1. Correlation (Numerical Features)
    print("\n--- Correlation Matrix (Pearson) ---")
    corr_matrix = X.corr(method="pearson")
    print(corr_matrix.round(4))

    # Check for collinearity
    collinear_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                collinear_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    if collinear_pairs:
        print(f"\nHighly Correlated Pairs (>0.90): {collinear_pairs}")
    else:
        print("\nNo highly correlated pairs found (>0.90).")

    # 2. Mutual Information
    print("\n--- Mutual Information with Target ---")
    # Discrete features? No, these are continuous meta-features.
    mi = mutual_info_classif(
        X, y_enc, discrete_features=False, random_state=RANDOM_SEED
    )
    mi_series = pd.Series(mi, index=features).sort_values(ascending=False)
    print(mi_series.to_string())

    # 3. Feature Importance (Random Forest)
    print("\n--- Random Forest Feature Importance ---")
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X, y_enc)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print(importances.to_string())

    # Meta-feature insight
    print("\n--- Meta-Feature Insights ---")
    print(
        "Interpretation: Low Mutual Information and Feature Importance values suggest that basic image geometry"
    )
    print(
        "(size, aspect ratio) is not highly predictive of the species class on its own, which is expected"
    )
    print("for fine-grained classification tasks.")


def main():
    # Load Data
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    if df.empty:
        print("Error: Metadata dataframe is empty.")
        return

    # 1. Target Analysis
    analyze_target(df)

    # 2. Image Analysis
    stats_df = analyze_images(df)

    # 3. Relationship Analysis
    analyze_relationships(stats_df)

    print_section("End of Report")


if __name__ == "__main__":
    main()
