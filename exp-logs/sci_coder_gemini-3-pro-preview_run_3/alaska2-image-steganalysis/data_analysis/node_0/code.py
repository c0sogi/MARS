import os
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE = 5000  # Number of images to sample for pixel-level analysis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_FILE}")
    return pd.read_csv(METADATA_FILE)


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    # Distribution
    counts = df["label"].value_counts()
    total = len(df)

    print(f"Total Samples: {total}")
    print("Label Distribution:")
    for label, count in counts.items():
        print(f"  Class {label}: {count} ({count/total*100:.4f}%)")

    # Imbalance
    if len(counts) == 2:
        ratio = counts.max() / counts.min()
        print(f"Class Imbalance Ratio: 1 : {ratio:.4f}")
    else:
        print("Multiclass distribution detected.")
    print("\n")


def get_image_stats(row):
    """
    Reads an image and returns basic stats.
    """
    path = os.path.join(INPUT_DIR, row["image_path"])
    if not os.path.exists(path):
        return None

    # Read image
    img = cv2.imread(path)
    if img is None:
        return None

    # Dimensions
    h, w, c = img.shape
    aspect_ratio = w / h if h > 0 else 0

    # Pixel stats
    # Convert to float for accurate mean/std
    img_float = img.astype(np.float32) / 255.0
    mean_val = np.mean(img_float)
    std_val = np.std(img_float)

    return {
        "width": w,
        "height": h,
        "channels": c,
        "aspect_ratio": aspect_ratio,
        "pixel_mean": mean_val,
        "pixel_std": std_val,
        "label": row["label"],
    }


def analyze_images(df):
    print("INPUT DATA ANALYSIS (IMAGE MODALITY)")
    print("=" * 30)

    # Sampling
    if len(df) > SAMPLE_SIZE:
        # Stratified sampling to ensure we see both classes
        sample_df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), int(SAMPLE_SIZE / df["label"].nunique())), random_state=SEED
            )
        )
    else:
        sample_df = df.copy()

    print(f"Analyzing a sample of {len(sample_df)} images for pixel statistics...")

    stats_list = []
    for _, row in sample_df.iterrows():
        stats = get_image_stats(row)
        if stats:
            stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Dimensions
    print("\n--- Dimensions ---")
    print(
        f"Widths: Mean={stats_df['width'].mean():.4f}, Std={stats_df['width'].std():.4f}, Min={stats_df['width'].min()}, Max={stats_df['width'].max()}"
    )
    print(
        f"Heights: Mean={stats_df['height'].mean():.4f}, Std={stats_df['height'].std():.4f}, Min={stats_df['height'].min()}, Max={stats_df['height'].max()}"
    )
    print(
        f"Aspect Ratios: Mean={stats_df['aspect_ratio'].mean():.4f}, Std={stats_df['aspect_ratio'].std():.4f}"
    )

    # Channels
    print("\n--- Channels ---")
    channel_counts = stats_df["channels"].value_counts()
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} samples ({count/len(stats_df)*100:.2f}%)")

    # Pixel Stats
    print("\n--- Pixel Stats (Normalized 0-1) ---")
    print(
        f"Global Pixel Mean: {stats_df['pixel_mean'].mean():.4f} (Std: {stats_df['pixel_mean'].std():.4f})"
    )
    print(
        f"Global Pixel Std Dev: {stats_df['pixel_std'].mean():.4f} (Std: {stats_df['pixel_std'].std():.4f})"
    )

    print("\n")
    return stats_df


def analyze_relationships(stats_df):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Prepare data for analysis
    # We treat extracted image stats as features
    features = ["width", "height", "aspect_ratio", "pixel_mean", "pixel_std"]
    X = stats_df[features]
    y = stats_df["label"]

    # 1. Correlation
    print("--- Meta-Feature Correlations with Target ---")
    # Calculate correlation of features with the label
    correlations = X.apply(lambda x: x.corr(y))
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # 2. Feature Importance (Random Forest)
    print("\n--- Meta-Feature Importance (Random Forest) ---")
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")

    # 3. Unstructured/Meta Relationships
    print("\n--- Unstructured Relationships ---")
    # Check if larger images correlate with specific classes
    # We compare the mean width/height for class 0 vs class 1
    class_0_stats = stats_df[stats_df["label"] == 0]
    class_1_stats = stats_df[stats_df["label"] == 1]

    print(
        f"Avg File Size (Pixels) Class 0: {(class_0_stats['width'] * class_0_stats['height']).mean():.4f}"
    )
    print(
        f"Avg File Size (Pixels) Class 1: {(class_1_stats['width'] * class_1_stats['height']).mean():.4f}"
    )
    print(f"Avg Pixel Intensity Class 0: {class_0_stats['pixel_mean'].mean():.4f}")
    print(f"Avg Pixel Intensity Class 1: {class_1_stats['pixel_mean'].mean():.4f}")

    print("\n")


def main():
    set_seed(SEED)

    try:
        # Load Data
        df = load_metadata()

        # Target Analysis
        analyze_target(df)

        # Image Analysis
        stats_df = analyze_images(df)

        # Relationship Analysis
        if not stats_df.empty:
            analyze_relationships(stats_df)

    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
