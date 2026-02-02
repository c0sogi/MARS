import os
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
METADATA_PATH = "./metadata/train.csv"
INPUT_ROOT = "./input"
SEED = 42
SAMPLE_SIZE_PIXELS = 1000  # Number of images to sample for pixel stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def main():
    # --------------------------------------------------------------------------
    # 1. Data Loading
    # --------------------------------------------------------------------------
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Ensure relevant columns exist
    required_cols = ["category_id", "file_path", "width", "height", "location"]
    if not all(col in df.columns for col in required_cols):
        print("Error: Metadata missing required columns.")
        return

    print("========================================")
    print("      EXPLORATORY DATA ANALYSIS")
    print("========================================")

    # --------------------------------------------------------------------------
    # 2. Target Variable Analysis
    # --------------------------------------------------------------------------
    print("\n[TARGET VARIABLE ANALYSIS]")

    # Distribution
    class_counts = df["category_id"].value_counts()
    n_classes = len(class_counts)
    total_samples = len(df)

    print(f"Target Variable: category_id")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {n_classes}")

    # Imbalance
    most_freq_class = class_counts.idxmax()
    most_freq_count = class_counts.max()
    least_freq_class = class_counts.idxmin()
    least_freq_count = class_counts.min()

    imbalance_ratio = (
        most_freq_count / least_freq_count if least_freq_count > 0 else float("inf")
    )

    print(
        f"Most Frequent Class: {most_freq_class} (Count: {most_freq_count}, {most_freq_count/total_samples*100:.2f}%)"
    )
    print(
        f"Least Frequent Class: {least_freq_class} (Count: {least_freq_count}, {least_freq_count/total_samples*100:.2f}%)"
    )
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top 5 Classes
    print("Top 5 Classes:")
    print(class_counts.head(5).to_string())

    # --------------------------------------------------------------------------
    # 3. Input Data Analysis (Image Modality)
    # --------------------------------------------------------------------------
    print("\n[INPUT DATA ANALYSIS - IMAGE MODALITY]")

    # Dimensions (from metadata)
    widths = df["width"]
    heights = df["height"]
    aspect_ratios = widths / heights

    print("Image Dimensions (Metadata):")
    print(
        f"  Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"  Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"  Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    # Pixel Stats & Channels (Sampling)
    print(f"Analyzing Pixel Stats (Sample Size: {SAMPLE_SIZE_PIXELS})...")

    sample_df = df.sample(n=min(SAMPLE_SIZE_PIXELS, len(df)), random_state=SEED)

    channel_counts = {}
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    valid_images = 0

    for _, row in sample_df.iterrows():
        # Construct full path
        img_path = os.path.join(INPUT_ROOT, row["file_path"])

        # Read image
        try:
            # imread loads as BGR by default
            img = cv2.imread(img_path)
            if img is None:
                continue

            valid_images += 1

            # Check channels
            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1

            if c not in channel_counts:
                channel_counts[c] = 0
            channel_counts[c] += 1

            # Accumulate for mean/std (assuming BGR or RGB, 3 channels)
            # If grayscale, convert to 3 channels for consistent stats or handle separately
            if c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Normalize to 0-1 for calculation stability, then scale back or keep raw
            # Here we keep raw [0, 255] for reporting
            img_data = img.reshape(-1, 3)
            pixel_sum += img_data.sum(axis=0)
            pixel_sq_sum += (img_data**2).sum(axis=0)
            pixel_count += h * w

        except Exception:
            continue

    if valid_images > 0:
        # Calculate global mean and std
        # Note: OpenCV loads BGR. We report as BGR.
        global_mean = pixel_sum / pixel_count
        global_variance = (pixel_sq_sum / pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_variance)

        print(f"  Processed {valid_images} images successfully.")
        print(f"  Channel Distribution: {channel_counts}")
        print(
            f"  Pixel Mean (BGR): [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]"
        )
        print(
            f"  Pixel Std  (BGR): [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]"
        )
    else:
        print("  Could not process any images for pixel stats.")

    # --------------------------------------------------------------------------
    # 4. Feature/Signal Relationships
    # --------------------------------------------------------------------------
    print("\n[FEATURE/SIGNAL RELATIONSHIPS]")

    # 4a. Unstructured/Meta-Feature Relationships
    # Correlation between Dimensions and Target (using encoded target if needed, but target is int)
    # Since target is categorical (nominal), Pearson isn't ideal, but we can check if size varies by class
    # Let's use Mutual Information for categorical dependencies or ANOVA.
    # For simplicity/robustness, we'll check if width/height correlate with class ID (weak proxy)
    # and then use a model to find importance.

    # 4b. Structured Relationships (Metadata as Features)
    # We will treat 'location', 'width', 'height' as features to predict 'category_id'

    # Prepare data for lightweight model
    # Encode location
    le_loc = LabelEncoder()
    # Handle potential non-string locations
    df["location_str"] = df["location"].astype(str)
    df["location_enc"] = le_loc.fit_transform(df["location_str"])

    features = ["location_enc", "width", "height"]
    X = df[features]
    y = df["category_id"]

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_imp = dict(zip(features, importances))
    sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)

    print("Metadata Feature Importance (Random Forest):")
    for name, imp in sorted_imp:
        print(f"  {name}: {imp:.4f}")

    print("\nInterpretation:")
    if feature_imp["location_enc"] > 0.5:
        print(
            "  Location is a dominant predictor. This indicates strong location bias (common in camera traps)."
        )
        print(
            "  Models must generalize to new locations, not just memorize background/location specific animals."
        )
    else:
        print("  Location is not the sole dominant predictor.")

    # Check correlation between image size and class (Are some animals only in high-res images?)
    # We'll calculate the correlation ratio or just mean size per class for top 5 classes
    print("\nMean Image Area (Width * Height) for Top 5 Classes:")
    top_5_classes = class_counts.head(5).index
    for cls in top_5_classes:
        subset = df[df["category_id"] == cls]
        mean_area = (subset["width"] * subset["height"]).mean()
        print(f"  Class {cls}: {mean_area:.2f} pixels^2")


if __name__ == "__main__":
    main()
