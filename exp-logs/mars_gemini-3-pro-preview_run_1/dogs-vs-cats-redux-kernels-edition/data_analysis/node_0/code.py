import os
import cv2
import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed()

    # Define paths
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    print("SECTION 1: DATA INTEGRITY CHECK")
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load training metadata
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded training metadata with {len(df)} samples.")
    print("-" * 30)

    # ==========================================
    # SECTION 2: TARGET VARIABLE ANALYSIS
    # ==========================================
    print("SECTION 2: TARGET VARIABLE ANALYSIS")

    target_col = "label"
    class_counts = df[target_col].value_counts()
    total_samples = len(df)

    print(f"Target Variable: '{target_col}'")
    print("Distribution:")
    for label, count in class_counts.items():
        ratio = count / total_samples
        label_name = "Dog" if label == 1 else "Cat"
        print(f"  Class {label} ({label_name}): {count} samples ({ratio:.4f})")

    # Check for imbalance
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    print(f"Class Balance Ratio (Max/Min): {imbalance_ratio:.4f}")
    if imbalance_ratio < 1.1:
        print("  -> The dataset is balanced.")
    else:
        print("  -> The dataset shows imbalance.")
    print("-" * 30)

    # ==========================================
    # SECTION 3: INPUT DATA ANALYSIS (IMAGE)
    # ==========================================
    print("SECTION 3: INPUT DATA ANALYSIS (IMAGE)")

    # Initialize aggregators
    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = {}
    file_sizes = []

    # For pixel stats (Welford's algorithm or simple accumulation)
    # We will use simple accumulation for mean/std to keep it fast and memory efficient
    # Accumulating per channel: B, G, R
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixel_count = 0

    # We will iterate through the dataset.
    # To ensure execution speed, if the dataset is massive, we might sample,
    # but 18k images is manageable.

    print(f"Processing {len(df)} images for statistical analysis...")

    valid_images_count = 0

    for idx, row in df.iterrows():
        rel_path = row["filepath"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        try:
            # Get file size
            f_size = os.path.getsize(full_path)
            file_sizes.append(f_size)

            # Read image
            img = cv2.imread(full_path)

            if img is None:
                continue

            h, w, c = img.shape

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

            # Channel count
            if c not in channel_counts:
                channel_counts[c] = 0
            channel_counts[c] += 1

            # Pixel stats accumulation
            # Normalize to 0-1 range for calculation to avoid overflow with squares, then scale back or keep as is.
            # Standard practice often reports 0-255 or standardized. Let's calculate on 0-255 scale.

            # Flatten spatial dimensions
            pixels = img.reshape(-1, 3)
            n_pixels = pixels.shape[0]

            # Accumulate
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels**2).sum(axis=0)
            total_pixel_count += n_pixels
            valid_images_count += 1

        except Exception as e:
            continue

    # Convert lists to arrays for stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    file_sizes = np.array(file_sizes)

    # Dimensions Analysis
    print("Dimensions:")
    print(
        f"  Widths:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Heights: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  Aspect Ratios: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Outlier detection for dimensions (IQR)
    def count_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return np.sum((data < lower) | (data > upper))

    print(f"  Width Outliers (IQR method): {count_outliers(widths)}")
    print(f"  Height Outliers (IQR method): {count_outliers(heights)}")

    # Channels Analysis
    print("Channels:")
    for c, count in channel_counts.items():
        print(f"  {c} Channels: {count} images ({count/valid_images_count:.4f})")
        if c == 3:
            print("  -> Majority are RGB.")
        elif c == 1:
            print("  -> Some Grayscale images found.")

    # Pixel Stats Analysis
    # Mean = Sum / N
    # Std = Sqrt(E[X^2] - (E[X])^2)
    if total_pixel_count > 0:
        global_mean = channel_sum / total_pixel_count
        global_sq_mean = channel_sq_sum / total_pixel_count
        global_std = np.sqrt(global_sq_mean - global_mean**2)

        # OpenCV reads as BGR
        print("Pixel Statistics (Global, BGR order, 0-255 scale):")
        print(
            f"  Mean: B={global_mean[0]:.4f}, G={global_mean[1]:.4f}, R={global_mean[2]:.4f}"
        )
        print(
            f"  Std:  B={global_std[0]:.4f},  G={global_std[1]:.4f},  R={global_std[2]:.4f}"
        )
    else:
        print("Pixel Statistics: No valid pixels processed.")

    print("-" * 30)

    # ==========================================
    # SECTION 4: FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("SECTION 4: FEATURE/SIGNAL RELATIONSHIPS (UNSTRUCTURED META-FEATURES)")

    # Create a temporary dataframe for correlation analysis
    # We need to align the stats collected with the labels.
    # Since we iterated sequentially and skipped failures, we need to be careful.
    # For this report, we'll assume the 'valid_images_count' matches the loop order.
    # To be robust, we should have stored these in a list of dicts.

    # Re-collecting metadata into a structured format for correlation
    # (Since we already have the arrays and we iterated df sequentially,
    # we just need to filter df for the rows we successfully processed.
    # Assuming no read errors for this clean dataset, lengths should match).

    if len(widths) == len(df):
        meta_df = df.copy()
        meta_df["width"] = widths
        meta_df["height"] = heights
        meta_df["aspect_ratio"] = aspect_ratios
        meta_df["file_size"] = file_sizes

        # Correlation with Target
        # Target is binary (0/1). We can use Point Biserial Correlation.
        print("Correlation between Metadata and Target (Label: 0=Cat, 1=Dog):")

        features_to_check = ["width", "height", "aspect_ratio", "file_size"]

        for feat in features_to_check:
            # Point Biserial: correlation between binary and continuous
            corr, p_val = stats.pointbiserialr(meta_df[target_col], meta_df[feat])
            print(f"  {feat}: Correlation={corr:.4f}, P-value={p_val:.4f}")

            # Interpretation
            if abs(corr) > 0.1:
                direction = "positive" if corr > 0 else "negative"
                print(f"    -> Weak {direction} relationship detected.")

        # Compare means by class
        print("\nMean Metadata values by Class:")
        grouped = meta_df.groupby(target_col)[features_to_check].mean()
        print(grouped)

        # Check for redundancy (Collinearity between meta-features)
        print("\nMeta-feature Redundancy (Pearson Correlation > 0.90):")
        corr_matrix = meta_df[features_to_check].corr()
        found_redundancy = False
        for i in range(len(features_to_check)):
            for j in range(i + 1, len(features_to_check)):
                c_val = corr_matrix.iloc[i, j]
                if abs(c_val) > 0.90:
                    print(
                        f"  High correlation between {features_to_check[i]} and {features_to_check[j]}: {c_val:.4f}"
                    )
                    found_redundancy = True
        if not found_redundancy:
            print("  No high redundancy found among meta-features.")

    else:
        print(
            "Warning: Number of processed images does not match metadata length. Skipping detailed correlation analysis to avoid misalignment."
        )

    print("-" * 30)
    print("EDA COMPLETE")


if __name__ == "__main__":
    main()
