import os
import cv2
import numpy as np
import pandas as pd
import random
from scipy import stats
import sys

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_targets(df):
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    target_col = "breed"

    # Distribution
    class_counts = df[target_col].value_counts()
    num_classes = len(class_counts)
    total_samples = len(df)

    print(f"Target Variable: '{target_col}'")
    print(f"Task Type: Classification")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {num_classes}")

    # Imbalance/Skew
    min_class = class_counts.min()
    max_class = class_counts.max()
    mean_class = class_counts.mean()
    imbalance_ratio = max_class / min_class

    print(f"Class Distribution Stats:")
    print(f"  Min samples per class: {min_class} ({class_counts.idxmin()})")
    print(f"  Max samples per class: {max_class} ({class_counts.idxmax()})")
    print(f"  Mean samples per class: {mean_class:.4f}")
    print(f"  Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Check for rare classes
    rare_threshold = 0.01  # 1%
    class_proportions = class_counts / total_samples
    rare_classes = class_proportions[class_proportions < rare_threshold]

    print(f"  Rare Classes (<1% frequency): {len(rare_classes)} found")
    if len(rare_classes) > 0:
        print(
            f"  Example rare class: {rare_classes.index[0]} ({rare_classes.iloc[0]*100:.4f}%)"
        )

    print("-" * 30)


def analyze_images(df):
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("=" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = {}

    # Accumulators for global pixel stats (R, G, B)
    # We will assume images are read as BGR by OpenCV and convert to RGB
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # Iterate through images
    # Using a subset if dataset is massive, but 7k is small enough to process all
    # for accurate global stats.

    valid_images_count = 0

    for idx, row in df.iterrows():
        # Construct full path. Metadata contains relative path 'train/id.jpg'
        # Input dir is './input'
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read image
        # IMREAD_UNCHANGED to detect if grayscale or alpha channel exists
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

        channel_counts[c] = channel_counts.get(c, 0) + 1

        # Update global stats accumulators
        # Normalize to RGB 3-channel for calculation
        if c == 3:
            # OpenCV is BGR, convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Flatten spatial dimensions
            pixels = img_rgb.reshape(-1, 3) / 255.0

            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels**2).sum(axis=0)
            total_pixels += pixels.shape[0]

        elif c == 1:
            # Grayscale
            pixels = img.reshape(-1, 1) / 255.0
            # Treat as repeating across RGB for global stats or skip?
            # Usually better to skip grayscale for RGB norm stats to avoid skew
            # Or replicate. Let's replicate to represent "visual content"
            pixels_rgb = np.repeat(pixels, 3, axis=1)
            channel_sum += pixels_rgb.sum(axis=0)
            channel_sq_sum += (pixels_rgb**2).sum(axis=0)
            total_pixels += pixels.shape[0]

        elif c == 4:
            # RGBA - Drop Alpha for stats
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            pixels = img_rgb.reshape(-1, 3) / 255.0
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels**2).sum(axis=0)
            total_pixels += pixels.shape[0]

        valid_images_count += 1

    # Dimensions Analysis
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print("Dimensions:")
    print(
        f"  Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"  Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"  Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}"
    )

    # Channels Analysis
    print("\nChannels:")
    for c, count in channel_counts.items():
        print(f"  {c} channels: {count} images ({count/valid_images_count*100:.2f}%)")

    # Pixel Stats Analysis
    if total_pixels > 0:
        global_mean = channel_sum / total_pixels
        # Var(X) = E[X^2] - (E[X])^2
        global_var = (channel_sq_sum / total_pixels) - (global_mean**2)
        global_std = np.sqrt(global_var)

        print("\nPixel Stats (RGB, Normalized [0,1]):")
        print(
            f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
        )
        print(
            f"  Std:  R={global_std[0]:.4f},  G={global_std[1]:.4f},  B={global_std[2]:.4f}"
        )
    else:
        print("\nPixel Stats: No data processed.")

    # Return data for relationship analysis
    return pd.DataFrame(
        {"width": widths, "height": heights, "aspect_ratio": aspect_ratios}
    )


def analyze_relationships(meta_df, target_df):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Merge metadata features with target
    # target_df index aligns with loop order if we didn't shuffle/skip
    # But safer to merge on ID if we had it in the loop.
    # Since we iterated df rows sequentially, we can assign directly if lengths match.

    if len(meta_df) != len(target_df):
        print(
            "Warning: Mismatch in processed images and target labels. Skipping relationship analysis."
        )
        return

    df = pd.concat(
        [target_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1
    )

    # 1. Structured Relationship: Correlation between image size and aspect ratio
    # (Not strictly target-related, but signal structure)
    corr_wh = df["width"].corr(df["height"])
    print(f"Structured Relationships:")
    print(f"  Correlation (Width vs Height): {corr_wh:.4f}")

    # 2. Unstructured (Meta-Feature) Relationships: Metadata vs Target
    # Does breed predict image dimensions?
    print("\nMeta-Feature vs Target Relationships:")

    # Group by breed and check variance
    grouped = df.groupby("breed")[["width", "height", "aspect_ratio"]].mean()

    # Identify breeds with extreme average dimensions
    max_width_breed = grouped["width"].idxmax()
    min_width_breed = grouped["width"].idxmin()

    print(
        f"  Breed with largest avg width: {max_width_breed} ({grouped.loc[max_width_breed, 'width']:.2f} px)"
    )
    print(
        f"  Breed with smallest avg width: {min_width_breed} ({grouped.loc[min_width_breed, 'width']:.2f} px)"
    )

    # Statistical Test: Kruskal-Wallis H-test
    # Null hypothesis: The population median of all of the groups are equal.
    # i.e., Does image width vary significantly across breeds?

    breeds = df["breed"].unique()
    width_groups = [df[df["breed"] == b]["width"].values for b in breeds]

    try:
        stat, p_value = stats.kruskal(*width_groups)
        print(f"\n  Statistical Test (Kruskal-Wallis) for Width across Breeds:")
        print(f"    H-statistic: {stat:.4f}")
        print(f"    P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("    Result: Significant difference in image width across breeds.")
        else:
            print("    Result: No significant difference in image width across breeds.")
    except Exception as e:
        print(f"    Could not perform statistical test: {e}")

    print("-" * 30)


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    analyze_targets(df)

    # 2. Image Analysis
    meta_features = analyze_images(df)

    # 3. Relationship Analysis
    analyze_relationships(meta_features, df)


if __name__ == "__main__":
    main()
