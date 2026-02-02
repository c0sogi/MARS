import os
import pandas as pd
import numpy as np
import cv2
import sys

# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    import random

    random.seed(seed)


def analyze_target_variable(df):
    print("TARGET VARIABLE ANALYSIS")

    # 1. Distribution of the target variable 'Id'
    id_counts = df["Id"].value_counts()
    num_classes = len(id_counts)
    num_samples = len(df)

    print(f"Total Samples: {num_samples}")
    print(f"Total Unique Classes: {num_classes}")

    # 2. Imbalance/Skew
    # Identify 'new_whale' vs specific whales
    new_whale_count = id_counts.get("new_whale", 0)
    known_whale_count = num_samples - new_whale_count

    print(
        f"Class 'new_whale' count: {new_whale_count} ({new_whale_count/num_samples:.4f})"
    )
    print(
        f"Known whale samples: {known_whale_count} ({known_whale_count/num_samples:.4f})"
    )

    # Analyze distribution of known whales
    known_counts = id_counts.drop("new_whale", errors="ignore")

    if len(known_counts) > 0:
        max_class_count = known_counts.max()
        min_class_count = known_counts.min()
        mean_class_count = known_counts.mean()
        median_class_count = known_counts.median()

        # Count singletons (classes with only 1 sample)
        singletons = (known_counts == 1).sum()

        print(f"Most frequent known class count: {max_class_count}")
        print(f"Least frequent known class count: {min_class_count}")
        print(f"Average samples per known class: {mean_class_count:.4f}")
        print(f"Median samples per known class: {median_class_count:.4f}")
        print(
            f"Number of singleton classes (1 sample): {singletons} ({singletons/len(known_counts):.4f} of known classes)"
        )

        # Imbalance Ratio (Max / Min)
        imbalance_ratio = max_class_count / min_class_count
        print(
            f"Class Imbalance Ratio (Max/Min for known classes): {imbalance_ratio:.4f}"
        )
    else:
        print("No known whale classes found.")

    print("-" * 30)


def analyze_image_data(df):
    print("INPUT DATA ANALYSIS (IMAGE)")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # For pixel stats (Global Mean/Std)
    # We will use Welford's algorithm or simple accumulation for mean/std to avoid loading all images into RAM
    # Given the dataset size, simple accumulation of sum and sum_sq is sufficient and robust enough.
    # We'll normalize pixel values to [0, 1] for calculation to keep numbers manageable.

    total_sum_r = 0.0
    total_sum_g = 0.0
    total_sum_b = 0.0

    total_sq_sum_r = 0.0
    total_sq_sum_g = 0.0
    total_sq_sum_b = 0.0

    total_pixels = 0

    # To track meta-features for relationship analysis later
    meta_features = []

    # Iterate through images
    # Using a counter to avoid infinite loops if something goes wrong, though unlikely
    count = 0

    for idx, row in df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(full_path):
            continue

        # Read image
        # IMREAD_UNCHANGED to detect if it's grayscale or color naturally
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        h, w = img.shape[:2]

        # Determine channels
        if len(img.shape) == 2:
            c = 1
            # Convert to RGB for consistent pixel stats calculation if needed,
            # or handle separately. Let's treat as single channel intensity.
            # To unify stats, we can treat grayscale as R=G=B.
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            c = img.shape[2]
            # OpenCV reads as BGR, convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channel_counts.append(c)

        # Normalize to 0-1 for stats
        img_norm = img_rgb.astype(np.float32) / 255.0

        # Accumulate stats
        # img_norm is (H, W, 3)
        n_pix = w * h

        sum_ch = np.sum(img_norm, axis=(0, 1))  # [sum_r, sum_g, sum_b]
        sum_sq_ch = np.sum(img_norm**2, axis=(0, 1))

        total_sum_r += sum_ch[0]
        total_sum_g += sum_ch[1]
        total_sum_b += sum_ch[2]

        total_sq_sum_r += sum_sq_ch[0]
        total_sq_sum_g += sum_sq_ch[1]
        total_sq_sum_b += sum_sq_ch[2]

        total_pixels += n_pix

        meta_features.append(
            {
                "Image": row["Image"],
                "Width": w,
                "Height": h,
                "AspectRatio": w / h,
                "Channels": c,
                "MeanIntensity": np.mean(img_norm),  # Average brightness of this image
            }
        )

        count += 1

    # 1. Dimensions
    widths = np.array(widths)
    heights = np.array(heights)
    ratios = np.array(aspect_ratios)

    print(f"Processed {count} images.")
    print(
        f"Width: Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"Aspect Ratio: Mean={np.mean(ratios):.4f}, Std={np.std(ratios):.4f}, Min={np.min(ratios):.4f}, Max={np.max(ratios):.4f}"
    )

    # 2. Channels
    unique_channels, counts_channels = np.unique(channel_counts, return_counts=True)
    print("Channel Distribution:")
    for ch, cnt in zip(unique_channels, counts_channels):
        print(f"  {ch} channels: {cnt} images ({cnt/count:.4f})")

    # 3. Pixel Stats (Global)
    # Mean = Sum / N
    # Std = Sqrt( (SumSq / N) - Mean^2 )

    mean_r = total_sum_r / total_pixels
    mean_g = total_sum_g / total_pixels
    mean_b = total_sum_b / total_pixels

    std_r = np.sqrt((total_sq_sum_r / total_pixels) - (mean_r**2))
    std_g = np.sqrt((total_sq_sum_g / total_pixels) - (mean_g**2))
    std_b = np.sqrt((total_sq_sum_b / total_pixels) - (mean_b**2))

    print("Global Pixel Statistics (Normalized [0, 1]):")
    print(f"  Red Channel:   Mean={mean_r:.4f}, Std={std_r:.4f}")
    print(f"  Green Channel: Mean={mean_g:.4f}, Std={std_g:.4f}")
    print(f"  Blue Channel:  Mean={mean_b:.4f}, Std={std_b:.4f}")

    # Calculate overall grayscale mean/std for reference
    overall_mean = (mean_r + mean_g + mean_b) / 3.0
    overall_std = np.sqrt(((std_r**2 + std_g**2 + std_b**2) / 3.0))  # Approximation
    print(f"  Overall Intensity: Mean={overall_mean:.4f}")

    print("-" * 30)

    return pd.DataFrame(meta_features)


def analyze_relationships(df, meta_df):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Merge metadata with original dataframe to get Ids
    merged_df = pd.merge(df, meta_df, on="Image")

    # Create a binary target for high-level analysis: new_whale vs known_whale
    merged_df["is_new_whale"] = merged_df["Id"] == "new_whale"

    # 1. Unstructured (Meta-Feature) Relationships
    # Compare dimensions and intensity between new_whale and known_whale

    print("Relationship: Metadata vs Target Class (New vs Known Whale)")

    groups = merged_df.groupby("is_new_whale")

    for is_new, group in groups:
        label = "New Whale" if is_new else "Known Whale"
        print(f"[{label}]")
        print(f"  Count: {len(group)}")
        print(f"  Avg Width: {group['Width'].mean():.4f}")
        print(f"  Avg Height: {group['Height'].mean():.4f}")
        print(f"  Avg Aspect Ratio: {group['AspectRatio'].mean():.4f}")
        print(f"  Avg Intensity: {group['MeanIntensity'].mean():.4f}")

    # Check correlation between image size (Area) and being a new_whale
    # We convert boolean to int for correlation
    merged_df["target_binary"] = merged_df["is_new_whale"].astype(int)
    merged_df["Area"] = merged_df["Width"] * merged_df["Height"]

    corr_area = merged_df["Area"].corr(merged_df["target_binary"])
    corr_ratio = merged_df["AspectRatio"].corr(merged_df["target_binary"])
    corr_intensity = merged_df["MeanIntensity"].corr(merged_df["target_binary"])

    print("\nCorrelations with Target (1=New Whale, 0=Known):")
    print(f"  Image Area: {corr_area:.4f}")
    print(f"  Aspect Ratio: {corr_ratio:.4f}")
    print(f"  Mean Intensity: {corr_intensity:.4f}")

    print("-" * 30)


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df_train = pd.read_csv(METADATA_FILE)

    # Run Analyses
    analyze_target_variable(df_train)
    meta_df = analyze_image_data(df_train)
    analyze_relationships(df_train, meta_df)


if __name__ == "__main__":
    main()
