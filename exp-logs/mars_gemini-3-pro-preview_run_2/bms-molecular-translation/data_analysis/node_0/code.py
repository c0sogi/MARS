import os
import cv2
import numpy as np
import pandas as pd
import random
from collections import Counter
from scipy.stats import pearsonr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target_text(df):
    print("=" * 40)
    print("TARGET VARIABLE ANALYSIS (TEXT)")
    print("=" * 40)

    targets = df["InChI"].astype(str).tolist()
    lengths = [len(t) for t in targets]

    # Distribution of lengths
    print(f"Target: InChI String Lengths (Characters)")
    print(f"Count: {len(lengths)}")
    print(f"Mean: {np.mean(lengths):.4f}")
    print(f"Std Dev: {np.std(lengths):.4f}")
    print(f"Min: {np.min(lengths):.4f}")
    print(f"Max: {np.max(lengths):.4f}")

    # Percentiles
    p25, p50, p75, p95, p99 = np.percentile(lengths, [25, 50, 75, 95, 99])
    print(f"25th Percentile: {p25:.4f}")
    print(f"Median (50th): {p50:.4f}")
    print(f"75th Percentile: {p75:.4f}")
    print(f"95th Percentile: {p95:.4f}")
    print(f"99th Percentile: {p99:.4f}")

    # Vocabulary Analysis
    unique_chars = set()
    for t in targets:
        unique_chars.update(t)

    vocab_size = len(unique_chars)
    print(f"\nVocabulary Size (Unique Characters): {vocab_size}")
    print(f"Vocabulary: {sorted(list(unique_chars))}")

    # Structural Check
    prefix = "InChI=1S/"
    has_prefix = sum([1 for t in targets if t.startswith(prefix)])
    print(
        f"\nFormat Check: {has_prefix} out of {len(targets)} strings start with '{prefix}'"
    )
    print(
        f"Percentage complying with standard prefix: {(has_prefix/len(targets))*100:.4f}%"
    )

    return lengths


def analyze_input_images(df, input_dir, sample_size=5000):
    print("\n" + "=" * 40)
    print("INPUT DATA ANALYSIS (IMAGE)")
    print("=" * 40)

    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42).copy()
        print(
            f"Sampling {sample_size} images for statistical analysis out of {len(df)} total."
        )
    else:
        sample_df = df.copy()
        print(f"Using all {len(df)} images for analysis.")

    widths = []
    heights = []
    aspect_ratios = []
    pixel_means = []
    pixel_stds = []
    channels = []

    # Process images
    # We construct the full path using input_dir + relative path from metadata
    for _, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        try:
            # Load image unchanged to check channels
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Check channels
            if len(img.shape) == 2:
                c = 1
            else:
                c = img.shape[2]

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channels.append(c)

            # Pixel stats (normalize to 0-1 range for calculation)
            img_norm = img.astype(float) / 255.0
            pixel_means.append(np.mean(img_norm))
            pixel_stds.append(np.std(img_norm))

        except Exception as e:
            continue

    # Dimensions
    print("\n[Dimensions]")
    print(
        f"Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    # Channels
    print("\n[Channels]")
    chan_counts = Counter(channels)
    for c, count in chan_counts.items():
        print(f"Channel count {c}: {count} images ({(count/len(channels))*100:.2f}%)")

    # Pixel Stats
    print("\n[Pixel Intensity (0-1 Scale)]")
    print(f"Global Mean Intensity: {np.mean(pixel_means):.4f}")
    print(f"Global Std Intensity:  {np.mean(pixel_stds):.4f}")

    # Store stats in the dataframe for relationship analysis
    sample_df["width"] = widths
    sample_df["height"] = heights
    sample_df["aspect_ratio"] = aspect_ratios
    sample_df["pixel_mean"] = pixel_means

    return sample_df


def analyze_relationships(sample_df):
    print("\n" + "=" * 40)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 40)

    # Calculate target length for the sample
    sample_df["target_len"] = sample_df["InChI"].apply(len)
    sample_df["image_area"] = sample_df["width"] * sample_df["height"]

    print("\n[Unstructured Relationships]")
    print("Analyzing correlation between Image Meta-Features and Target Text Length.")

    # Correlation: Image Width vs Target Length
    corr_w, _ = pearsonr(sample_df["width"], sample_df["target_len"])
    print(f"Correlation (Width vs InChI Length): {corr_w:.4f}")

    # Correlation: Image Height vs Target Length
    corr_h, _ = pearsonr(sample_df["height"], sample_df["target_len"])
    print(f"Correlation (Height vs InChI Length): {corr_h:.4f}")

    # Correlation: Image Area vs Target Length
    corr_area, _ = pearsonr(sample_df["image_area"], sample_df["target_len"])
    print(f"Correlation (Area vs InChI Length): {corr_area:.4f}")

    # Correlation: Aspect Ratio vs Target Length
    corr_ar, _ = pearsonr(sample_df["aspect_ratio"], sample_df["target_len"])
    print(f"Correlation (Aspect Ratio vs InChI Length): {corr_ar:.4f}")

    print("\nInterpretation:")
    if abs(corr_area) > 0.3:
        print(
            "-> Moderate to strong correlation detected between image size and chemical formula length."
        )
        print("   Larger images likely contain more complex molecules.")
    else:
        print("-> Weak correlation between image size and formula length.")
        print(
            "   Molecule complexity may not strictly dictate image dimensions in this dataset."
        )


def main():
    set_seed(42)

    # Paths
    input_dir = "./input"
    metadata_path = "./metadata/train.csv"

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    # Load Metadata
    print(f"Loading training metadata from {metadata_path}...")
    df_train = pd.read_csv(metadata_path)

    # 1. Target Analysis
    analyze_target_text(df_train)

    # 2. Input Analysis (on a sample)
    # We pass the input_dir so the function can construct full paths
    sample_df_with_stats = analyze_input_images(df_train, input_dir, sample_size=5000)

    # 3. Relationship Analysis
    analyze_relationships(sample_df_with_stats)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
