import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_FILE = os.path.join(METADATA_DIR, "train_metadata.csv")
SAMPLE_SIZE = (
    5000  # Number of images to sample for pixel/dimension analysis to keep runtime low
)


def perform_eda():
    print("STARTING EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 50)

    # 1. Load Metadata
    # We strictly use the training set metadata generated previously
    if not os.path.exists(TRAIN_METADATA_FILE):
        print(f"Error: Metadata file {TRAIN_METADATA_FILE} not found.")
        return

    df = pd.read_csv(TRAIN_METADATA_FILE)

    # 2. Target Variable Analysis (Text Modality)
    print("\nSECTION 1: TARGET VARIABLE ANALYSIS (TEXT)")
    print("-" * 50)

    # InChI strings are the target. We analyze them as text sequences.
    target_col = "InChI"

    # Length Analysis
    df["seq_len"] = df[target_col].astype(str).apply(len)

    mean_len = df["seq_len"].mean()
    std_len = df["seq_len"].std()
    min_len = df["seq_len"].min()
    max_len = df["seq_len"].max()

    print(f"Target (InChI) Sequence Lengths:")
    print(f"  Count: {len(df)}")
    print(f"  Mean Length: {mean_len:.4f}")
    print(f"  Std Dev: {std_len:.4f}")
    print(f"  Min Length: {min_len}")
    print(f"  Max Length: {max_len}")

    # Vocabulary Analysis
    # We'll use a Counter to find unique characters and their frequencies
    # To save memory/time on huge datasets, we can sample if needed, but for text length analysis
    # usually full pass is fast enough. Let's do full pass for vocab.

    all_text = "".join(df[target_col].astype(str).tolist())
    vocab_counter = Counter(all_text)
    vocab = sorted(vocab_counter.keys())
    vocab_size = len(vocab)

    print(f"\nVocabulary Analysis:")
    print(f"  Unique Character Count: {vocab_size}")
    print(f"  Characters: {''.join(vocab)}")

    # Check for Out of Vocabulary potential isn't strictly applicable without a reference vocab,
    # but we can list rare characters.
    rare_chars = [
        char for char, count in vocab_counter.items() if count < len(df) * 0.001
    ]  # appearing in < 0.1% of average length * rows roughly
    # Actually let's just show least common
    least_common = vocab_counter.most_common()[:-6:-1]
    print(f"  Least Common Characters: {least_common}")

    # 3. Input Data Analysis (Image Modality)
    print("\nSECTION 2: INPUT DATA ANALYSIS (IMAGE)")
    print("-" * 50)

    # We sample the dataset to perform image analysis to respect the runtime limit.
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    print(f"Sampling {len(sample_df)} images for detailed pixel/dimension analysis...")

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # Accumulators for pixel stats (Welford's or simple sum for approx)
    # We will use simple sum accumulation for mean, and sum of squares for std
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    # To check channel consistency
    channel_counts = Counter()

    for _, row in sample_df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # Read image
        # cv2.imread loads as BGR. If flags=cv2.IMREAD_UNCHANGED, it loads as is.
        # Chemical images are often grayscale but saved as PNG which might be RGB or Palette.
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        # Dimensions
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)
        channel_counts[c] += 1

        # Pixel Stats
        # Normalize to 0-1 for calculation
        if c == 1:
            flat_pixels = img.flatten() / 255.0
        else:
            # If RGB, flatten all channels together for global stats or handle separately.
            # Requirement asks for global mean/std.
            flat_pixels = img.flatten() / 255.0

        pixel_sum += np.sum(flat_pixels)
        pixel_sq_sum += np.sum(flat_pixels**2)
        total_pixels += len(flat_pixels)

    # Calculate Dimension Stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print(f"\nImage Dimensions (Sampled N={len(widths)}):")
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
    print(f"\nChannel Distribution:")
    for c, count in channel_counts.items():
        type_str = (
            "Grayscale"
            if c == 1
            else ("RGB" if c == 3 else "RGBA" if c == 4 else "Unknown")
        )
        print(
            f"  {c} Channels ({type_str}): {count} images ({(count/len(widths))*100:.2f}%)"
        )

    # Pixel Stats
    if total_pixels > 0:
        global_mean = pixel_sum / total_pixels
        global_var = (pixel_sq_sum / total_pixels) - (global_mean**2)
        global_std = np.sqrt(global_var)
        print(f"\nPixel Statistics (Normalized 0-1):")
        print(f"  Global Mean: {global_mean:.4f}")
        print(f"  Global Std Dev: {global_std:.4f}")
    else:
        print("\nPixel Statistics: No data processed.")

    # 4. Feature/Signal Relationships
    print("\nSECTION 3: FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 50)

    # We analyze the relationship between Image Metadata (H, W, AR) and Target Metadata (Sequence Length)
    # Add image stats to the sample dataframe
    sample_df = sample_df.iloc[
        : len(widths)
    ].copy()  # Ensure alignment if some files failed
    sample_df["img_width"] = widths
    sample_df["img_height"] = heights
    sample_df["img_aspect_ratio"] = aspect_ratios

    # Correlations
    # We are looking for: Do larger images contain longer InChI strings?
    correlations = sample_df[
        ["seq_len", "img_width", "img_height", "img_aspect_ratio"]
    ].corr(method="pearson")

    print("Correlations with Target Sequence Length:")
    print(f"  Width vs Seq Length: {correlations.loc['seq_len', 'img_width']:.4f}")
    print(f"  Height vs Seq Length: {correlations.loc['seq_len', 'img_height']:.4f}")
    print(
        f"  Aspect Ratio vs Seq Length: {correlations.loc['seq_len', 'img_aspect_ratio']:.4f}"
    )

    # Interpretation
    w_corr = correlations.loc["seq_len", "img_width"]
    if abs(w_corr) > 0.5:
        print(
            "  -> Strong relationship detected: Molecule complexity (string length) correlates with image width."
        )
    elif abs(w_corr) > 0.2:
        print("  -> Moderate relationship detected between width and sequence length.")
    else:
        print(
            "  -> Weak or no linear relationship between image size and sequence length."
        )

    print("\n" + "=" * 50)
    print("EDA COMPLETE")


if __name__ == "__main__":
    perform_eda()
