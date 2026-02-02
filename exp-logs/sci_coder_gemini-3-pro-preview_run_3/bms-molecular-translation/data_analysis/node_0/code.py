import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for image statistics
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target_text(df):
    print("--- TARGET VARIABLE ANALYSIS (TEXT: InChI) ---")

    # 1. Sequence Lengths
    inchi_lengths = df["InChI"].str.len()
    print(f"Total Samples: {len(df)}")
    print(f"Sequence Length Mean: {inchi_lengths.mean():.4f}")
    print(f"Sequence Length Std:  {inchi_lengths.std():.4f}")
    print(f"Sequence Length Min:  {inchi_lengths.min():.4f}")
    print(f"Sequence Length Max:  {inchi_lengths.max():.4f}")

    # Outliers (IQR method)
    Q1 = inchi_lengths.quantile(0.25)
    Q3 = inchi_lengths.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers = inchi_lengths[inchi_lengths > upper_bound]
    print(
        f"Length Outliers (> {upper_bound:.2f}): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)"
    )

    # 2. Vocabulary Analysis
    # Concatenate a subset of text to estimate vocabulary if dataset is massive,
    # but for character-level InChI, we can likely scan unique chars efficiently.
    # We'll use a set comprehension over the series.
    unique_chars = set()
    # To keep it fast, we sample if > 100k rows, otherwise full.
    # Actually, InChI vocab is small (alphanumeric + symbols), so we can iterate full or large sample.
    # Let's iterate over the whole column to be accurate for the tokenizer.
    # Using Counter to find rare characters.
    vocab_counter = Counter()

    # Processing in chunks to avoid massive memory string concat
    chunk_size = 10000
    for i in range(0, len(df), chunk_size):
        chunk = df["InChI"].iloc[i : i + chunk_size]
        text_blob = "".join(chunk.values)
        vocab_counter.update(text_blob)

    unique_chars = sorted(vocab_counter.keys())
    print(f"Unique Character Vocabulary Size: {len(unique_chars)}")
    print(f"Vocabulary: {''.join(unique_chars)}")

    # Check for rare characters (< 0.001% of total characters)
    total_chars = sum(vocab_counter.values())
    rare_threshold = total_chars * 0.00001
    rare_chars = [
        char for char, count in vocab_counter.items() if count < rare_threshold
    ]
    print(f"Rare Characters (< 0.001% freq): {rare_chars}")

    return inchi_lengths


def analyze_input_images(df):
    print("\n--- INPUT DATA ANALYSIS (IMAGE) ---")

    # Sample the dataframe
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).copy()
    else:
        sample_df = df.copy()

    widths = []
    heights = []
    aspect_ratios = []
    channels = []

    # Pixel stats accumulators
    # We will compute online mean/std: E[x], E[x^2]
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0.0

    valid_indices = []

    print(f"Analyzing {len(sample_df)} sampled images...")

    for idx, row in sample_df.iterrows():
        # Construct full path
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read image
        try:
            # Read as is to detect channels
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            if len(img.shape) == 2:
                c = 1
            else:
                c = img.shape[2]

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            channels.append(c)

            # Normalize to 0-1 for stats
            img_norm = img.astype(np.float32) / 255.0

            pixel_sum += np.sum(img_norm)
            pixel_sq_sum += np.sum(img_norm**2)
            pixel_count += img_norm.size

            valid_indices.append(idx)

        except Exception as e:
            continue

    # Convert to numpy for stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    channels = np.array(channels)

    # 1. Dimensions
    print(
        f"Image Width:  Mean={widths.mean():.4f}, Std={widths.std():.4f}, Min={widths.min()}, Max={widths.max()}"
    )
    print(
        f"Image Height: Mean={heights.mean():.4f}, Std={heights.std():.4f}, Min={heights.min()}, Max={heights.max()}"
    )
    print(
        f"Aspect Ratio: Mean={aspect_ratios.mean():.4f}, Std={aspect_ratios.std():.4f}"
    )

    # 2. Channels
    unique_channels, counts = np.unique(channels, return_counts=True)
    print(f"Channel Distribution: {dict(zip(unique_channels, counts))}")

    # 3. Pixel Stats
    if pixel_count > 0:
        global_mean = pixel_sum / pixel_count
        global_var = (pixel_sq_sum / pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)
        print(f"Pixel Value Mean (0-1): {global_mean:.4f}")
        print(f"Pixel Value Std (0-1):  {global_std:.4f}")

    # Return metrics for relationship analysis
    # We need to align these with the sampled dataframe
    sample_df = sample_df.loc[valid_indices]
    sample_df["width"] = widths
    sample_df["height"] = heights
    sample_df["aspect_ratio"] = aspect_ratios
    sample_df["area"] = widths * heights

    return sample_df


def analyze_relationships(sample_df):
    print("\n--- FEATURE/SIGNAL RELATIONSHIPS ---")

    # Ensure we have the target length in the sample_df
    sample_df["inchi_len"] = sample_df["InChI"].str.len()

    # 1. Unstructured Relationships (Meta-Feature vs Target)
    # Correlation between Image Size and Sequence Length
    corr_area = sample_df["area"].corr(sample_df["inchi_len"], method="pearson")
    corr_ar = sample_df["aspect_ratio"].corr(sample_df["inchi_len"], method="pearson")

    print(f"Correlation (Image Area vs InChI Length): {corr_area:.4f}")
    print(f"Correlation (Aspect Ratio vs InChI Length): {corr_ar:.4f}")

    if abs(corr_area) > 0.3:
        print(
            "-> Insight: Larger images tend to contain more complex molecules (longer InChI strings)."
        )
    else:
        print("-> Insight: Image size is weakly correlated with molecule complexity.")

    # Check if outliers in length correspond to outliers in image size
    len_q3 = sample_df["inchi_len"].quantile(0.75)
    long_inchi_df = sample_df[sample_df["inchi_len"] > len_q3]
    avg_area_long = long_inchi_df["area"].mean()
    avg_area_all = sample_df["area"].mean()

    print(f"Avg Image Area (All): {avg_area_all:.2f}")
    print(f"Avg Image Area (Longest 25% InChI): {avg_area_long:.2f}")
    ratio = avg_area_long / avg_area_all if avg_area_all > 0 else 0
    print(f"Ratio (Long/All): {ratio:.4f}")


def run_eda():
    set_seed(RANDOM_SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Run Analyses
    analyze_target_text(df)
    sample_df_with_stats = analyze_input_images(df)
    analyze_relationships(sample_df_with_stats)


if __name__ == "__main__":
    run_eda()
