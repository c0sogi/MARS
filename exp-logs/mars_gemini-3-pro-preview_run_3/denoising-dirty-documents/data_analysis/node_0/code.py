import os
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    np.random.seed(seed)


def calculate_moments_from_hist(hist, total_count):
    """
    Calculate Mean, Std, Skewness, and Kurtosis from a histogram.
    hist: array of counts for values 0..255
    """
    values = np.arange(256)

    mean = np.average(values, weights=hist)
    variance = np.average((values - mean) ** 2, weights=hist)
    std = np.sqrt(variance)

    # Skewness: E[((X-mu)/sigma)^3]
    skew = np.average(((values - mean) / std) ** 3, weights=hist)

    # Kurtosis: E[((X-mu)/sigma)^4]
    kurt = np.average(((values - mean) / std) ** 4, weights=hist)

    return mean, std, skew, kurt


def run_eda():
    set_seed(SEED)

    print("Loading metadata...")
    try:
        df_train = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # --- Initialization of Accumulators ---

    # Dimensions
    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # Pixel Stats (Input)
    input_sum = 0.0
    input_sq_sum = 0.0
    input_pixel_count = 0

    # Pixel Stats (Target - Clean)
    # We use a histogram for target to calculate higher order moments efficiently
    target_hist = np.zeros(256, dtype=np.int64)
    target_pixel_count = 0

    # Noise Stats (Input - Target)
    noise_diff_sum = 0.0
    noise_diff_sq_sum = 0.0

    # Correlations
    correlations = []

    # Meta-features for relationship analysis
    # Store (area, target_mean_intensity)
    meta_relationships = []

    print(f"Analyzing {len(df_train)} training samples...")

    for idx, row in df_train.iterrows():
        input_rel_path = row["input_path"]
        target_rel_path = row["target_path"]

        input_full_path = os.path.join(INPUT_DIR, input_rel_path)
        target_full_path = os.path.join(INPUT_DIR, target_rel_path)

        # Load images
        # Use IMREAD_UNCHANGED to detect if it's grayscale or RGB
        img_in = cv2.imread(input_full_path, cv2.IMREAD_UNCHANGED)
        img_tar = cv2.imread(target_full_path, cv2.IMREAD_UNCHANGED)

        if img_in is None or img_tar is None:
            continue

        # --- 1. Dimensions & Channels ---
        h, w = img_in.shape[:2]
        c = 1 if len(img_in.shape) == 2 else img_in.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        channel_counts.append(c)

        # --- Preprocessing for Stats ---
        # Ensure images are grayscale for intensity analysis as per task description
        if c > 1:
            # Convert to grayscale for statistical analysis if RGB
            # Assuming standard RGB to Grayscale conversion
            img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            img_tar_gray = cv2.cvtColor(img_tar, cv2.COLOR_BGR2GRAY)
        else:
            img_in_gray = img_in
            img_tar_gray = img_tar

        # Flatten
        flat_in = img_in_gray.flatten().astype(np.float64)
        flat_tar = img_tar_gray.flatten().astype(np.float64)
        n_pixels = len(flat_in)

        # --- 2. Pixel Stats Accumulation ---

        # Input Stats
        input_sum += np.sum(flat_in)
        input_sq_sum += np.sum(flat_in**2)
        input_pixel_count += n_pixels

        # Target Stats (Histogram)
        # bincount is fast for uint8
        hist, _ = np.histogram(img_tar_gray, bins=256, range=(0, 256))
        target_hist += hist
        target_pixel_count += n_pixels

        # Noise Stats (Input - Target)
        diff = flat_in - flat_tar
        noise_diff_sum += np.sum(diff)
        noise_diff_sq_sum += np.sum(diff**2)

        # --- 3. Relationships ---

        # Correlation between Noisy and Clean for this image
        # Avoiding division by zero if image is constant color
        std_in = np.std(flat_in)
        std_tar = np.std(flat_tar)
        if std_in > 0 and std_tar > 0:
            corr = np.corrcoef(flat_in, flat_tar)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)  # Fallback

        # Meta-feature relationship: Area vs Mean Target Intensity
        mean_tar_intensity = np.mean(flat_tar)
        meta_relationships.append((w * h, mean_tar_intensity))

    # --- Final Calculations ---

    # Input Global Stats
    input_mean = input_sum / input_pixel_count
    input_var = (input_sq_sum / input_pixel_count) - (input_mean**2)
    input_std = np.sqrt(input_var)

    # Target Global Stats (from Histogram)
    target_mean, target_std, target_skew, target_kurt = calculate_moments_from_hist(
        target_hist, target_pixel_count
    )

    # Noise Stats
    noise_mean = noise_diff_sum / input_pixel_count
    noise_var = (noise_diff_sq_sum / input_pixel_count) - (noise_mean**2)
    noise_std = np.sqrt(noise_var)

    # Dimensions Stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Meta Relationships
    areas, tar_means = zip(*meta_relationships)
    # Correlation between Image Area and Mean Pixel Intensity
    meta_corr, _ = stats.pearsonr(areas, tar_means)

    # --- Report Generation ---

    print("\n==== TARGET VARIABLE ANALYSIS ====")
    print(f"Target Type: Image (Pixel Intensities 0-255)")
    print(f"Global Mean Intensity: {target_mean:.4f}")
    print(f"Global Std Deviation:  {target_std:.4f}")
    print(
        f"Distribution Skewness: {target_skew:.4f} (Positive = Tail on white side, Negative = Tail on black side)"
    )
    print(f"Distribution Kurtosis: {target_kurt:.4f} (Higher = Heavy tails/Outliers)")

    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")
    print("--- Dimensions ---")
    print(f"Count: {len(widths)}")
    print(
        f"Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    print("\n--- Channels ---")
    unique_channels, channel_counts_res = np.unique(channel_counts, return_counts=True)
    for c, count in zip(unique_channels, channel_counts_res):
        c_type = (
            "Grayscale"
            if c == 1
            else "RGB" if c == 3 else "RGBA" if c == 4 else f"{c}-channel"
        )
        print(f"{c_type} ({c}): {count} images")

    print("\n--- Pixel Statistics (Global) ---")
    print(f"Input Mean: {input_mean:.4f} (Normalized [0-1]: {input_mean/255.0:.4f})")
    print(f"Input Std:  {input_std:.4f}  (Normalized [0-1]: {input_std/255.0:.4f})")

    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")
    print("--- Noise Analysis (Input - Target) ---")
    print(
        f"Noise Mean: {noise_mean:.4f} (Positive implies Input is generally brighter than Target)"
    )
    print(f"Noise Std:  {noise_std:.4f} (Measure of noise intensity)")

    print("\n--- Input-Target Correlation ---")
    print(f"Average Pearson Correlation: {np.mean(correlations):.4f}")
    print(f"Min Correlation: {np.min(correlations):.4f}")
    print(f"Max Correlation: {np.max(correlations):.4f}")

    print("\n--- Metadata Relationships ---")
    print(f"Correlation (Image Area vs. Mean Target Intensity): {meta_corr:.4f}")
    if abs(meta_corr) < 0.1:
        print(
            "Interpretation: Negligible relationship between image size and text density."
        )
    elif meta_corr > 0:
        print(
            "Interpretation: Larger images tend to have higher intensity (more white space)."
        )
    else:
        print(
            "Interpretation: Larger images tend to have lower intensity (more text/darkness)."
        )


if __name__ == "__main__":
    run_eda()
