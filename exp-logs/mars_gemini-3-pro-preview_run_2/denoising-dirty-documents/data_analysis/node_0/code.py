import os
import cv2
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_eda():
    print("Starting Exploratory Data Analysis...")
    set_seed()

    # Paths
    metadata_path = "./metadata/train.csv"
    input_dir = "./input"

    # Load Metadata
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df_train = pd.read_csv(metadata_path)

    # Data Containers
    clean_pixel_samples = (
        []
    )  # Store a subset if too large, but with 220GB RAM, we can store a lot.
    noisy_pixel_samples = []

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    meta_areas = []
    meta_mean_clean_intensities = []

    # Iterate through training data
    # We will load all images.
    # To ensure we don't hit memory issues if images are massive, we'll process stats incrementally or list-append.
    # Given 92 images, full loading is fine.

    all_noisy_flat = []
    all_clean_flat = []

    for idx, row in df_train.iterrows():
        f_path = os.path.join(input_dir, row["feature_path"])
        l_path = os.path.join(input_dir, row["label_path"])

        # Read images
        # IMREAD_UNCHANGED to detect if it's 1 channel or 3
        img_noisy = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        img_clean = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)

        if img_noisy is None or img_clean is None:
            continue

        # Dimensions
        if len(img_noisy.shape) == 2:
            h, w = img_noisy.shape
            c = 1
        else:
            h, w, c = img_noisy.shape

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channel_counts.append(c)

        # Convert to Grayscale for Intensity Analysis if necessary
        # The task specifies grayscale pixel intensities.
        if c == 3:
            # Convert to gray
            img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)
            img_clean_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
        elif c == 4:
            # Handle RGBA if present (unlikely for scanned text but possible)
            img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGRA2GRAY)
            img_clean_gray = cv2.cvtColor(img_clean, cv2.COLOR_BGRA2GRAY)
        else:
            img_noisy_gray = img_noisy
            img_clean_gray = img_clean

        # Normalize to 0-1 range
        flat_noisy = img_noisy_gray.flatten().astype(np.float32) / 255.0
        flat_clean = img_clean_gray.flatten().astype(np.float32) / 255.0

        all_noisy_flat.append(flat_noisy)
        all_clean_flat.append(flat_clean)

        # Meta features for relationship analysis
        meta_areas.append(w * h)
        meta_mean_clean_intensities.append(np.mean(flat_clean))

    # Concatenate all pixels
    # Using a subset for distribution analysis if total count is massive to save time,
    # but exact calculation is preferred if fast enough.
    # 92 images * ~1M pixels = ~100M floats. fast enough for numpy.

    total_noisy = np.concatenate(all_noisy_flat)
    total_clean = np.concatenate(all_clean_flat)

    # --- 2. Target Variable Analysis (Clean Images) ---
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)

    target_mean = np.mean(total_clean)
    target_std = np.std(total_clean)
    target_min = np.min(total_clean)
    target_max = np.max(total_clean)

    # Skew and Kurtosis
    # These can be slow on 100M items. We can sample 1M pixels for robust estimation.
    sample_indices = np.random.choice(
        len(total_clean), size=min(1000000, len(total_clean)), replace=False
    )
    target_sample = total_clean[sample_indices]

    target_skew = skew(target_sample)
    target_kurt = kurtosis(target_sample)

    print(f"Target (Clean Pixel) Mean: {target_mean:.4f}")
    print(f"Target (Clean Pixel) Std:  {target_std:.4f}")
    print(f"Target (Clean Pixel) Min:  {target_min:.4f}")
    print(f"Target (Clean Pixel) Max:  {target_max:.4f}")
    print(f"Target Skewness:           {target_skew:.4f}")
    print(f"Target Kurtosis:           {target_kurt:.4f}")

    # --- 3. Input Data Analysis (Image Data) ---
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    # Dimensions
    print(f"Image Count: {len(widths)}")
    print(
        f"Width  (Mean ± Std): {np.mean(widths):.4f} ± {np.std(widths):.4f} [Min: {np.min(widths)}, Max: {np.max(widths)}]"
    )
    print(
        f"Height (Mean ± Std): {np.mean(heights):.4f} ± {np.std(heights):.4f} [Min: {np.min(heights)}, Max: {np.max(heights)}]"
    )
    print(f"Aspect Ratio Mean:   {np.mean(aspect_ratios):.4f}")

    # Channels
    unique_channels, counts_channels = np.unique(channel_counts, return_counts=True)
    channel_dist_str = ", ".join(
        [f"{ch}ch: {cnt}" for ch, cnt in zip(unique_channels, counts_channels)]
    )
    print(f"Channel Distribution: {channel_dist_str}")

    # Pixel Stats (Noisy)
    noisy_mean = np.mean(total_noisy)
    noisy_std = np.std(total_noisy)
    print(f"Global Pixel Mean (Noisy): {noisy_mean:.4f}")
    print(f"Global Pixel Std (Noisy):  {noisy_std:.4f}")

    # --- 4. Feature/Signal Relationships ---
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Correlation between Noisy and Clean Pixels
    # Using the sample indices from before for speed
    noisy_sample = total_noisy[sample_indices]
    pixel_corr, _ = pearsonr(noisy_sample, target_sample)
    print(f"Pixel-wise Correlation (Noisy vs Clean): {pixel_corr:.4f}")

    # Noise Analysis (Difference)
    # Noise = Noisy - Clean
    noise_diff = noisy_sample - target_sample
    noise_mean = np.mean(noise_diff)
    noise_std = np.std(noise_diff)
    print(f"Noise Residual Mean (Noisy - Clean):     {noise_mean:.4f}")
    print(f"Noise Residual Std:                      {noise_std:.4f}")

    # Meta-Feature Relationship
    # Correlation between Image Area (W*H) and Mean Clean Intensity
    # Does the size of the image predict how much 'white space' (high intensity) there is?
    if len(meta_areas) > 1:
        meta_corr, _ = pearsonr(meta_areas, meta_mean_clean_intensities)
        print(f"Correlation (Image Area vs Mean Intensity): {meta_corr:.4f}")
    else:
        print("Correlation (Image Area vs Mean Intensity): N/A (Insufficient data)")


if __name__ == "__main__":
    run_eda()
