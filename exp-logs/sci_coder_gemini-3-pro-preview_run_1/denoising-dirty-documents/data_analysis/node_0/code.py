import os
import cv2
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import random

# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    # 1. Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file {METADATA_FILE} not found.")
        return

    df = pd.read_csv(METADATA_FILE)

    # 2. Data Structures for Analysis
    # Target stats (Clean images)
    clean_pixels_sample = []

    # Input stats (Noisy images)
    noisy_pixels_sample = []
    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []

    # Meta-features for relationship analysis
    meta_areas = []
    meta_noisy_means = []
    meta_clean_means = []

    # 3. Process Images
    # We iterate through the training set defined in metadata
    for _, row in df.iterrows():
        noisy_path = os.path.join(INPUT_DIR, row["noisy_image_path"])
        clean_path = os.path.join(INPUT_DIR, row["clean_image_path"])

        # Load images (Load unchanged to detect original channels)
        img_n_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
        img_c_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)

        if img_n_raw is None or img_c_raw is None:
            continue

        # Dimensions & Channels
        h, w = img_n_raw.shape[:2]
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        meta_areas.append(w * h)

        # Check channels
        if len(img_n_raw.shape) == 2:
            channel_counts.append(1)
        else:
            channel_counts.append(img_n_raw.shape[2])

        # Preprocessing for Pixel Stats
        # Convert to grayscale if necessary (as task is grayscale denoising)
        if len(img_n_raw.shape) > 2:
            img_n_gray = cv2.cvtColor(img_n_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_n_gray = img_n_raw

        if len(img_c_raw.shape) > 2:
            img_c_gray = cv2.cvtColor(img_c_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_c_gray = img_c_raw

        # Normalize to [0, 1] float
        norm_n = img_n_gray.astype(float) / 255.0
        norm_c = img_c_gray.astype(float) / 255.0

        # Calculate Image-Level Means
        meta_noisy_means.append(np.mean(norm_n))
        meta_clean_means.append(np.mean(norm_c))

        # Subsample pixels for global distribution analysis
        # Taking 5% of pixels per image to keep memory usage low while maintaining statistical significance
        flat_n = norm_n.flatten()
        flat_c = norm_c.flatten()

        sample_size = int(len(flat_n) * 0.05)
        if sample_size > 0:
            indices = np.random.choice(len(flat_n), sample_size, replace=False)
            noisy_pixels_sample.extend(flat_n[indices])
            clean_pixels_sample.extend(flat_c[indices])

    # Convert lists to numpy arrays for calculation
    clean_pixels_sample = np.array(clean_pixels_sample)
    noisy_pixels_sample = np.array(noisy_pixels_sample)

    # 4. Generate Report

    # --- Target Variable Analysis ---
    print("TARGET VARIABLE ANALYSIS")
    print(f"Distribution Mean: {np.mean(clean_pixels_sample):.4f}")
    print(f"Distribution Std: {np.std(clean_pixels_sample):.4f}")
    # Skewness and Kurtosis to assess normality of the regression target (pixel intensity)
    print(f"Skewness: {skew(clean_pixels_sample):.4f}")
    print(f"Kurtosis: {kurtosis(clean_pixels_sample):.4f}")

    # --- Input Data Analysis ---
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Dimensions
    print(f"Width Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}")
    print(f"Height Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}")
    print(
        f"Aspect Ratio Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    # Channels
    unique_ch, counts_ch = np.unique(channel_counts, return_counts=True)
    ch_dist = {str(k): int(v) for k, v in zip(unique_ch, counts_ch)}
    print(f"Channel Distribution: {ch_dist}")

    # Pixel Stats
    print(f"Global Pixel Mean: {np.mean(noisy_pixels_sample):.4f}")
    print(f"Global Pixel Std: {np.std(noisy_pixels_sample):.4f}")

    # --- Feature/Signal Relationships ---
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Relationship 1: Correlation between Noisy Mean Intensity and Clean Mean Intensity
    # High correlation implies the noise preserves the global brightness structure.
    corr_means = np.corrcoef(meta_noisy_means, meta_clean_means)[0, 1]
    print(f"Correlation (Noisy Mean vs Clean Mean): {corr_means:.4f}")

    # Relationship 2: Correlation between Image Area and Clean Mean Intensity
    # Checks if larger images tend to have different brightness distributions (e.g. more whitespace).
    corr_area_mean = np.corrcoef(meta_areas, meta_clean_means)[0, 1]
    print(f"Correlation (Image Area vs Clean Mean): {corr_area_mean:.4f}")


if __name__ == "__main__":
    main()
