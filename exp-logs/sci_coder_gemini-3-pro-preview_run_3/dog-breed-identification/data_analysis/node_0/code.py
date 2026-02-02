import os
import cv2
import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(SEED)

    # ==========================================
    # 1. Load Data
    # ==========================================
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Construct full file paths
    # The metadata contains relative paths like 'train/id.jpg'
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    breed_counts = df["breed"].value_counts()
    n_classes = len(breed_counts)
    min_count = breed_counts.min()
    max_count = breed_counts.max()
    mean_count = breed_counts.mean()
    std_count = breed_counts.std()

    print(f"Number of Classes: {n_classes}")
    print(f"Class Distribution Stats:")
    print(f"  Min Samples per Class: {min_count}")
    print(f"  Max Samples per Class: {max_count}")
    print(f"  Mean Samples per Class: {mean_count:.4f}")
    print(f"  Std Dev Samples per Class: {std_count:.4f}")

    # Imbalance Ratio
    imbalance_ratio = max_count / min_count
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Check for rare classes (arbitrary threshold, e.g., < 1% of total dataset)
    # Total samples
    n_samples = len(df)
    threshold = 0.01 * n_samples
    rare_classes = breed_counts[breed_counts < threshold]
    print(f"Number of Rare Classes (< 1% freq): {len(rare_classes)}")

    # ==========================================
    # 3. Input Data Analysis (Image Modality)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []

    # For pixel stats (Welford's algorithm or running sum for global mean/std)
    # We will use running sum for simplicity and speed on this dataset size
    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    # Iterate through images
    # We process all images to get accurate stats
    valid_images_count = 0

    for idx, row in df.iterrows():
        path = row["full_path"]

        # Get file size in bytes
        try:
            f_size = os.path.getsize(path)
            file_sizes.append(f_size)
        except OSError:
            file_sizes.append(0)
            continue

        # Load image
        # cv2.imread loads as BGR. We usually convert to RGB for ML.
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)

        # Pixel stats calculation
        # Normalize to 0-1 for calculation to avoid overflow, then scale back or report as is.
        # Standard practice is reporting on 0-255 scale or 0-1. We'll do 0-255.

        if c == 3:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_data = img_rgb.astype(np.float64)
        elif c == 1:
            # Treat grayscale as repeating 3 channels for global stats consistency
            # or just track it. Given most models use RGB, we simulate RGB stats.
            img_data = np.stack((img, img, img), axis=-1).astype(np.float64)
        else:
            # Unexpected channel count, skip pixel stats
            continue

        # Flatten H*W, 3
        pixels = img_data.reshape(-1, 3)
        n_pixels = pixels.shape[0]

        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels**2).sum(axis=0)
        total_pixel_count += n_pixels
        valid_images_count += 1

    # Convert lists to arrays
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    channels = np.array(channels)
    file_sizes = np.array(file_sizes)

    # Dimensions Analysis
    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"  Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"  Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    # Channel Analysis
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print(f"Channel Distribution: {dict(zip(unique_channels, channel_counts))}")

    # Pixel Stats
    if total_pixel_count > 0:
        global_mean = channel_sum / total_pixel_count
        # Var = E[X^2] - (E[X])^2
        global_var = (channel_sq_sum / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)

        print("Global Pixel Statistics (RGB, 0-255 range):")
        print(
            f"  Mean - R: {global_mean[0]:.4f}, G: {global_mean[1]:.4f}, B: {global_mean[2]:.4f}"
        )
        print(
            f"  Std  - R: {global_std[0]:.4f}, G: {global_std[1]:.4f}, B: {global_std[2]:.4f}"
        )
    else:
        print(
            "Global Pixel Statistics: Unable to calculate (no valid images processed)."
        )

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Create a temporary dataframe for analysis
    # Ensure lengths match (filter df to valid images if any were skipped)
    # Since we appended to lists, we need to handle potential index mismatches if images failed loading
    # Re-aligning is safer.

    # We will assume for this report that failures are negligible or we just use the lists
    # But to correlate with breed, we need alignment.
    # Let's rebuild the meta-features dataframe carefully.

    meta_features = []

    # Reset file pointer or just re-loop?
    # We already have lists aligned with the iteration order.
    # If we skipped any, the lists are shorter than df.
    # Let's assume we skipped `img is None`.

    # Efficient way: add columns to df based on valid indices, but we didn't track indices strictly.
    # Let's just zip the results. The loop order was deterministic (df.iterrows).
    # If `img is None` happened, we `continue`d, so lists are shorter.
    # We need to filter the original DF to match the successful loads.

    # Re-check logic:
    # We used `continue` on `img is None` or `OSError`.
    # To map back to breed, we should have stored breed in a list parallel to widths.

    # Let's redo the extraction of breed for the valid images.
    # Since we can't easily rewind, we will just note that for the correlation analysis,
    # we need the breed.

    # Strategy: We will create a list of dicts during a second quick pass or
    # (Better) we should have stored it in the first pass.
    # Since we can't edit the code above in the "thought" process, I will implement the fix in the final code block:
    # I will store `breeds_processed` list in the loop.

    # (Self-correction for implementation below)

    # Let's analyze relationship between Aspect Ratio and Breed.
    # Hypothesis: Some breeds might have distinct photography styles (e.g. long dogs vs tall dogs).

    # We'll use the data collected.
    # We need to reconstruct the dataframe for valid images.
    # Since I cannot modify the loop above in this text block, I will write the code to handle this correctly.

    # ... (Implementation detail: `breeds_processed` list added to loop) ...

    # Statistical Test: One-way ANOVA
    # H0: The mean aspect ratio is the same for all breeds.
    # H1: At least one breed has a different mean aspect ratio.

    # We need a dataframe of [breed, aspect_ratio, file_size]
    # We'll construct it in the code.

    # Placeholder for logic in code:
    # df_meta = pd.DataFrame({'breed': breeds_processed, 'aspect_ratio': aspect_ratios, 'file_size': file_sizes})
    # group_means = df_meta.groupby('breed')[['aspect_ratio', 'file_size']].mean()

    # Calculate correlation between image size and breed?
    # Breed is categorical. We can check if file_size varies by breed.

    # Report top 5 breeds with largest avg file size and smallest.
    # Report top 5 breeds with largest avg aspect ratio and smallest.

    pass  # Logic continues in code block


if __name__ == "__main__":
    # Redefine main to include the logic discussed

    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        exit(1)

    df = pd.read_csv(METADATA_PATH)
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    # 2. Target Analysis
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)
    breed_counts = df["breed"].value_counts()
    print(f"Number of Classes: {len(breed_counts)}")
    print(f"Class Distribution Stats:")
    print(f"  Min: {breed_counts.min()}")
    print(f"  Max: {breed_counts.max()}")
    print(f"  Mean: {breed_counts.mean():.4f}")
    print(f"  Std: {breed_counts.std():.4f}")
    print(f"Class Imbalance Ratio: {breed_counts.max() / breed_counts.min():.4f}")

    # 3. Image Analysis
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("-" * 30)

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []
    breeds_processed = []  # To maintain alignment

    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    for idx, row in df.iterrows():
        path = row["full_path"]
        breed = row["breed"]

        try:
            f_size = os.path.getsize(path)
        except OSError:
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)
        file_sizes.append(f_size)
        breeds_processed.append(breed)

        # Pixel stats
        if c == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_data = img_rgb.astype(np.float64)
        elif c == 1:
            img_data = np.stack((img, img, img), axis=-1).astype(np.float64)
        else:
            continue

        pixels = img_data.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels**2).sum(axis=0)
        total_pixel_count += pixels.shape[0]

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"  Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"  Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}"
    )

    unique_c, counts_c = np.unique(channels, return_counts=True)
    print(f"Channel Distribution: {dict(zip(unique_c, counts_c))}")

    if total_pixel_count > 0:
        global_mean = channel_sum / total_pixel_count
        global_var = (channel_sq_sum / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(global_var)
        print("Global Pixel Statistics (RGB):")
        print(
            f"  Mean - R: {global_mean[0]:.4f}, G: {global_mean[1]:.4f}, B: {global_mean[2]:.4f}"
        )
        print(
            f"  Std  - R: {global_std[0]:.4f}, G: {global_std[1]:.4f}, B: {global_std[2]:.4f}"
        )

    # 4. Feature Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    df_meta = pd.DataFrame(
        {
            "breed": breeds_processed,
            "aspect_ratio": aspect_ratios,
            "file_size": file_sizes,
            "width": widths,
            "height": heights,
        }
    )

    # Group by breed
    breed_stats = df_meta.groupby("breed").agg(
        {"aspect_ratio": "mean", "file_size": "mean", "width": "mean", "height": "mean"}
    )

    print("Relationship: Metadata vs Target (Breed)")

    # Check for variance in Aspect Ratio across breeds
    # Using Kruskal-Wallis H-test (non-parametric ANOVA) as distributions might not be normal
    # Prepare samples
    samples = [group["aspect_ratio"].values for name, group in df_meta.groupby("breed")]
    if len(samples) > 1:
        stat, p_value = stats.kruskal(*samples)
        print(f"Kruskal-Wallis Test for Aspect Ratio across Breeds:")
        print(f"  H-statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  Result: Significant difference in aspect ratios between breeds.")
        else:
            print(
                "  Result: No significant difference in aspect ratios between breeds."
            )

    # Correlation between File Size and Image Dimensions (sanity check)
    corr_size_width = df_meta["file_size"].corr(df_meta["width"])
    print(f"Correlation (File Size vs Width): {corr_size_width:.4f}")

    # Top 3 Breeds by Average File Size
    print("\nTop 3 Breeds by Average File Size (bytes):")
    print(breed_stats["file_size"].sort_values(ascending=False).head(3))

    # Top 3 Breeds by Average Aspect Ratio
    print("\nTop 3 Breeds by Average Aspect Ratio:")
    print(breed_stats["aspect_ratio"].sort_values(ascending=False).head(3))
