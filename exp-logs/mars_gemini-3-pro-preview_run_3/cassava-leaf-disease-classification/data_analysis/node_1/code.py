import os
import cv2
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train.csv"
INPUT_ROOT = "./input"
SEED = 42


def set_seed(seed):
    np.random.seed(seed)


def get_image_stats(row_tuple):
    """
    Worker function to extract stats from a single image.
    row_tuple: (index, row_series) or just row_dict
    """
    # Unpack the row. Since we iterate over rows, we expect a namedtuple or series-like object
    # We will pass a dictionary for simplicity
    file_rel_path = row_tuple["file_path"]
    full_path = os.path.join(INPUT_ROOT, file_rel_path)

    stats = {
        "width": np.nan,
        "height": np.nan,
        "channels": np.nan,
        "aspect_ratio": np.nan,
        "mean_pixel": np.nan,
        "std_pixel": np.nan,
        "valid": False,
    }

    try:
        # Load image
        # cv2.imread returns None if file not found or invalid
        img = cv2.imread(full_path)

        if img is not None:
            # Shape: H, W, C
            h, w, c = img.shape

            # Convert to RGB for consistent pixel stats (OpenCV is BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mean_val = np.mean(img_rgb)
            std_val = np.std(img_rgb)

            stats["width"] = w
            stats["height"] = h
            stats["channels"] = c
            stats["aspect_ratio"] = w / h if h > 0 else 0
            stats["mean_pixel"] = mean_val
            stats["std_pixel"] = std_val
            stats["valid"] = True

    except Exception:
        pass

    return stats


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # === TARGET VARIABLE ANALYSIS ===
    print("=== TARGET VARIABLE ANALYSIS ===")

    # Distribution
    label_counts = df_train["label"].value_counts().sort_index()
    total_count = len(df_train)

    print(f"Total Samples: {total_count}")
    print("Class Distribution:")
    for label, count in label_counts.items():
        ratio = count / total_count
        print(f"Class {label}: {count} ({ratio:.4f})")

    # Imbalance
    if not label_counts.empty:
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        print(f"Class Balance Ratio (Max/Min): {imbalance_ratio:.4f}")
    else:
        print("Class Balance Ratio: N/A (Empty dataset)")

    # === INPUT DATA ANALYSIS (IMAGE) ===
    print("\n=== INPUT DATA ANALYSIS (IMAGE) ===")

    # Prepare data for multiprocessing
    # Convert dataframe rows to list of dicts for pickling
    rows_data = df_train.to_dict("records")

    # Process sequentially to avoid pickling errors in the exec environment
    results = [get_image_stats(row) for row in rows_data]

    # Convert results to DataFrame
    stats_df = pd.DataFrame(results)

    # Filter valid images
    valid_stats = stats_df[stats_df["valid"] == True].copy()
    n_valid = len(valid_stats)
    n_invalid = len(stats_df) - n_valid

    if n_invalid > 0:
        print(f"Warning: {n_invalid} images could not be processed.")

    if n_valid == 0:
        print("No valid images found for analysis.")
        return

    # Dimensions
    print("Dimensions:")
    print(
        f"Width:  Mean={valid_stats['width'].mean():.4f}, Std={valid_stats['width'].std():.4f}, "
        f"Min={valid_stats['width'].min():.4f}, Max={valid_stats['width'].max():.4f}"
    )
    print(
        f"Height: Mean={valid_stats['height'].mean():.4f}, Std={valid_stats['height'].std():.4f}, "
        f"Min={valid_stats['height'].min():.4f}, Max={valid_stats['height'].max():.4f}"
    )

    # Aspect Ratios
    print("Aspect Ratios:")
    print(
        f"Mean={valid_stats['aspect_ratio'].mean():.4f}, Std={valid_stats['aspect_ratio'].std():.4f}, "
        f"Min={valid_stats['aspect_ratio'].min():.4f}, Max={valid_stats['aspect_ratio'].max():.4f}"
    )

    # Channels
    print("Channels:")
    channel_counts = valid_stats["channels"].value_counts().sort_index()
    for ch, count in channel_counts.items():
        print(f"Channel {int(ch)}: {count} images ({count/n_valid:.4f})")

    # Pixel Stats
    print("Pixel Stats (Global approx):")
    # Note: This is the mean of means, which is a sufficient approximation for EDA
    # unless image sizes vary drastically and are correlated with brightness.
    print(f"Global Mean Pixel Value: {valid_stats['mean_pixel'].mean():.4f}")
    print(f"Global Std Pixel Value:  {valid_stats['std_pixel'].mean():.4f}")

    # === FEATURE/SIGNAL RELATIONSHIPS ===
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")
    print("Unstructured (Meta-Feature) Relationships:")

    # Merge stats back with labels
    # We assume the order is preserved in pool.map (it is guaranteed by multiprocessing.Pool)
    # We concat horizontally.
    merged_df = pd.concat([df_train, stats_df], axis=1)

    # Filter only valid ones for correlation analysis
    merged_valid = merged_df[merged_df["valid"] == True]

    # Group by label to see if metadata varies by class
    # Selecting relevant meta-features
    meta_features = ["width", "height", "aspect_ratio", "mean_pixel", "std_pixel"]

    grouped_means = merged_valid.groupby("label")[meta_features].mean()

    print("Mean Metadata Values per Class:")
    # Format the dataframe output manually for cleaner look
    header = f"{'Label':<6} | {'Width':<10} | {'Height':<10} | {'Aspect':<10} | {'Mean Pix':<10} | {'Std Pix':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for label, row in grouped_means.iterrows():
        print(
            f"{label:<6} | {row['width']:<10.4f} | {row['height']:<10.4f} | {row['aspect_ratio']:<10.4f} | {row['mean_pixel']:<10.4f} | {row['std_pixel']:<10.4f}"
        )

    # Check for specific correlations (e.g. brightness vs class)
    # Since class is categorical, we look for large deviations in the table above.

    # Calculate correlation between meta-features and target (just to see if any linear relationship exists with label ID,
    # though label ID is nominal, it gives a rough sense if there's a drift)
    # A better metric for nominal vs continuous is Point Biserial or ANOVA, but we'll stick to the grouped report above as primary.
    # We can report the standard deviation of the group means to see if they differ significantly.

    print(
        "\nVariance of Metadata Means across Classes (Indicator of discriminative potential):"
    )
    means_std = grouped_means.std()
    for feat, val in means_std.items():
        print(f"{feat}: {val:.4f}")


if __name__ == "__main__":
    main()
