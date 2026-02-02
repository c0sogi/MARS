import os
import cv2
import pandas as pd
import numpy as np
import random
import warnings

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    target_col = "breed"
    class_counts = df[target_col].value_counts()

    n_classes = len(class_counts)
    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()
    std_count = class_counts.std()

    balance_ratio = max_count / min_count if min_count > 0 else 0

    print(f"Target Variable: {target_col}")
    print(f"Number of Classes: {n_classes}")
    print(f"Class Balance Ratio (Max/Min): {balance_ratio:.4f}")
    print(f"Samples per Class - Mean: {mean_count:.4f}, Std: {std_count:.4f}")
    print(f"Samples per Class - Min: {min_count}, Max: {max_count}")

    # Top and Bottom classes
    print(f"Most Frequent Class: {class_counts.idxmax()} ({max_count} samples)")
    print(f"Least Frequent Class: {class_counts.idxmin()} ({min_count} samples)")


def analyze_images(df):
    print("\nSECTION 2: INPUT DATA ANALYSIS (IMAGE)")

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = {}

    # Accumulators for pixel stats (R, G, B)
    # Using float64 to prevent overflow
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # To store meta-features for relationship analysis later
    meta_features = []

    # Iterate through all images
    # We assume file_path in metadata is relative to INPUT_DIR/.. or just needs INPUT_DIR prepended
    # The metadata generation script output shows paths like "train/<id>.jpg"
    # So we join INPUT_DIR with the relative path.

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)

        if img is None:
            continue

        # Shape: H, W, C
        h, w = img.shape[:2]
        c = 1 if len(img.shape) == 2 else img.shape[2]

        widths.append(w)
        heights.append(h)
        ar = w / h
        aspect_ratios.append(ar)

        # Channel counts
        channel_counts[c] = channel_counts.get(c, 0) + 1

        # Pixel Stats Calculation
        # Convert BGR to RGB if it's a color image
        if c == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img_rgb.reshape(-1, 3)
        elif c == 1:
            # Treat grayscale as repeating channels for consistent stats or handle separately
            # Standard practice: calculate for single channel.
            # But here we initialized accumulators for 3. Let's just track single channel on index 0
            # or replicate. Let's replicate to keep RGB structure valid or just note it.
            # Given the dataset is dogs, likely mostly RGB.
            # If grayscale, we'll just add to all 3 to represent 'intensity'.
            pixels = np.repeat(img.reshape(-1, 1), 3, axis=1)
        else:
            # RGBA or other, take first 3
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            pixels = img_rgb.reshape(-1, 3)

        n_p = pixels.shape[0]
        total_pixels += n_p

        # Accumulate
        pixel_sum += pixels.sum(axis=0)
        pixel_sq_sum += (pixels**2).sum(axis=0)

        # Store for next section
        meta_features.append(
            {
                "id": row["id"],
                "width": w,
                "height": h,
                "aspect_ratio": ar,
                "area": w * h,
            }
        )

    # Convert lists to arrays for stats
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Dimension Stats
    print(f"Image Count Processed: {len(widths)}")
    print(
        f"Width - Mean: {widths.mean():.4f}, Std: {widths.std():.4f}, Min: {widths.min()}, Max: {widths.max()}"
    )
    print(
        f"Height - Mean: {heights.mean():.4f}, Std: {heights.std():.4f}, Min: {heights.min()}, Max: {heights.max()}"
    )
    print(
        f"Aspect Ratio - Mean: {aspect_ratios.mean():.4f}, Std: {aspect_ratios.std():.4f}, Min: {aspect_ratios.min():.4f}, Max: {aspect_ratios.max():.4f}"
    )

    # Channel Stats
    print(f"Channel Distribution: {channel_counts}")

    # Global Pixel Stats
    if total_pixels > 0:
        # Mean = Sum / N
        global_mean = pixel_sum / total_pixels

        # Std = Sqrt( (SumSq / N) - Mean^2 )
        global_std = np.sqrt((pixel_sq_sum / total_pixels) - (global_mean**2))

        print(
            f"Pixel Mean (RGB): R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
        )
        print(
            f"Pixel Std (RGB):  R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
        )
    else:
        print("No pixels processed.")

    return pd.DataFrame(meta_features)


def analyze_relationships(df, meta_df):
    print("\nSECTION 3: FEATURE/SIGNAL RELATIONSHIPS")

    # Merge metadata with labels
    merged = pd.merge(df, meta_df, on="id")

    # Group by breed
    breed_stats = merged.groupby("breed")[
        ["width", "height", "aspect_ratio", "area"]
    ].mean()

    # Analyze if image properties vary by class
    # We look at the standard deviation of the means across breeds
    print("Relationship between Breed and Image Metadata (Aggregated by Class):")

    for col in ["width", "height", "aspect_ratio", "area"]:
        mean_of_means = breed_stats[col].mean()
        std_of_means = breed_stats[col].std()
        cv = std_of_means / mean_of_means if mean_of_means > 0 else 0

        print(
            f"Breed-wise Mean {col.capitalize()}: Global Avg={mean_of_means:.4f}, Std across breeds={std_of_means:.4f}, CV={cv:.4f}"
        )

    # Identify outliers in class-wise stats
    max_area_breed = breed_stats["area"].idxmax()
    min_area_breed = breed_stats["area"].idxmin()

    print(
        f"Breed with Largest Average Image Area: {max_area_breed} ({breed_stats.loc[max_area_breed, 'area']:.2f} pixels)"
    )
    print(
        f"Breed with Smallest Average Image Area: {min_area_breed} ({breed_stats.loc[min_area_breed, 'area']:.2f} pixels)"
    )

    # Correlation check (Encode breed? No, breed is nominal. We check if meta features predict breed essentially)
    # We can check if aspect ratio is a strong differentiator.
    # If std across breeds is low, then image size/shape doesn't tell us much about breed.

    print("\nInterpretation:")
    if breed_stats["aspect_ratio"].std() < 0.1:
        print("Aspect ratios are relatively consistent across breeds.")
    else:
        print(
            "Aspect ratios vary significantly between breeds, potentially acting as a signal."
        )


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load Data
    df = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    analyze_target(df)

    # 2. Image Analysis
    meta_df = analyze_images(df)

    # 3. Relationships
    analyze_relationships(df, meta_df)


if __name__ == "__main__":
    main()
