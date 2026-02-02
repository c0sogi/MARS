import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_targets(df):
    print("SECTION 2: TARGET VARIABLE ANALYSIS")

    # Parse attribute_ids
    # Handle potential NaNs or non-strings by converting to string first, though metadata should be clean
    all_labels = []
    label_counts_per_image = []

    for x in df["attribute_ids"]:
        if pd.isna(x) or x == "":
            labels = []
        else:
            labels = str(x).split()

        all_labels.extend(labels)
        label_counts_per_image.append(len(labels))

    label_counts_per_image = np.array(label_counts_per_image)
    total_images = len(df)
    unique_labels = set(all_labels)
    label_counter = Counter(all_labels)

    # 2.1 Distribution of Labels (Multi-label specific)
    print(f"Total Training Images: {total_images}")
    print(f"Total Unique Labels: {len(unique_labels)}")
    print(f"Total Annotations: {len(all_labels)}")

    # 2.2 Label Cardinality (Labels per Image)
    print(f"Labels per Image - Mean: {np.mean(label_counts_per_image):.4f}")
    print(f"Labels per Image - Std: {np.std(label_counts_per_image):.4f}")
    print(f"Labels per Image - Min: {np.min(label_counts_per_image):.4f}")
    print(f"Labels per Image - Max: {np.max(label_counts_per_image):.4f}")

    # 2.3 Class Imbalance
    frequencies = list(label_counter.values())
    if len(frequencies) > 0:
        max_freq = np.max(frequencies)
        min_freq = np.min(frequencies)
        mean_freq = np.mean(frequencies)
        median_freq = np.median(frequencies)

        print(
            f"Label Frequency - Max (Most Common): {max_freq} ({max_freq/total_images:.4f} of images)"
        )
        print(
            f"Label Frequency - Min (Least Common): {min_freq} ({min_freq/total_images:.4f} of images)"
        )
        print(f"Label Frequency - Mean: {mean_freq:.4f}")
        print(f"Label Frequency - Median: {median_freq:.4f}")
        print(f"Imbalance Ratio (Max/Min): {max_freq/min_freq:.4f}")

        # Top 5 most common labels
        print("Top 5 Most Common Labels (ID: Count):")
        for lbl, count in label_counter.most_common(5):
            print(f"  {lbl}: {count}")
    else:
        print("No labels found.")

    return label_counts_per_image


def process_image(args):
    """
    Helper function to process a single image path.
    Returns: (width, height, channels, mean_color, std_color, file_size_bytes)
    """
    path, input_dir = args
    full_path = os.path.join(input_dir, path)

    try:
        # Get file size
        file_size = os.path.getsize(full_path)

        # Read image
        # cv2.imread loads as BGR
        img = cv2.imread(full_path)

        if img is None:
            return None

        h, w = img.shape[:2]

        if len(img.shape) == 2:
            c = 1  # Grayscale
            # Convert to RGB for consistent stats
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            c = img.shape[2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate pixel stats for this image (normalized 0-1)
        # We calculate per-image mean/std here to aggregate later
        # Note: Global mean/std is better calculated by accumulating sums,
        # but for EDA, averaging per-image stats is a reasonable approximation
        # to avoid massive memory usage or overflow with simple code.
        # However, let's do it slightly more rigorously: return sum and sum_sq.

        img_norm = img_rgb.astype(np.float32) / 255.0
        pixel_sum = np.sum(img_norm, axis=(0, 1))
        pixel_sq_sum = np.sum(img_norm**2, axis=(0, 1))
        num_pixels = h * w

        return (w, h, c, pixel_sum, pixel_sq_sum, num_pixels, file_size)

    except Exception:
        return None


def analyze_images(df, input_dir, sample_size=5000):
    print("SECTION 3: INPUT DATA ANALYSIS (IMAGES)")

    # Sample data to save time
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    print(f"Analyzing sample of {len(df_sample)} images...")

    paths = df_sample["file_path"].tolist()
    args = [(p, input_dir) for p in paths]

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    file_sizes = []

    # Accumulators for global pixel stats (R, G, B)
    total_pixel_sum = np.zeros(3)
    total_pixel_sq_sum = np.zeros(3)
    total_pixel_count = 0

    # Use ThreadPool for IO bound tasks
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_image, args))

    valid_results = [r for r in results if r is not None]

    for res in valid_results:
        w, h, c, p_sum, p_sq_sum, n_pix, f_size = res

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)
        channels.append(c)
        file_sizes.append(f_size)

        # Accumulate for global stats
        # If image was grayscale converted to RGB, it contributes to all 3 channels
        if len(p_sum) == 3:
            total_pixel_sum += p_sum
            total_pixel_sq_sum += p_sq_sum
        else:
            # Fallback if shape weirdness, though we forced RGB
            pass
        total_pixel_count += n_pix

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    file_sizes = np.array(file_sizes)

    # 3.1 Dimensions
    print(
        f"Image Widths - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
    )
    print(
        f"Image Heights - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
    )
    print(
        f"Aspect Ratios - Mean: {np.mean(aspect_ratios):.4f}, Std: {np.std(aspect_ratios):.4f}"
    )

    # 3.2 Channels
    c_counts = Counter(channels)
    print(f"Channel Distribution: {dict(c_counts)}")

    # 3.3 Pixel Stats (Global)
    if total_pixel_count > 0:
        global_mean = total_pixel_sum / total_pixel_count
        # Var = E[X^2] - (E[X])^2
        global_var = (total_pixel_sq_sum / total_pixel_count) - (global_mean**2)
        global_std = np.sqrt(np.maximum(global_var, 0))

        print(
            f"Global Pixel Mean (RGB): [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]"
        )
        print(
            f"Global Pixel Std (RGB):  [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]"
        )
    else:
        print("Could not compute pixel stats.")

    # Return meta features for correlation analysis
    # We need to align these with the sampled dataframe
    meta_features = pd.DataFrame(
        {
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "file_size": file_sizes,
        }
    )

    # We need to get the label counts for these specific sampled images
    # Since df_sample index is preserved, we can map back if needed,
    # but simpler to just re-extract label counts for the sample.
    sample_label_counts = []
    for x in df_sample["attribute_ids"]:
        if pd.isna(x) or x == "":
            sample_label_counts.append(0)
        else:
            sample_label_counts.append(len(str(x).split()))

    meta_features["label_count"] = sample_label_counts

    return meta_features


def analyze_relationships(meta_df):
    print("SECTION 4: FEATURE/SIGNAL RELATIONSHIPS")

    if meta_df is None or len(meta_df) == 0:
        print("No metadata available for relationship analysis.")
        return

    # 4.1 Correlations
    # Correlation between physical properties and number of labels
    corr_matrix = meta_df.corr(method="pearson")

    print("Correlation with Label Count:")
    target_corr = corr_matrix["label_count"].drop("label_count")
    for feature, val in target_corr.items():
        print(f"  {feature}: {val:.4f}")

    # Check for redundancy (collinearity)
    print("Highly Correlated Feature Pairs (> 0.90):")
    features = meta_df.columns
    found = False
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.90:
                print(f"  {features[i]} - {features[j]}: {c:.4f}")
                found = True
    if not found:
        print("  None found.")

    # 4.2 Meta-Feature Insights
    # Example: Do larger images (file size) tend to have more labels?
    # We can bin file size and check avg label count
    meta_df["size_bin"] = pd.qcut(
        meta_df["file_size"], q=4, labels=["Small", "Medium", "Large", "Very Large"]
    )
    print("Average Label Count by File Size Quartile:")
    print(
        meta_df.groupby("size_bin", observed=True)["label_count"]
        .mean()
        .to_string(float_format="{:.4f}".format)
    )


def main():
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train_metadata.csv"

    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return

    print("SECTION 1: DATA LOADING")
    df_train = pd.read_csv(METADATA_PATH)
    print(f"Loaded training metadata with {len(df_train)} rows.")

    # Run Target Analysis
    _ = analyze_targets(df_train)

    # Run Image Analysis
    # We pass the input directory to locate files
    meta_df = analyze_images(df_train, INPUT_DIR, sample_size=5000)

    # Run Relationship Analysis
    analyze_relationships(meta_df)

    print("EDA Complete.")


if __name__ == "__main__":
    main()
