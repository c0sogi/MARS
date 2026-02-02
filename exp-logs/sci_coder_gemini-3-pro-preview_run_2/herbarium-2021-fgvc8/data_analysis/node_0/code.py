import os
import cv2
import numpy as np
import pandas as pd
import random
import concurrent.futures
from collections import Counter

# ==========================================
# Configuration & Constants
# ==========================================
INPUT_DIR = "./input"
TRAIN_METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = 5000  # Number of images to sample for detailed image analysis
SEED = 42
NUM_WORKERS = 12  # Utilize available vCPUs


# ==========================================
# Helper Functions
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def process_image(row):
    """
    Reads an image and extracts metadata and pixel stats.
    """
    image_id, rel_path, category_id = row
    full_path = os.path.join(INPUT_DIR, rel_path)

    try:
        # Get file size
        file_size = os.path.getsize(full_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, channels = img.shape

        # Pixel stats (normalize to 0-1 range for calculation)
        img_norm = img / 255.0
        mean_color = img_norm.mean(axis=(0, 1))
        std_color = img_norm.std(axis=(0, 1))

        return {
            "image_id": image_id,
            "category_id": category_id,
            "width": width,
            "height": height,
            "channels": channels,
            "aspect_ratio": width / height if height > 0 else 0,
            "file_size_bytes": file_size,
            "mean_r": mean_color[0],
            "mean_g": mean_color[1],
            "mean_b": mean_color[2],
            "std_r": std_color[0],
            "std_g": std_color[1],
            "std_b": std_color[2],
        }
    except Exception:
        return None


# ==========================================
# Main Analysis Execution
# ==========================================
def main():
    set_seed(SEED)

    # ------------------------------------------------------
    # 1. Data Integrity & Loading
    # ------------------------------------------------------
    print("DATA INTEGRITY")
    if not os.path.exists(TRAIN_METADATA_PATH):
        print(f"Error: Metadata file not found at {TRAIN_METADATA_PATH}")
        return

    df_train = pd.read_csv(TRAIN_METADATA_PATH)

    # Ensure analysis is strictly on training set
    print(f"Source: {TRAIN_METADATA_PATH}")
    print(f"Total Training Samples: {len(df_train)}")
    print(f"Columns: {list(df_train.columns)}")
    print("-" * 30)

    # ------------------------------------------------------
    # 2. Target Variable Analysis
    # ------------------------------------------------------
    print("TARGET VARIABLE ANALYSIS")

    # Target is 'category_id'
    class_counts = df_train["category_id"].value_counts()
    num_classes = len(class_counts)

    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()
    median_count = class_counts.median()

    print(f"Target Variable: category_id (Classification)")
    print(f"Number of Unique Classes: {num_classes}")
    print(f"Class Distribution Stats:")
    print(f"  Min Samples per Class: {min_count}")
    print(f"  Max Samples per Class: {max_count}")
    print(f"  Mean Samples per Class: {mean_count:.4f}")
    print(f"  Median Samples per Class: {median_count:.4f}")

    # Imbalance Ratio
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Top classes
    print(
        f"Top 5 Most Frequent Classes: {class_counts.head(5).index.tolist()} with counts {class_counts.head(5).values.tolist()}"
    )
    print("-" * 30)

    # ------------------------------------------------------
    # 3. Input Data Analysis (Image)
    # ------------------------------------------------------
    print("INPUT DATA ANALYSIS (IMAGE)")

    # Sample data for image processing
    if len(df_train) > SAMPLE_SIZE:
        df_sample = df_train.sample(n=SAMPLE_SIZE, random_state=SEED)
    else:
        df_sample = df_train

    print(f"Analyzing a random sample of {len(df_sample)} images...")

    # Prepare data for threading
    sample_rows = df_sample[["image_id", "file_path", "category_id"]].values.tolist()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Map process_image over the rows
        futures = [executor.submit(process_image, row) for row in sample_rows]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    df_img_stats = pd.DataFrame(results)

    if df_img_stats.empty:
        print("Error: No images could be processed.")
        return

    # Dimensions
    width_stats = df_img_stats["width"].describe()
    height_stats = df_img_stats["height"].describe()
    ar_stats = df_img_stats["aspect_ratio"].describe()

    print("Image Dimensions:")
    print(
        f"  Width  - Mean: {width_stats['mean']:.4f}, Std: {width_stats['std']:.4f}, Min: {width_stats['min']:.4f}, Max: {width_stats['max']:.4f}"
    )
    print(
        f"  Height - Mean: {height_stats['mean']:.4f}, Std: {height_stats['std']:.4f}, Min: {height_stats['min']:.4f}, Max: {height_stats['max']:.4f}"
    )
    print(f"  Aspect Ratio - Mean: {ar_stats['mean']:.4f}, Std: {ar_stats['std']:.4f}")

    # Channels
    channel_counts = df_img_stats["channels"].value_counts()
    print("Channel Distribution:")
    for ch, count in channel_counts.items():
        print(f"  {ch} Channels: {count} images ({count/len(df_img_stats)*100:.2f}%)")

    # Pixel Stats (Global)
    # Averaging the means of samples (approximation for EDA)
    global_mean_r = df_img_stats["mean_r"].mean()
    global_mean_g = df_img_stats["mean_g"].mean()
    global_mean_b = df_img_stats["mean_b"].mean()

    global_std_r = df_img_stats["std_r"].mean()
    global_std_g = df_img_stats["std_g"].mean()
    global_std_b = df_img_stats["std_b"].mean()

    print("Pixel Value Statistics (Normalized 0-1):")
    print(
        f"  Global Mean (R, G, B): ({global_mean_r:.4f}, {global_mean_g:.4f}, {global_mean_b:.4f})"
    )
    print(
        f"  Global Std  (R, G, B): ({global_std_r:.4f}, {global_std_g:.4f}, {global_std_b:.4f})"
    )
    print("-" * 30)

    # ------------------------------------------------------
    # 4. Feature/Signal Relationships
    # ------------------------------------------------------
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Unstructured (Meta-Feature) Relationships
    # Analyze if image size/aspect ratio varies by class
    # We take the top 5 most frequent classes in our SAMPLE to ensure we have enough data points
    top_classes_sample = df_img_stats["category_id"].value_counts().head(5).index

    print("Meta-Feature Analysis by Class (Top 5 Frequent Classes in Sample):")
    print(
        f"{'Class ID':<10} | {'Count':<6} | {'Avg Width':<10} | {'Avg Height':<10} | {'Avg AR':<10} | {'Avg FileSize(KB)':<16}"
    )
    print("-" * 80)

    for cls_id in top_classes_sample:
        subset = df_img_stats[df_img_stats["category_id"] == cls_id]
        count = len(subset)
        avg_w = subset["width"].mean()
        avg_h = subset["height"].mean()
        avg_ar = subset["aspect_ratio"].mean()
        avg_size = subset["file_size_bytes"].mean() / 1024.0

        print(
            f"{cls_id:<10} | {count:<6} | {avg_w:<10.4f} | {avg_h:<10.4f} | {avg_ar:<10.4f} | {avg_size:<16.4f}"
        )

    # Correlation between file size and image dimensions
    corr_size_width = df_img_stats["file_size_bytes"].corr(df_img_stats["width"])
    corr_size_height = df_img_stats["file_size_bytes"].corr(df_img_stats["height"])

    print("\nMeta-Feature Correlations:")
    print(f"  Correlation (File Size vs Width): {corr_size_width:.4f}")
    print(f"  Correlation (File Size vs Height): {corr_size_height:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
