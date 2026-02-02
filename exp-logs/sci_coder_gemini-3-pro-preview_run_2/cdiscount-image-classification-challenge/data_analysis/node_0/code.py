import os
import struct
import pandas as pd
import numpy as np
import cv2
import io
import time
from collections import Counter

# ==== Configuration & Setup ====
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
CAT_NAMES = os.path.join(INPUT_DIR, "category_names.csv")

# Sample sizes for analysis to keep runtime within limits
GENERAL_SAMPLE_SIZE = 10000  # For dimensions, counts, file sizes
PIXEL_SAMPLE_SIZE = 2000  # For pixel mean/std calculation

# Random Seed
SEED = 42
np.random.seed(SEED)

# ==== BSON Parsing Utilities ====
# BSON Type Constants
TYPE_DOUBLE = 1
TYPE_STRING = 2
TYPE_DOC = 3
TYPE_ARRAY = 4
TYPE_BINARY = 5
TYPE_BOOL = 8
TYPE_INT32 = 16
TYPE_INT64 = 18
TYPE_OBJECT_ID = 7
TYPE_DATETIME = 9
TYPE_NULL = 10


def read_cstring(buffer, offset):
    """Reads a null-terminated string from the buffer."""
    end = offset
    while end < len(buffer) and buffer[end] != 0:
        end += 1
    return buffer[offset:end].decode("utf-8", errors="ignore"), end + 1


def skip_value(buffer, offset, dtype):
    """Calculates the new offset after skipping a value of a given BSON type."""
    if dtype == TYPE_DOUBLE:
        return offset + 8
    elif dtype == TYPE_STRING:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + l
    elif dtype == TYPE_DOC:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == TYPE_ARRAY:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == TYPE_BINARY:
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + 1 + l  # length + subtype + data
    elif dtype == TYPE_BOOL:
        return offset + 1
    elif dtype == TYPE_INT32:
        return offset + 4
    elif dtype == TYPE_INT64:
        return offset + 8
    elif dtype == TYPE_OBJECT_ID:
        return offset + 12
    elif dtype == TYPE_DATETIME:
        return offset + 8
    elif dtype == TYPE_NULL:
        return offset
    else:
        # Fallback for unknown types to prevent crash, though risky
        return offset


def extract_images_from_record(data):
    """
    Parses a single raw BSON record and extracts image binary data.
    Returns a list of bytes objects (JPEG data).
    """
    images = []
    offset = 4  # Skip total size header
    length = len(data)

    while offset < length - 1:
        dtype = data[offset]
        offset += 1
        key, offset = read_cstring(data, offset)

        # We are looking for the 'imgs' array
        if key == "imgs" and dtype == TYPE_ARRAY:
            arr_size = struct.unpack_from("<i", data, offset)[0]
            arr_end = offset + arr_size
            offset += 4

            # Iterate through array elements
            while offset < arr_end - 1:
                e_type = data[offset]
                offset += 1
                e_key, offset = read_cstring(data, offset)

                # Each element should be a document containing 'picture'
                if e_type == TYPE_DOC:
                    doc_size = struct.unpack_from("<i", data, offset)[0]
                    doc_end = offset + doc_size

                    # Dive into the document to find 'picture'
                    sub_offset = offset + 4
                    while sub_offset < doc_end - 1:
                        s_type = data[sub_offset]
                        sub_offset += 1
                        s_key, sub_offset = read_cstring(data, sub_offset)

                        if s_key == "picture" and s_type == TYPE_BINARY:
                            b_len = struct.unpack_from("<i", data, sub_offset)[0]
                            sub_offset += 4
                            subtype = data[sub_offset]
                            sub_offset += 1
                            # Extract image data
                            img_data = data[sub_offset : sub_offset + b_len]
                            images.append(img_data)
                            sub_offset += b_len
                        else:
                            sub_offset = skip_value(data, sub_offset, s_type)

                    offset = doc_end
                else:
                    offset = skip_value(data, offset, e_type)
        else:
            offset = skip_value(data, offset, dtype)

    return images


# ==== Analysis Functions ====


def analyze_target_variable(df_meta, df_cats):
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Merge with category names to get hierarchy
    df_merged = df_meta.merge(df_cats, on="category_id", how="left")

    # 1. Distribution of Level 1 Categories (High level)
    level1_counts = df_merged["category_level1"].value_counts()
    print(f"Number of Unique Classes (category_id): {df_meta['category_id'].nunique()}")
    print(f"Number of Level 1 Categories: {df_merged['category_level1'].nunique()}")

    print("\nTop 5 Level 1 Categories by Frequency:")
    for cat, count in level1_counts.head(5).items():
        print(f"  - {cat}: {count} ({count/len(df_meta)*100:.2f}%)")

    # 2. Class Balance (Lowest Level)
    class_counts = df_meta["category_id"].value_counts()
    min_cls = class_counts.min()
    max_cls = class_counts.max()
    mean_cls = class_counts.mean()

    print(f"\nClass Balance (Lowest Level):")
    print(f"  - Min samples per class: {min_cls}")
    print(f"  - Max samples per class: {max_cls}")
    print(f"  - Mean samples per class: {mean_cls:.2f}")

    # Imbalance Ratio
    imbalance_ratio = max_cls / min_cls
    print(f"  - Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")

    # Rare classes
    rare_threshold = len(df_meta) * 0.0001  # 0.01%
    rare_classes = class_counts[class_counts < rare_threshold]
    print(f"  - Number of rare classes (< 0.01% freq): {len(rare_classes)}")


def analyze_image_data(df_meta):
    print("\n==== INPUT DATA ANALYSIS (IMAGE) ====")

    # Sample data
    sample_df = df_meta.sample(
        n=min(GENERAL_SAMPLE_SIZE, len(df_meta)), random_state=SEED
    )
    pixel_sample_indices = set(
        sample_df.sample(
            n=min(PIXEL_SAMPLE_SIZE, len(sample_df)), random_state=SEED
        ).index
    )

    widths = []
    heights = []
    aspect_ratios = []
    channels_list = []
    img_counts_per_product = []

    # Pixel stats accumulators
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    print(f"Analyzing {len(sample_df)} products (sampling images)...")

    with open(TRAIN_BSON, "rb") as f:
        for idx, row in sample_df.iterrows():
            offset = row["bson_offset"]
            length = row["bson_length"]

            f.seek(offset)
            data = f.read(length)

            try:
                images_data = extract_images_from_record(data)
                img_counts_per_product.append(len(images_data))

                for i, img_bytes in enumerate(images_data):
                    # Decode image
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                    if img is None:
                        continue

                    # Dimensions
                    h, w = img.shape[:2]
                    c = 1 if len(img.shape) == 2 else img.shape[2]

                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h if h > 0 else 0)
                    channels_list.append(c)

                    # Pixel Stats (only for subset)
                    if (
                        idx in pixel_sample_indices and i == 0
                    ):  # Take first image of the product for pixel stats
                        # Convert to RGB if grayscale
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        elif img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Normalize to 0-1 for calculation stability
                        img_norm = img / 255.0
                        pixel_sum += img_norm.sum(axis=(0, 1))
                        pixel_sq_sum += (img_norm**2).sum(axis=(0, 1))
                        pixel_count += img.shape[0] * img.shape[1]

            except Exception as e:
                # Skip corrupt records in analysis
                continue

    # 1. Dimensions
    print("\nImage Dimensions:")
    print(
        f"  - Width: Mean={np.mean(widths):.2f}, Std={np.std(widths):.2f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  - Height: Mean={np.mean(heights):.2f}, Std={np.std(heights):.2f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )

    # 2. Aspect Ratios
    ar_mean = np.mean(aspect_ratios)
    ar_std = np.std(aspect_ratios)
    print(f"  - Aspect Ratio: Mean={ar_mean:.4f}, Std={ar_std:.4f}")

    # 3. Channels
    c_counts = Counter(channels_list)
    print(f"  - Channel Distribution: {dict(c_counts)}")

    # 4. Pixel Stats
    if pixel_count > 0:
        rgb_mean = pixel_sum / pixel_count
        rgb_std = np.sqrt((pixel_sq_sum / pixel_count) - (rgb_mean**2))
        print("\nPixel Statistics (Normalized 0-1):")
        print(
            f"  - Mean (R, G, B): [{rgb_mean[0]:.4f}, {rgb_mean[1]:.4f}, {rgb_mean[2]:.4f}]"
        )
        print(
            f"  - Std  (R, G, B): [{rgb_std[0]:.4f}, {rgb_std[1]:.4f}, {rgb_std[2]:.4f}]"
        )

    return img_counts_per_product


def analyze_meta_relationships(df_meta, img_counts):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Relationship 1: Number of images per product vs Category
    # Since we only have counts for a sample, we can't do a full correlation,
    # but we can report the distribution of image counts.

    print("Meta-Feature Analysis:")

    # Image Count Distribution
    count_dist = Counter(img_counts)
    print("Distribution of Images per Product:")
    for k in sorted(count_dist.keys()):
        print(f"  - {k} images: {count_dist[k]} products")

    # Relationship 2: BSON Record Size vs Category
    # Does the amount of data (bytes) correlate with specific categories?
    # We can check if 'bson_length' varies significantly by Level 1 category.
    # We need to merge with categories again for the sample.

    # We perform a quick ANOVA-like check or just report mean size per top category
    # to see if some categories have "heavier" data (more/larger images).

    # Load categories
    df_cats = pd.read_csv(CAT_NAMES)
    df_merged = df_meta.merge(df_cats, on="category_id", how="left")

    # Get top 3 categories
    top_cats = df_merged["category_level1"].value_counts().head(3).index.tolist()

    print("\nAverage Record Size (Bytes) for Top 3 Level 1 Categories:")
    for cat in top_cats:
        subset = df_merged[df_merged["category_level1"] == cat]
        mean_size = subset["bson_length"].mean()
        print(f"  - {cat}: {mean_size:.2f} bytes")


def main():
    # Load Metadata
    if not os.path.exists(TRAIN_META):
        print("Metadata not found. Please ensure metadata generation script has run.")
        return

    df_train = pd.read_csv(TRAIN_META)
    df_cats = pd.read_csv(CAT_NAMES)

    # 1. Target Analysis
    analyze_target_variable(df_train, df_cats)

    # 2. Image Analysis
    img_counts = analyze_image_data(df_train)

    # 3. Relationships
    # We pass the full df_train for meta-analysis, but we use the img_counts from the sample
    analyze_meta_relationships(
        df_train.sample(n=len(img_counts), random_state=SEED), img_counts
    )


if __name__ == "__main__":
    main()
