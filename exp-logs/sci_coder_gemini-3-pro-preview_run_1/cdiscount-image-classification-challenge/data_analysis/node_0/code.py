import os
import sys
import struct
import io
import time
import random
import pandas as pd
import numpy as np
from PIL import Image

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")

# BSON Constants
BSON_TYPE_DOUBLE = 1
BSON_TYPE_STRING = 2
BSON_TYPE_OBJECT = 3
BSON_TYPE_ARRAY = 4
BSON_TYPE_BINARY = 5
BSON_TYPE_OBJECTID = 7
BSON_TYPE_BOOL = 8
BSON_TYPE_DATE = 9
BSON_TYPE_NULL = 10
BSON_TYPE_INT32 = 16
BSON_TYPE_INT64 = 18


def read_c_string(buffer, start):
    """Reads a C-style null-terminated string from buffer."""
    end = buffer.find(b"\x00", start)
    if end == -1:
        return None, -1
    return buffer[start:end].decode("utf-8", errors="ignore"), end + 1


def extract_images_from_bson(buffer):
    """
    Parses a BSON document buffer to extract image binary data from the 'imgs' array.
    Returns a list of binary image data.
    """
    images = []
    idx = 0
    buf_len = len(buffer)

    # Skip size (4 bytes) if it's included in the buffer passed,
    # but usually we pass the payload. If the buffer starts with size, we skip it.
    # We'll assume the buffer passed is the full document content after the initial read.
    # However, the standard BSON doc starts with int32 size.
    # Let's assume we start parsing elements immediately after size.
    # If the buffer is the whole doc, idx starts at 4.

    # Heuristic: check if first 4 bytes match buffer length
    if buf_len >= 4:
        size = struct.unpack("<i", buffer[0:4])[0]
        if size == buf_len:
            idx = 4

    while idx < buf_len - 1:
        type_byte = buffer[idx]
        idx += 1

        name, idx = read_c_string(buffer, idx)
        if idx == -1:
            break

        # We are looking for 'imgs'
        if name == "imgs" and type_byte == BSON_TYPE_ARRAY:
            # Parse Array (which is a BSON object)
            if idx + 4 > buf_len:
                break
            arr_len = struct.unpack("<i", buffer[idx : idx + 4])[0]
            arr_end = idx + arr_len

            # Enter array
            a_idx = idx + 4
            while a_idx < arr_end - 1:
                e_type = buffer[a_idx]
                a_idx += 1
                e_name, a_idx = read_c_string(buffer, a_idx)  # index "0", "1", etc.

                if e_type == BSON_TYPE_OBJECT:
                    # Inside the array element (dict with 'picture')
                    o_len = struct.unpack("<i", buffer[a_idx : a_idx + 4])[0]
                    o_end = a_idx + o_len

                    o_curr = a_idx + 4
                    while o_curr < o_end - 1:
                        p_type = buffer[o_curr]
                        o_curr += 1
                        p_name, o_curr = read_c_string(buffer, o_curr)

                        if p_name == "picture" and p_type == BSON_TYPE_BINARY:
                            b_len = struct.unpack("<i", buffer[o_curr : o_curr + 4])[0]
                            subtype = buffer[o_curr + 4]  # usually 0
                            # Image data starts at o_curr + 5
                            img_data = buffer[o_curr + 5 : o_curr + 5 + b_len]
                            images.append(img_data)
                            o_curr += 5 + b_len
                        else:
                            # Skip value
                            o_curr = skip_bson_value(buffer, o_curr, p_type)

                    a_idx = o_end
                else:
                    a_idx = skip_bson_value(buffer, a_idx, e_type)

            # We found 'imgs', we can stop parsing the main doc
            return images
        else:
            # Skip value
            idx = skip_bson_value(buffer, idx, type_byte)

    return images


def skip_bson_value(buffer, idx, type_byte):
    """Helper to skip a BSON value based on its type."""
    if idx >= len(buffer):
        return len(buffer)

    if type_byte == BSON_TYPE_DOUBLE:
        return idx + 8
    elif type_byte == BSON_TYPE_STRING:
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + l
    elif type_byte == BSON_TYPE_OBJECT or type_byte == BSON_TYPE_ARRAY:
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + l
    elif type_byte == BSON_TYPE_BINARY:
        l = struct.unpack("<i", buffer[idx : idx + 4])[0]
        return idx + 4 + 1 + l
    elif type_byte == BSON_TYPE_OBJECTID:
        return idx + 12
    elif type_byte == BSON_TYPE_BOOL:
        return idx + 1
    elif type_byte == BSON_TYPE_DATE:
        return idx + 8
    elif type_byte == BSON_TYPE_NULL:
        return idx
    elif type_byte == BSON_TYPE_INT32:
        return idx + 4
    elif type_byte == BSON_TYPE_INT64:
        return idx + 8
    else:
        # Unknown, jump to end to be safe
        return len(buffer)


def main():
    # 1. Load Data
    try:
        df_meta = pd.read_csv(TRAIN_METADATA)
        df_cats = pd.read_csv(CATEGORY_NAMES)
    except FileNotFoundError:
        print("Error: Metadata or Category files not found.")
        return

    # Merge category names for analysis
    df_full = df_meta.merge(df_cats, on="category_id", how="left")

    # ==========================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    # Class Balance
    class_counts = df_full["category_id"].value_counts()
    n_classes = len(class_counts)
    total_samples = len(df_full)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {n_classes}")

    # Imbalance Metrics
    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()

    print(f"Class Distribution:")
    print(f"  Min samples per class: {min_count}")
    print(f"  Max samples per class: {max_count}")
    print(f"  Mean samples per class: {mean_count:.4f}")
    print(f"  Imbalance Ratio (Max/Min): {max_count/min_count:.4f}")

    # Top 5 Categories (Level 1)
    if "category_level1" in df_full.columns:
        top_l1 = df_full["category_level1"].value_counts(normalize=True).head(5)
        print("\nTop 5 Level-1 Categories (by frequency):")
        for cat, freq in top_l1.items():
            print(f"  {cat}: {freq:.4f}")

    # ==========================================
    # 3. INPUT DATA ANALYSIS (IMAGE)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Sampling for Image Analysis
    SAMPLE_SIZE = 5000
    if len(df_meta) > SAMPLE_SIZE:
        sample_indices = df_meta.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).index
        df_sample = df_meta.loc[sample_indices].copy()
    else:
        df_sample = df_meta.copy()

    widths = []
    heights = []
    aspect_ratios = []
    channel_counts = []
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0
    img_counts_per_product = []

    # Open BSON file
    with open(TRAIN_BSON, "rb") as f:
        for _, row in df_sample.iterrows():
            offset = row["bson_offset"]
            length = row["bson_length"]

            f.seek(offset)
            doc_bytes = f.read(length)

            # Extract images
            img_binaries = extract_images_from_bson(doc_bytes)
            img_counts_per_product.append(len(img_binaries))

            for b_data in img_binaries:
                try:
                    with Image.open(io.BytesIO(b_data)) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        aspect_ratios.append(w / h if h > 0 else 0)

                        # Convert to RGB for stats
                        img_rgb = img.convert("RGB")
                        channel_counts.append(len(img.getbands()))  # Original bands

                        # Pixel Stats (Accumulate)
                        arr = np.array(img_rgb) / 255.0
                        pixel_sum += arr.sum(axis=(0, 1))
                        pixel_sq_sum += (arr**2).sum(axis=(0, 1))
                        total_pixels += w * h

                except Exception:
                    # Skip corrupt images
                    pass

    # Dimensions
    w_series = pd.Series(widths)
    h_series = pd.Series(heights)
    ar_series = pd.Series(aspect_ratios)

    print("Image Dimensions:")
    print(
        f"  Width:  Mean={w_series.mean():.4f}, Std={w_series.std():.4f}, Min={w_series.min()}, Max={w_series.max()}"
    )
    print(
        f"  Height: Mean={h_series.mean():.4f}, Std={h_series.std():.4f}, Min={h_series.min()}, Max={h_series.max()}"
    )
    print(f"  Aspect Ratio: Mean={ar_series.mean():.4f}, Std={ar_series.std():.4f}")

    # Channels
    # Since we converted to RGB for stats, we check the original modes encountered
    # Assuming most are RGB, but let's report if we found grayscale
    # (Here we just tracked count of bands from original load)
    # Note: JPEG is usually 3 (RGB) or 1 (L).
    unique_channels, channel_counts_res = np.unique(channel_counts, return_counts=True)
    print("\nChannel Distribution:")
    for c, count in zip(unique_channels, channel_counts_res):
        print(f"  {c} Channels: {count} images ({count/len(channel_counts):.4f})")

    # Pixel Stats
    if total_pixels > 0:
        global_mean = pixel_sum / total_pixels
        # Std = sqrt( E[x^2] - (E[x])^2 )
        global_std = np.sqrt((pixel_sq_sum / total_pixels) - (global_mean**2))

        print("\nPixel Statistics (Normalized [0,1], RGB):")
        print(
            f"  Mean: R={global_mean[0]:.4f}, G={global_mean[1]:.4f}, B={global_mean[2]:.4f}"
        )
        print(
            f"  Std:  R={global_std[0]:.4f}, G={global_std[1]:.4f}, B={global_std[2]:.4f}"
        )

    # ==========================================
    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Add extracted meta-features to the sample dataframe
    df_sample["num_images"] = img_counts_per_product

    # Join with category info if not already present (df_sample is subset of df_meta)
    # We need category_level1 for the sample
    df_sample = df_sample.merge(df_cats, on="category_id", how="left")

    # Relationship: Number of Images vs Category (Level 1)
    print("Meta-Feature Relationship: Avg Images per Product by Category (Top 5)")
    if "category_level1" in df_sample.columns:
        # Group by level 1 and calc mean num_images
        cat_img_stats = (
            df_sample.groupby("category_level1")["num_images"]
            .mean()
            .sort_values(ascending=False)
        )

        print("  Highest Avg Images per Product:")
        for cat, val in cat_img_stats.head(5).items():
            print(f"    {cat}: {val:.4f}")

        print("  Lowest Avg Images per Product:")
        for cat, val in cat_img_stats.tail(5).items():
            print(f"    {cat}: {val:.4f}")

    # Relationship: Image File Size (proxy via bson_length) vs Category
    # bson_length includes metadata overhead but is dominated by image size
    print("\nMeta-Feature Relationship: Avg Record Size (Bytes) by Category (Top 5)")
    if "category_level1" in df_sample.columns:
        cat_size_stats = (
            df_sample.groupby("category_level1")["bson_length"]
            .mean()
            .sort_values(ascending=False)
        )

        print("  Largest Avg Record Size:")
        for cat, val in cat_size_stats.head(5).items():
            print(f"    {cat}: {val:.4f} bytes")


if __name__ == "__main__":
    main()
