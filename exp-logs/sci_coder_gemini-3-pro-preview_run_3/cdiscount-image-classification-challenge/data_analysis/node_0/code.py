import os
import struct
import pandas as pd
import numpy as np
import cv2
import random

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
CATEGORY_NAMES_PATH = os.path.join(INPUT_DIR, "category_names.csv")
SEED = 42

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


# ==========================================
# HELPER FUNCTIONS (BSON PARSING)
# ==========================================
def get_val_size(type_byte, data, ptr):
    """Returns the size of a BSON value based on its type byte."""
    if type_byte == 0x01:  # double
        return 8
    elif type_byte == 0x02:  # string
        s_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + s_len
    elif type_byte == 0x03:  # document
        d_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return d_len
    elif type_byte == 0x04:  # array
        a_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return a_len
    elif type_byte == 0x05:  # binary
        b_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + 1 + b_len
    elif type_byte == 0x07:  # objectid
        return 12
    elif type_byte == 0x08:  # boolean
        return 1
    elif type_byte == 0x09:  # utc datetime
        return 8
    elif type_byte == 0x0A:  # null
        return 0
    elif type_byte == 0x10:  # int32
        return 4
    elif type_byte == 0x12:  # int64
        return 8
    else:
        return 0


def extract_images_from_bson(data):
    """
    Parses a raw BSON document to find the 'imgs' array and extract 'picture' binaries.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        name = data[ptr:name_end].decode("utf-8", errors="ignore")
        ptr = name_end + 1

        if name == "imgs" and type_byte == 0x04:
            # Found 'imgs' array
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len

            # Enter Array (skip length int)
            ap = ptr + 4
            while ap < arr_end - 1:
                etype = data[ap]
                ap += 1

                # Array keys are "0", "1"... skip them
                ename_end = data.find(b"\x00", ap)
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len

                    # Enter Document
                    dp = ap + 4
                    while dp < doc_end - 1:
                        dtype = data[dp]
                        dp += 1

                        dname_end = data.find(b"\x00", dp)
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            # Found picture binary
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            # Skip other fields in image doc
                            v_len = get_val_size(dtype, data, dp)
                            dp += v_len

                    ap += doc_len
                else:
                    v_len = get_val_size(etype, data, ap)
                    ap += v_len

            ptr += arr_len
        else:
            # Skip this field
            v_len = get_val_size(type_byte, data, ptr)
            ptr += v_len

    return images


# ==========================================
# MAIN EDA SCRIPT
# ==========================================
def main():
    # 1. DATA INTEGRITY
    print("DATA INTEGRITY")
    if not os.path.exists(TRAIN_META_PATH):
        print("Error: Metadata file not found.")
        return

    df_meta = pd.read_csv(TRAIN_META_PATH)
    print(f"Loaded training metadata with {len(df_meta)} records.")
    print("Strictly using training set for analysis to prevent leakage.")

    # Load Category Names
    df_cats = pd.read_csv(CATEGORY_NAMES_PATH)

    # Merge for analysis
    df_full = df_meta.merge(df_cats, on="category_id", how="left")

    # 2. TARGET VARIABLE ANALYSIS
    print("\nTARGET VARIABLE ANALYSIS")

    # Distribution
    num_classes = df_full["category_id"].nunique()
    class_counts = df_full["category_id"].value_counts()

    print(f"Total Unique Categories: {num_classes}")
    print(f"Samples per Class - Mean: {class_counts.mean():.4f}")
    print(f"Samples per Class - Std:  {class_counts.std():.4f}")
    print(f"Samples per Class - Min:  {class_counts.min()}")
    print(f"Samples per Class - Max:  {class_counts.max()}")

    # Imbalance
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Hierarchy Analysis
    print("Top 5 Level 1 Categories (Domain Overview):")
    l1_counts = df_full["category_level1"].value_counts(normalize=True)
    for name, freq in l1_counts.head(5).items():
        print(f"  - {name}: {freq:.4f}")

    # 3. INPUT DATA ANALYSIS (IMAGE MODALITY)
    print("\nINPUT DATA ANALYSIS (IMAGE MODALITY)")

    # Sampling Strategy
    SAMPLE_SIZE = 2000
    print(f"Sampling {SAMPLE_SIZE} records for deep image analysis...")
    sample_df = df_meta.sample(n=min(SAMPLE_SIZE, len(df_meta)), random_state=SEED)

    widths = []
    heights = []
    aspect_ratios = []
    channels = []
    pixel_means = []
    pixel_stds = []
    num_imgs_per_product = []
    file_sizes = []

    # Processing Images
    with open(TRAIN_BSON_PATH, "rb") as f:
        for _, row in sample_df.iterrows():
            offset = row["bson_offset"]
            length = row["bson_length"]

            # Seek and Read
            f.seek(offset)
            doc_data = f.read(length)

            # Extract
            img_binaries = extract_images_from_bson(doc_data)
            num_imgs_per_product.append(len(img_binaries))
            file_sizes.append(length)

            for img_bytes in img_binaries:
                # Decode
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
                channels.append(c)

                # Pixel Stats (Global estimation via per-image stats)
                pixel_means.append(img.mean())
                pixel_stds.append(img.std())

    # Reporting Image Stats
    print(f"Analyzed {len(widths)} individual images.")

    print("Dimensions:")
    print(
        f"  - Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  - Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  - Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    print("Channels:")
    unique_channels, c_counts = np.unique(channels, return_counts=True)
    for c, count in zip(unique_channels, c_counts):
        print(f"  - {c} Channels: {count} images ({count/len(channels):.2%})")

    print("Pixel Statistics (0-255):")
    print(f"  - Global Pixel Mean: {np.mean(pixel_means):.4f}")
    print(f"  - Global Pixel Std:  {np.mean(pixel_stds):.4f}")

    # 4. FEATURE/SIGNAL RELATIONSHIPS
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Add meta features to sample dataframe
    sample_df["num_imgs"] = num_imgs_per_product
    sample_df["file_size"] = file_sizes

    # Merge with categories for relationship analysis
    sample_full = sample_df.merge(df_cats, on="category_id", how="left")

    # Relationship: File Size vs Number of Images
    corr_size_imgs = sample_full["file_size"].corr(sample_full["num_imgs"])
    print("Structured Relationships:")
    print(f"  - Correlation (File Size vs Num Images): {corr_size_imgs:.4f}")

    # Relationship: Category vs Number of Images
    print("Unstructured (Meta-Feature) Relationships:")
    print("  - Average Images per Product by Top 5 Categories (Level 1):")
    top_cats = sample_full["category_level1"].value_counts().head(5).index
    for cat in top_cats:
        avg_imgs = sample_full[sample_full["category_level1"] == cat]["num_imgs"].mean()
        print(f"    - {cat}: {avg_imgs:.4f}")


if __name__ == "__main__":
    main()
