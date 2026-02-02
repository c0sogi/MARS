import os
import cv2
import numpy as np
import pandas as pd
import random
from collections import Counter
from scipy.stats import skew, kurtosis

# ==========================================
# 1. Setup and Configuration
# ==========================================
SEED = 42
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def main():
    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH, keep_default_na=False)

    # Pre-calculate full paths
    # The metadata contains relative paths like 'train_images/ID.jpg'
    # We need to prepend the INPUT_DIR
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(INPUT_DIR, x))

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    # Parse labels
    # Format: "Unicode X Y W H Unicode X Y W H ..."
    all_chars = []
    all_bboxes = []  # List of [x, y, w, h]
    chars_per_image = []

    for labels in df["labels"]:
        if not labels:
            chars_per_image.append(0)
            continue

        parts = labels.split()
        # Each label is 5 parts: Code, X, Y, W, H
        if len(parts) % 5 != 0:
            # Should not happen given dataset specs, but safety check
            continue

        num_chars = len(parts) // 5
        chars_per_image.append(num_chars)

        for i in range(num_chars):
            code = parts[i * 5]
            try:
                x = int(parts[i * 5 + 1])
                y = int(parts[i * 5 + 2])
                w = int(parts[i * 5 + 3])
                h = int(parts[i * 5 + 4])

                all_chars.append(code)
                all_bboxes.append([x, y, w, h])
            except ValueError:
                continue

    # 2.1 Distribution
    unique_classes = set(all_chars)
    class_counts = Counter(all_chars)
    sorted_counts = class_counts.most_common()

    print(f"Total Annotations: {len(all_chars)}")
    print(f"Unique Classes (Unicode Chars): {len(unique_classes)}")

    if len(chars_per_image) > 0:
        print(
            f"Avg Characters per Image: {np.mean(chars_per_image):.4f} (Std: {np.std(chars_per_image):.4f})"
        )
        print(f"Max Characters in an Image: {np.max(chars_per_image)}")
        print(f"Min Characters in an Image: {np.min(chars_per_image)}")

    # 2.2 Imbalance
    if sorted_counts:
        top_5 = sorted_counts[:5]
        bottom_5 = sorted_counts[-5:]
        print(f"Most Common Class: {top_5[0][0]} (Count: {top_5[0][1]})")
        print(f"Least Common Class: {bottom_5[-1][0]} (Count: {bottom_5[-1][1]})")

        # Calculate imbalance ratio
        max_count = top_5[0][1]
        min_count = bottom_5[-1][1]
        print(f"Class Imbalance Ratio (Max/Min): {max_count/min_count:.4f}")

        # Share of top 10 classes
        top_10_count = sum(c for _, c in sorted_counts[:10])
        print(f"Top 10 Classes Share: {top_10_count / len(all_chars):.4f}")

    # 2.3 Bounding Box Stats (Regression Targets)
    if all_bboxes:
        bbox_arr = np.array(all_bboxes)
        # bbox_arr columns: x, y, w, h
        areas = bbox_arr[:, 2] * bbox_arr[:, 3]
        ratios = bbox_arr[:, 2] / (bbox_arr[:, 3] + 1e-6)  # w / h

        print(f"BBox Area Mean: {np.mean(areas):.4f} (Std: {np.std(areas):.4f})")
        print(f"BBox Aspect Ratio Mean: {np.mean(ratios):.4f}")
        print(f"BBox Width Mean: {np.mean(bbox_arr[:, 2]):.4f}")
        print(f"BBox Height Mean: {np.mean(bbox_arr[:, 3]):.4f}")

    print("-" * 30)

    # ==========================================
    # 3. Input Data Analysis (Image Data)
    # ==========================================
    print("INPUT DATA ANALYSIS")

    img_widths = []
    img_heights = []
    img_channels = []

    # We will sample for pixel stats to save time, but check dimensions for all (reading header is fast usually,
    # but cv2.imread loads full image. We'll optimize by just doing full load on all since dataset is ~2k images).
    # 2245 images is small enough to loop through for dimensions.

    # For pixel stats, we will use a running accumulator on a sample.
    pixel_sample_size = min(300, len(df))
    pixel_sample_indices = set(random.sample(range(len(df)), pixel_sample_size))

    sum_r, sum_g, sum_b = 0, 0, 0
    sum_sq_r, sum_sq_g, sum_sq_b = 0, 0, 0
    pixel_count = 0

    valid_images_count = 0

    for idx, row in df.iterrows():
        path = row["full_path"]
        if not os.path.exists(path):
            continue

        # Read image
        img = cv2.imread(path)
        if img is None:
            continue

        h, w, c = img.shape
        img_widths.append(w)
        img_heights.append(h)
        img_channels.append(c)
        valid_images_count += 1

        # Calculate pixel stats only on sample
        if idx in pixel_sample_indices:
            # Convert to RGB for consistency in reporting
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Flatten
            pixels = img_rgb.reshape(-1, 3) / 255.0

            n_pixels = pixels.shape[0]

            sum_r += pixels[:, 0].sum()
            sum_g += pixels[:, 1].sum()
            sum_b += pixels[:, 2].sum()

            sum_sq_r += (pixels[:, 0] ** 2).sum()
            sum_sq_g += (pixels[:, 1] ** 2).sum()
            sum_sq_b += (pixels[:, 2] ** 2).sum()

            pixel_count += n_pixels

    # 3.1 Dimensions
    if valid_images_count > 0:
        print(f"Image Count Analyzed: {valid_images_count}")
        print(
            f"Image Width Mean: {np.mean(img_widths):.4f} (Min: {np.min(img_widths)}, Max: {np.max(img_widths)})"
        )
        print(
            f"Image Height Mean: {np.mean(img_heights):.4f} (Min: {np.min(img_heights)}, Max: {np.max(img_heights)})"
        )

        aspect_ratios = np.array(img_widths) / np.array(img_heights)
        print(f"Image Aspect Ratio Mean: {np.mean(aspect_ratios):.4f}")

        # 3.2 Channels
        channel_counts = Counter(img_channels)
        print(f"Channel Distribution: {dict(channel_counts)}")

        # 3.3 Pixel Stats
        if pixel_count > 0:
            mean_r = sum_r / pixel_count
            mean_g = sum_g / pixel_count
            mean_b = sum_b / pixel_count

            std_r = np.sqrt((sum_sq_r / pixel_count) - (mean_r**2))
            std_g = np.sqrt((sum_sq_g / pixel_count) - (mean_g**2))
            std_b = np.sqrt((sum_sq_b / pixel_count) - (mean_b**2))

            print(
                f"Pixel Mean (RGB) [0-1]: R={mean_r:.4f}, G={mean_g:.4f}, B={mean_b:.4f}"
            )
            print(
                f"Pixel Std (RGB) [0-1]:  R={std_r:.4f}, G={std_g:.4f}, B={std_b:.4f}"
            )

    print("-" * 30)

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 4.1 Meta-Feature Relationships
    # Correlation between Image Area and Number of Characters
    if valid_images_count > 0 and len(chars_per_image) == len(img_widths):
        img_areas = np.array(img_widths) * np.array(img_heights)
        n_chars = np.array(chars_per_image)

        # Filter out potential mismatches if any (though lengths should match from iterrows)
        # We used iterrows for images, but chars_per_image was built from df['labels']
        # Assuming 1-to-1 correspondence and order preservation

        corr_area_chars = np.corrcoef(img_areas, n_chars)[0, 1]
        print(f"Correlation (Image Area vs Num Characters): {corr_area_chars:.4f}")

        # Correlation between Aspect Ratio and Num Characters
        # (e.g. are wider pages more text-heavy?)
        ar = np.array(img_widths) / np.array(img_heights)
        corr_ar_chars = np.corrcoef(ar, n_chars)[0, 1]
        print(f"Correlation (Aspect Ratio vs Num Characters): {corr_ar_chars:.4f}")

    # 4.2 Class-Specific Features (Structured)
    # Do specific classes have distinct sizes?
    # We'll take the top 5 classes and compare their average BBox area
    if sorted_counts:
        print("Average BBox Area for Top 5 Classes:")
        top_5_codes = [c[0] for c in sorted_counts[:5]]

        # Create a lookup for bbox areas by class
        class_areas = {code: [] for code in top_5_codes}

        # Re-iterate labels (efficient enough for metadata)
        for labels in df["labels"]:
            if not labels:
                continue
            parts = labels.split()
            num_chars = len(parts) // 5
            for i in range(num_chars):
                code = parts[i * 5]
                if code in class_areas:
                    try:
                        w = int(parts[i * 5 + 3])
                        h = int(parts[i * 5 + 4])
                        class_areas[code].append(w * h)
                    except:
                        pass

        for code in top_5_codes:
            areas = class_areas[code]
            if areas:
                mean_area = np.mean(areas)
                print(f"  {code}: {mean_area:.4f} px^2")


if __name__ == "__main__":
    main()
