import pandas as pd
import numpy as np
import os
import cv2
import random
from scipy import stats


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def main():
    set_seed(42)

    # --- Constants ---
    META_PATH = "./metadata/train_meta.csv"
    INPUT_ROOT = "./input/train"

    # --- 1. Data Integrity ---
    print("=== DATA INTEGRITY ===")
    if not os.path.exists(META_PATH):
        print(f"Error: Metadata file not found at {META_PATH}")
        return

    df = pd.read_csv(META_PATH)
    print(f"Training Metadata Loaded: {len(df)} rows")
    print(f"Unique Images: {df['image_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")

    # --- 2. Target Variable Analysis ---
    print("\n=== TARGET VARIABLE ANALYSIS ===")

    # Class Distribution
    class_counts = df["class_name"].value_counts()
    total_obs = len(df)

    print(f"Total Observations: {total_obs}")
    print("Top 5 Classes by Frequency:")
    for name, count in class_counts.head(5).items():
        print(f"  - {name}: {count} ({count/total_obs*100:.4f}%)")

    # Imbalance Analysis (Excluding 'No finding' to check pathology imbalance)
    pathology_df = df[df["class_id"] != 14]
    if len(pathology_df) > 0:
        path_counts = pathology_df["class_name"].value_counts()
        max_cls = path_counts.max()
        min_cls = path_counts.min()
        imbalance_ratio = max_cls / min_cls
        print(f"Pathology Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")
        print(f"  - Most Frequent: {path_counts.idxmax()} ({max_cls})")
        print(f"  - Least Frequent: {path_counts.idxmin()} ({min_cls})")

    # No Finding vs Finding (Image Level)
    # Group by image to see how many images have NO findings
    img_groups = df.groupby("image_id")["class_id"].apply(lambda x: 14 in x.values)
    no_finding_imgs = img_groups.sum()
    total_imgs = len(img_groups)
    print(
        f"Image-Level 'No Finding' Prevalence: {no_finding_imgs/total_imgs*100:.4f}% ({no_finding_imgs}/{total_imgs})"
    )

    # --- 3. Input Data Analysis (Image Modality) ---
    print("\n=== INPUT DATA ANALYSIS (IMAGE) ===")

    # File Size Analysis
    # Sample 1000 files for speed
    unique_paths = df["file_path"].unique()
    sample_size = min(1000, len(unique_paths))
    sample_paths = np.random.choice(unique_paths, sample_size, replace=False)

    file_sizes = []
    for p in sample_paths:
        if os.path.exists(p):
            # Size in MB
            file_sizes.append(os.path.getsize(p) / (1024 * 1024))

    if file_sizes:
        print(f"File Size Stats (MB) [N={len(file_sizes)}]:")
        print(f"  - Mean: {np.mean(file_sizes):.4f}")
        print(f"  - Std:  {np.std(file_sizes):.4f}")
        print(f"  - Min:  {np.min(file_sizes):.4f}")
        print(f"  - Max:  {np.max(file_sizes):.4f}")

    # Image Content Analysis (Attempt CV2)
    print("Attempting to read image dimensions and pixel stats...")
    # Try reading a few images to check if CV2 supports this DICOM format
    # Note: Standard CV2 often fails on DICOM. We handle this gracefully.

    valid_images = 0
    widths = []
    heights = []
    pixel_means = []
    pixel_stds = []

    # Try reading up to 20 images
    test_read_paths = sample_paths[:20]

    for p in test_read_paths:
        try:
            # Attempt read
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                widths.append(w)
                heights.append(h)
                pixel_means.append(np.mean(img))
                pixel_stds.append(np.std(img))
                valid_images += 1
        except Exception:
            continue

    if valid_images > 0:
        print(f"Successfully read {valid_images} images with OpenCV.")
        print(f"  - Mean Width:  {np.mean(widths):.4f}")
        print(f"  - Mean Height: {np.mean(heights):.4f}")
        print(
            f"  - Mean Aspect Ratio: {np.mean(np.array(widths)/np.array(heights)):.4f}"
        )
        print(f"  - Global Pixel Mean: {np.mean(pixel_means):.4f}")
        print(f"  - Global Pixel Std:  {np.mean(pixel_stds):.4f}")
    else:
        print("Notice: OpenCV could not decode the DICOM files directly.")
        print(
            "        Pixel-level analysis (dimensions, channels, mean/std) is skipped."
        )
        print(
            "        Recommendation: Use 'pydicom' or convert DICOMs to PNG/JPG for training."
        )

    # --- 4. Feature/Signal Relationships ---
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # 4.1 Bounding Box Analysis (Structured)
    # Filter out "No finding" (Class 14 has bbox 0,0,1,1 usually, or is irrelevant for size analysis)
    bbox_df = df[df["class_id"] != 14].copy()

    if len(bbox_df) > 0:
        bbox_df["width"] = bbox_df["x_max"] - bbox_df["x_min"]
        bbox_df["height"] = bbox_df["y_max"] - bbox_df["y_min"]
        bbox_df["area"] = bbox_df["width"] * bbox_df["height"]

        print("Bounding Box Area Statistics (pixels^2):")
        print(f"  - Mean Area: {bbox_df['area'].mean():.4f}")
        print(f"  - Std Area:  {bbox_df['area'].std():.4f}")

        # Relationship: Class vs BBox Area
        print("Top 3 Classes by Mean Bounding Box Area:")
        class_area = (
            bbox_df.groupby("class_name")["area"].mean().sort_values(ascending=False)
        )
        for name, area in class_area.head(3).items():
            print(f"  - {name}: {area:.4f}")

        print("Bottom 3 Classes by Mean Bounding Box Area:")
        for name, area in class_area.tail(3).items():
            print(f"  - {name}: {area:.4f}")

    # 4.2 Radiologist Annotation Analysis
    # Check if some radiologists annotate much more than others
    rad_counts = df["rad_id"].value_counts()
    print(f"\nRadiologist Annotation Stats (Total Radiologists: {len(rad_counts)}):")
    print(f"  - Mean Annotations per Rad: {rad_counts.mean():.4f}")
    print(f"  - Std Annotations per Rad:  {rad_counts.std():.4f}")
    print(f"  - Max Annotations by single Rad: {rad_counts.max()}")

    # 4.3 Meta-Feature Relationship: File Size vs Number of Findings
    # Do larger files (likely higher res or less compression) have more findings?
    # We aggregate by image
    img_stats = (
        df.groupby("image_id")
        .agg(
            {
                "class_id": "count",  # This counts rows, i.e., number of findings
                "file_path": "first",
            }
        )
        .rename(columns={"class_id": "num_findings"})
    )

    # Get file sizes for these images (subset)
    # We reuse the file_sizes logic but mapped to these specific images
    # Since getting size for all 10k images is fast enough (metadata only), we do it.

    sizes = []
    findings = []

    # Sample 1000 images for correlation check to keep runtime low
    sample_imgs = img_stats.sample(min(1000, len(img_stats)), random_state=42)

    for idx, row in sample_imgs.iterrows():
        p = row["file_path"]
        if os.path.exists(p):
            s = os.path.getsize(p)
            sizes.append(s)
            findings.append(row["num_findings"])

    if len(sizes) > 1:
        corr, _ = stats.pearsonr(sizes, findings)
        print(f"\nCorrelation between File Size and Number of Findings: {corr:.4f}")
        if abs(corr) < 0.1:
            print("  -> No significant linear relationship detected.")
        else:
            print("  -> Potential relationship detected.")


if __name__ == "__main__":
    main()
