import os
import pandas as pd
import numpy as np
import pydicom
import ast
from concurrent.futures import ProcessPoolExecutor
import warnings
import random
from scipy.stats import skew, kurtosis

# 1. Configuration & Setup
warnings.filterwarnings("ignore")

SEED = 42
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
N_PIXEL_SAMPLES = (
    1000  # Number of images to sample for pixel-level stats to ensure runtime safety
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)


def get_dicom_meta(args):
    """
    Reads a DICOM file and extracts metadata.
    If sample_pixels is True, also calculates pixel mean/std.
    """
    file_path, sample_pixels = args
    full_path = os.path.join(INPUT_DIR, file_path)

    try:
        # Read only headers first for speed
        dcm = pydicom.dcmread(full_path, stop_before_pixels=not sample_pixels)

        meta = {
            "height": dcm.Rows,
            "width": dcm.Columns,
            "channels": getattr(dcm, "SamplesPerPixel", 1),
            "photometric_interpretation": getattr(
                dcm, "PhotometricInterpretation", "UNKNOWN"
            ),
            "pixel_mean": np.nan,
            "pixel_std": np.nan,
        }

        if sample_pixels and hasattr(dcm, "pixel_array"):
            arr = dcm.pixel_array.astype(np.float32)
            meta["pixel_mean"] = np.mean(arr)
            meta["pixel_std"] = np.std(arr)

        return meta
    except Exception as e:
        return None


def main():
    print("DATA INTEGRITY")
    # Load the training metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Training set loaded. Shape: {df.shape}")
    print(f"Unique Studies: {df['study_id'].nunique()}")
    print(f"Unique Images: {df['image_id'].nunique()}")
    print("-" * 30)

    # 2. Target Variable Analysis
    print("\nTARGET VARIABLE ANALYSIS")

    # Classification Labels (Study Level)
    # We group by study_id to get study-level labels (since rows are images)
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    df_study = df.drop_duplicates(subset=["study_id"])

    print("Class Distribution (Study Level):")
    total_studies = len(df_study)
    for col in label_cols:
        count = df_study[col].sum()
        ratio = count / total_studies
        print(f"  {col}: {count} ({ratio:.4f})")

    # Check for multi-label studies
    label_sums = df_study[label_cols].sum(axis=1)
    multi_label_count = (label_sums > 1).sum()
    no_label_count = (label_sums == 0).sum()
    print(
        f"  Multi-label Studies: {multi_label_count} ({multi_label_count/total_studies:.4f})"
    )
    print(f"  Unlabeled Studies: {no_label_count} ({no_label_count/total_studies:.4f})")

    # Bounding Box Analysis (Image Level)
    print("\nBounding Box Analysis (Image Level):")

    def count_boxes(box_str):
        if pd.isna(box_str):
            return 0
        try:
            boxes = ast.literal_eval(box_str)
            return len(boxes)
        except:
            return 0

    df["num_boxes"] = df["boxes"].apply(count_boxes)

    images_with_boxes = (df["num_boxes"] > 0).sum()
    total_images = len(df)
    print(
        f"  Images with Opacities: {images_with_boxes} ({images_with_boxes/total_images:.4f})"
    )
    print(
        f"  Images without Opacities: {total_images - images_with_boxes} ({(total_images - images_with_boxes)/total_images:.4f})"
    )
    print(f"  Max boxes in single image: {df['num_boxes'].max()}")
    print(
        f"  Avg boxes per image (where > 0): {df[df['num_boxes'] > 0]['num_boxes'].mean():.4f}"
    )

    print("-" * 30)

    # 3. Input Data Analysis (Image)
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Prepare arguments for parallel processing
    # We will get dimensions for ALL images, but pixel stats for a SAMPLE

    # Identify indices to sample for pixel stats
    sample_indices = set(
        df.sample(n=min(N_PIXEL_SAMPLES, len(df)), random_state=SEED).index
    )

    tasks = []
    for idx, row in df.iterrows():
        sample_pixels = idx in sample_indices
        tasks.append((row["file_path"], sample_pixels))

    # Run parallel extraction
    # Using 10 workers to leave room for system overhead
    meta_results = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_dicom_meta, tasks))

    # Merge results back to dataframe
    # We need to handle potential None returns if file read failed
    valid_results = [r if r is not None else {} for r in results]

    img_meta_df = pd.DataFrame(valid_results)

    # Combine with main df for correlation analysis later
    # Reset index to ensure alignment if any rows were skipped (though we used map so order is preserved)
    df = pd.concat(
        [df.reset_index(drop=True), img_meta_df.reset_index(drop=True)], axis=1
    )

    # Dimensions
    print("Dimensions:")
    print(
        f"  Width:  Mean={df['width'].mean():.4f}, Std={df['width'].std():.4f}, Min={df['width'].min()}, Max={df['width'].max()}"
    )
    print(
        f"  Height: Mean={df['height'].mean():.4f}, Std={df['height'].std():.4f}, Min={df['height'].min()}, Max={df['height'].max()}"
    )

    df["aspect_ratio"] = df["width"] / df["height"]
    print(
        f"  Aspect Ratio: Mean={df['aspect_ratio'].mean():.4f}, Std={df['aspect_ratio'].std():.4f}"
    )

    # Channels
    print("\nChannels:")
    print(f"  Channel Counts: {df['channels'].value_counts().to_dict()}")
    print(
        f"  Photometric Interpretations: {df['photometric_interpretation'].value_counts().to_dict()}"
    )

    # Pixel Stats (Sampled)
    print(f"\nPixel Stats (Calculated on {len(sample_indices)} random samples):")
    sampled_df = df.dropna(subset=["pixel_mean"])

    if not sampled_df.empty:
        global_mean = sampled_df["pixel_mean"].mean()
        global_std_of_means = sampled_df["pixel_mean"].std()
        print(
            f"  Global Pixel Mean: {global_mean:.4f} (Std of means: {global_std_of_means:.4f})"
        )

        global_std = sampled_df["pixel_std"].mean()
        print(f"  Global Pixel Std:  {global_std:.4f}")
    else:
        print("  Could not calculate pixel stats.")

    print("-" * 30)

    # 4. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Unstructured (Meta-Feature) Relationships
    # Correlate Image Dimensions with Target Classes

    # Create a numeric target column for correlation (0/1 for each class)
    # We already have the one-hot columns.

    correlation_features = ["width", "height", "aspect_ratio"]
    # Add pixel stats if available for those rows (fill nan with mean for correlation check or drop)
    # We'll stick to geometry for the full dataset correlation

    print("Correlation between Image Geometry and Target Classes:")
    for label in label_cols:
        print(f"  Target: {label}")
        for feature in correlation_features:
            # Point-biserial correlation (since label is binary, feature is continuous)
            # Using Pearson correlation coefficient
            corr = df[feature].corr(df[label])
            print(f"    vs {feature}: {corr:.4f}")

    # Relationship between Image Size and Presence of Boxes
    df["has_boxes"] = (df["num_boxes"] > 0).astype(int)
    print("\nCorrelation between Image Geometry and Presence of Opacity (Boxes):")
    for feature in correlation_features:
        corr = df[feature].corr(df["has_boxes"])
        print(f"    vs {feature}: {corr:.4f}")

    # Compare Pixel Intensity for Negative vs Positive (using the sampled data)
    print("\nPixel Intensity vs Findings (Sampled Data):")
    if not sampled_df.empty:
        # We classify 'Negative for Pneumonia' as Negative, others as Positive findings
        neg_mean = sampled_df[sampled_df["Negative for Pneumonia"] == 1][
            "pixel_mean"
        ].mean()
        pos_mean = sampled_df[sampled_df["Negative for Pneumonia"] == 0][
            "pixel_mean"
        ].mean()

        print(f"  Avg Pixel Mean (Negative Cases): {neg_mean:.4f}")
        print(f"  Avg Pixel Mean (Positive Cases): {pos_mean:.4f}")

        diff = pos_mean - neg_mean
        print(f"  Difference (Pos - Neg): {diff:.4f}")

    print("-" * 30)


if __name__ == "__main__":
    main()
