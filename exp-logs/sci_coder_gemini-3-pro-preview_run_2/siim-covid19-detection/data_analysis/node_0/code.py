import os
import pandas as pd
import numpy as np
import ast
import random
import importlib.util
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuration
METADATA_PATH = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE_IMG_ANALYSIS = (
    300  # Number of images to sample for pixel stats to save time
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def read_dicom_safe(path):
    """
    Attempts to read a DICOM file using pydicom or cv2.
    Returns (pixel_array, height, width, channels) or None tuple.
    """
    # Try pydicom
    if importlib.util.find_spec("pydicom") is not None:
        try:
            import pydicom

            dcm = pydicom.dcmread(path)
            # Handle pixel data
            arr = dcm.pixel_array
            h, w = arr.shape[:2]
            c = 1 if len(arr.shape) == 2 else arr.shape[2]
            return arr, h, w, c
        except Exception:
            pass

    # Try OpenCV
    if importlib.util.find_spec("cv2") is not None:
        try:
            import cv2

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                c = 1 if len(img.shape) == 2 else img.shape[2]
                return img, h, w, c
        except Exception:
            pass

    return None, None, None, None


def analyze_targets(df):
    print("DATA INTEGRITY")
    print(f"Analysis performed on Training Set only.")
    print(f"Total Samples: {len(df)}")

    print("\nTARGET VARIABLE ANALYSIS")
    target_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # Check for multi-label vs multi-class
    # Create a single label column for convenience
    def get_label(row):
        for col in target_cols:
            if row[col] == 1:
                return col
        return "None"

    df["target_class"] = df.apply(get_label, axis=1)

    # Distribution
    counts = df[target_cols].sum()
    total = len(df)

    print("Class Distribution (Study Level):")
    for col in target_cols:
        count = counts[col]
        ratio = count / total
        print(f"  {col}: {count} ({ratio:.4f})")

    # Check for rows with multiple labels or no labels
    row_sums = df[target_cols].sum(axis=1)
    multi_label_count = (row_sums > 1).sum()
    no_label_count = (row_sums == 0).sum()

    if multi_label_count > 0:
        print(f"  Note: {multi_label_count} studies have multiple active labels.")
    if no_label_count > 0:
        print(f"  Note: {no_label_count} studies have no active labels.")

    return df, target_cols


def analyze_images(df):
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Sample for efficiency
    sample_df = df.sample(n=min(len(df), SAMPLE_SIZE_IMG_ANALYSIS), random_state=SEED)

    widths = []
    heights = []
    aspect_ratios = []
    pixel_means = []
    pixel_stds = []
    channels = []

    read_success_count = 0

    for _, row in sample_df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            continue

        arr, h, w, c = read_dicom_safe(full_path)

        if arr is not None:
            read_success_count += 1
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            channels.append(c)

            # Pixel stats (normalize to float for calculation)
            arr_flat = arr.astype(float)
            pixel_means.append(np.mean(arr_flat))
            pixel_stds.append(np.std(arr_flat))

    if read_success_count == 0:
        print(
            "  Warning: Could not read DICOM images with available libraries (pydicom/cv2). Skipping pixel analysis."
        )
        return pd.DataFrame()  # Return empty for feature eng if image read fails

    # Dimensions
    print("Dimensions:")
    print(
        f"  Width:  Mean={np.mean(widths):.4f}, Std={np.std(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
    )
    print(
        f"  Height: Mean={np.mean(heights):.4f}, Std={np.std(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
    )
    print(
        f"  Aspect Ratio: Mean={np.mean(aspect_ratios):.4f}, Std={np.std(aspect_ratios):.4f}"
    )

    # Channels
    unique_channels, channel_counts = np.unique(channels, return_counts=True)
    print("Channels:")
    for c, count in zip(unique_channels, channel_counts):
        print(f"  {c} Channel(s): {count} images ({count/read_success_count:.4f})")

    # Pixel Stats
    print("Pixel Stats (Global approx. based on sample):")
    print(f"  Mean Pixel Value: {np.mean(pixel_means):.4f}")
    print(f"  Std Pixel Value:  {np.mean(pixel_stds):.4f}")

    # Return stats dataframe for relationship analysis
    # We need to align this with the original df. Since we sampled, we create a small DF.
    stats_df = pd.DataFrame(
        {
            "id": sample_df["id_x"],  # Use original ID
            "img_width": widths,
            "img_height": heights,
            "img_aspect_ratio": aspect_ratios,
            "img_mean": pixel_means,
        }
    )
    return stats_df


def analyze_features_and_relationships(df, img_stats_df):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # 1. Parse Bounding Boxes to create Meta-Features
    # 'boxes' column format: [{'x':..., 'y':..., 'width':..., 'height':...}] or NaN

    def get_box_stats(box_str):
        if pd.isna(box_str):
            return 0, 0.0
        try:
            boxes = ast.literal_eval(box_str)
            if not isinstance(boxes, list):
                return 0, 0.0

            count = len(boxes)
            if count == 0:
                return 0, 0.0

            areas = [b["width"] * b["height"] for b in boxes]
            avg_area = np.mean(areas)
            return count, avg_area
        except:
            return 0, 0.0

    # Apply extraction
    box_stats = df["boxes"].apply(lambda x: get_box_stats(x))
    df["num_boxes"] = [x[0] for x in box_stats]
    df["avg_box_area"] = [x[1] for x in box_stats]

    # 2. Merge Image Stats if available
    # Since img_stats_df is a sample, we left join.
    # Note: 'id_x' in df corresponds to 'id' in img_stats_df based on previous logic
    if not img_stats_df.empty:
        # We need to match the index or ID.
        # In analyze_images, we used sample_df which is a subset of df.
        # Let's just analyze relationships on the subset where we have image stats.
        analysis_df = df.merge(img_stats_df, left_on="id_x", right_on="id", how="inner")
    else:
        analysis_df = df.copy()
        # Create dummy cols if image read failed
        analysis_df["img_width"] = np.nan
        analysis_df["img_height"] = np.nan

    # 3. Structured Relationships (Correlations)
    print("Structured Relationships (Correlation with Target Class Index):")

    # Encode target for correlation
    le = LabelEncoder()
    analysis_df["target_encoded"] = le.fit_transform(analysis_df["target_class"])

    # Numerical features
    num_cols = ["num_boxes", "avg_box_area"]
    if not img_stats_df.empty:
        num_cols += ["img_width", "img_height", "img_mean"]

    correlations = (
        analysis_df[num_cols + ["target_encoded"]]
        .corr(method="spearman")["target_encoded"]
        .drop("target_encoded")
    )

    for feat, corr in correlations.items():
        print(f"  {feat} vs Target: {corr:.4f}")

    # Redundancy Check
    print("Feature Redundancy (Collinear pairs > 0.90):")
    feature_corr = analysis_df[num_cols].corr().abs()
    found_redundancy = False
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if feature_corr.iloc[i, j] > 0.90:
                print(f"  {num_cols[i]} - {num_cols[j]}: {feature_corr.iloc[i, j]:.4f}")
                found_redundancy = True
    if not found_redundancy:
        print("  None found.")

    # 4. Feature Importance (Random Forest)
    print("Feature Importance (Random Forest Classifier):")
    # Drop NaNs for RF
    rf_df = analysis_df[num_cols + ["target_encoded"]].dropna()

    if len(rf_df) > 0:
        X = rf_df[num_cols]
        y = rf_df["target_encoded"]

        clf = RandomForestClassifier(n_estimators=50, random_state=SEED, max_depth=5)
        clf.fit(X, y)

        importances = pd.Series(clf.feature_importances_, index=num_cols).sort_values(
            ascending=False
        )
        for feat, imp in importances.head(5).items():
            print(f"  {feat}: {imp:.4f}")
    else:
        print("  Insufficient data for Random Forest training.")

    # 5. Unstructured/Meta-Feature Relationships
    print("Meta-Feature Analysis:")
    # Avg num boxes per class
    print("  Average Number of Opacity Boxes per Class:")
    avg_boxes_per_class = analysis_df.groupby("target_class")["num_boxes"].mean()
    for cls, val in avg_boxes_per_class.items():
        print(f"    {cls}: {val:.4f}")


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    df, target_cols = analyze_targets(df)

    # 2. Image Analysis
    img_stats_df = analyze_images(df)

    # 3. Relationships
    analyze_features_and_relationships(df, img_stats_df)


if __name__ == "__main__":
    main()
