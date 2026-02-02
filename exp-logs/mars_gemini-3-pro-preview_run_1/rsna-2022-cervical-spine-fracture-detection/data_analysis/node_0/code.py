import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings("ignore")

# Set Random Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Check for pydicom
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def run_eda():
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # Paths
    METADATA_PATH = "./metadata/train_metadata.csv"
    BBOX_PATH = "./input/train_bounding_boxes.csv"
    INPUT_DIR = "./input"

    # 1. DATA INTEGRITY
    # ---------------------------------------------------------
    # Load metadata
    if not os.path.exists(METADATA_PATH):
        print("Error: Metadata file not found.")
        return

    df_train = pd.read_csv(METADATA_PATH)

    print("\nDATA INTEGRITY")
    print(f"Analysis performed strictly on training set metadata: {METADATA_PATH}")
    print(f"Total Training Samples (Studies): {len(df_train)}")

    # 2. TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")

    # Target columns
    target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    # Distribution
    print("Class Balance Ratios:")
    for col in target_cols:
        pos_count = df_train[col].sum()
        total = len(df_train)
        ratio = pos_count / total
        print(f"  {col:<15}: {pos_count} positive ({ratio:.4f})")

    # Co-occurrence (Multi-label analysis)
    # How many vertebrae are fractured per patient?
    vert_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    df_train["fracture_count"] = df_train[vert_cols].sum(axis=1)

    print("\nFracture Multiplicity (Count of fractured vertebrae per patient):")
    counts = df_train["fracture_count"].value_counts().sort_index()
    for cnt, freq in counts.items():
        print(f"  {cnt} fractures: {freq} patients ({freq/len(df_train):.4f})")

    # 3. INPUT DATA ANALYSIS (IMAGE MODALITY)
    # ---------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (IMAGE MODALITY)")

    # We need to gather stats from the files.
    # Since reading all DICOMs is slow, we analyze file counts and sample headers.

    study_stats = []

    # Iterate over studies
    # To save time, we might limit deep inspection if dataset was huge,
    # but 161 studies is manageable.

    img_widths = []
    img_heights = []
    slice_counts = []
    file_sizes = []
    pixel_spacings = []

    print(f"Analyzing {len(df_train)} studies...")

    for idx, row in df_train.iterrows():
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        # List files
        files = os.listdir(full_path)
        # Filter for likely dicom files (no extension or .dcm)
        # The prompt says they are .dcm
        dcm_files = [f for f in files if f.endswith(".dcm")]

        n_slices = len(dcm_files)
        slice_counts.append(n_slices)

        # File sizes (average for the study)
        if n_slices > 0:
            sample_files = dcm_files[:5]  # check first 5 for speed
            sizes = [os.path.getsize(os.path.join(full_path, f)) for f in sample_files]
            avg_size = np.mean(sizes)
            file_sizes.append(avg_size)

            # Dimensions via pydicom if available
            if HAS_PYDICOM:
                try:
                    # Read middle slice
                    mid_file = dcm_files[n_slices // 2]
                    ds = pydicom.dcmread(
                        os.path.join(full_path, mid_file), stop_before_pixels=True
                    )
                    img_widths.append(ds.Columns)
                    img_heights.append(ds.Rows)
                    if "PixelSpacing" in ds:
                        pixel_spacings.append(
                            ds.PixelSpacing[0]
                        )  # Assuming square pixels x=y
                except Exception:
                    pass
        else:
            file_sizes.append(0)

    # Report Stats
    print("\nScan Dimensions (Slices per Scan):")
    if slice_counts:
        print(f"  Mean: {np.mean(slice_counts):.4f}")
        print(f"  Std : {np.std(slice_counts):.4f}")
        print(f"  Min : {np.min(slice_counts)}")
        print(f"  Max : {np.max(slice_counts)}")
    else:
        print("  No slice data found.")

    print("\nFile Sizes (Bytes):")
    if file_sizes:
        print(f"  Mean: {np.mean(file_sizes):.4f}")
        print(f"  Std : {np.std(file_sizes):.4f}")
    else:
        print("  No file size data found.")

    if HAS_PYDICOM and img_widths:
        print("\nImage Resolutions (Rows x Cols):")
        # Check if all are same
        unique_w = np.unique(img_widths)
        unique_h = np.unique(img_heights)
        print(f"  Unique Widths : {unique_w}")
        print(f"  Unique Heights: {unique_h}")

        print("\nPixel Spacing (mm):")
        print(f"  Mean: {np.mean(pixel_spacings):.4f}")
        print(f"  Min : {np.min(pixel_spacings):.4f}")
        print(f"  Max : {np.max(pixel_spacings):.4f}")
    else:
        print("\nImage Resolutions:")
        print(
            "  pydicom not available or failed to read headers. Pixel dimensions skipped."
        )

    # Bounding Box Analysis
    print("\nBOUNDING BOX ANALYSIS")
    if os.path.exists(BBOX_PATH):
        df_bbox = pd.read_csv(BBOX_PATH)
        # Filter bboxes to only those in our training set (df_train['StudyInstanceUID'])
        train_uids = set(df_train["StudyInstanceUID"])
        df_bbox_train = df_bbox[df_bbox["StudyInstanceUID"].isin(train_uids)]

        if len(df_bbox_train) > 0:
            print(f"  BBoxes in Train Set: {len(df_bbox_train)}")

            # Area
            df_bbox_train["area"] = df_bbox_train["width"] * df_bbox_train["height"]
            print(f"  Mean Area (px^2): {df_bbox_train['area'].mean():.4f}")
            print(f"  Min Area (px^2) : {df_bbox_train['area'].min():.4f}")
            print(f"  Max Area (px^2) : {df_bbox_train['area'].max():.4f}")

            # Aspect Ratio
            df_bbox_train["aspect_ratio"] = (
                df_bbox_train["width"] / df_bbox_train["height"]
            )
            print(f"  Mean Aspect Ratio: {df_bbox_train['aspect_ratio'].mean():.4f}")
        else:
            print(
                "  No bounding boxes found for the specific studies in the training split."
            )
    else:
        print("  train_bounding_boxes.csv not found.")

    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # A. Label Correlations
    print("Label Correlations (Pearson):")
    # Correlation between specific vertebrae fractures
    corr_matrix = df_train[vert_cols].corr()
    # Print pairs with correlation > 0.3
    printed_corr = False
    for i in range(len(vert_cols)):
        for j in range(i + 1, len(vert_cols)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.3:
                print(f"  {vert_cols[i]} - {vert_cols[j]}: {c:.4f}")
                printed_corr = True
    if not printed_corr:
        print("  No strong correlations (>0.3) between specific vertebrae pairs.")

    # B. Meta-Feature Importance (Random Forest)
    # Construct features
    # Map slice counts back to dataframe
    # We need a map from UID to slice count
    uid_to_slices = {}
    uid_to_size = {}

    # Re-loop to map (efficient enough for 161 items)
    for idx, row in df_train.iterrows():
        # We rely on the order matching or map by ID.
        # Let's map by ID to be safe.
        # Note: In the previous loop we appended to lists in order of iterrows,
        # so we can just assign if lengths match.
        pass

    df_train["num_slices"] = slice_counts
    df_train["avg_file_size"] = file_sizes

    # Features for RF
    rf_features = [
        "num_slices",
        "avg_file_size",
        "has_segmentation",
        "has_bounding_box",
    ]
    X = df_train[rf_features].fillna(0)
    y = df_train["patient_overall"]

    if len(df_train) > 10:
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=SEED)
        rf.fit(X, y)

        print("\nMeta-Feature Importance (RF predicting 'patient_overall'):")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        for f in range(len(rf_features)):
            print(f"  {rf_features[indices[f]]:<20}: {importances[indices[f]]:.4f}")

        # Correlation of num_slices with target
        corr_slices = df_train["num_slices"].corr(df_train["patient_overall"])
        print(f"\nCorrelation (Num Slices vs Target): {corr_slices:.4f}")
    else:
        print("\nNot enough data for reliable Feature Importance analysis.")


if __name__ == "__main__":
    run_eda()
