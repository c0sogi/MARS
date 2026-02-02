import os
import pandas as pd
import numpy as np
import random
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ------------------------------------------------------------------------------
# Analysis Functions
# ------------------------------------------------------------------------------


def perform_eda():
    print("STARTING EXPLORATORY DATA ANALYSIS\n")

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # --------------------------------------------------------------------------
    # SECTION 1: DATA INTEGRITY
    # --------------------------------------------------------------------------
    print("DATA INTEGRITY")
    print(f"Analysis performed strictly on: {METADATA_PATH}")
    print(f"Total training samples: {len(df)}")
    print("-" * 30)

    # --------------------------------------------------------------------------
    # SECTION 2: TARGET VARIABLE ANALYSIS
    # --------------------------------------------------------------------------
    print("TARGET VARIABLE ANALYSIS")
    target_col = "MGMT_value"

    # Distribution
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True)

    print(f"Target: {target_col} (Binary Classification)")
    print(f"Class 0 Count: {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"Class 1 Count: {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    # Balance Ratio
    ratio = counts.get(1, 0) / counts.get(0, 1) if counts.get(0, 1) > 0 else 0
    print(f"Class Balance Ratio (1:0): {ratio:.4f}")
    print("-" * 30)

    # --------------------------------------------------------------------------
    # SECTION 3: INPUT DATA ANALYSIS (IMAGE)
    # --------------------------------------------------------------------------
    print("INPUT DATA ANALYSIS (IMAGE)")

    # We need to extract features from the images.
    # Since reading every slice of every patient is slow, we will sample.
    # We will try to read pixels with cv2. If fails, we fallback to file stats.

    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    stats_data = []

    # Check if we can read a sample file
    sample_path_rel = df.iloc[0]["path_FLAIR"]
    sample_full_dir = os.path.join(INPUT_DIR, sample_path_rel)
    sample_files = os.listdir(sample_full_dir)
    can_read_pixels = False

    if sample_files:
        test_file = os.path.join(sample_full_dir, sample_files[0])
        try:
            img = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
            if img is not None:
                can_read_pixels = True
        except Exception:
            pass

    print(f"Pixel-level access available (via OpenCV): {can_read_pixels}")
    if not can_read_pixels:
        print(
            "Note: DICOM reading via OpenCV failed/unsupported. Analyzing File System Metadata (Slice Counts/Sizes)."
        )

    # Iterate through all training samples to gather metadata
    # We will collect: Slice Counts, File Sizes (proxy for info), and Dimensions (if readable)

    widths = []
    heights = []
    pixel_means = []
    pixel_stds = []

    # For structured dataframe
    meta_features = []

    print(f"Processing {len(df)} subjects for metadata extraction...")

    for idx, row in df.iterrows():
        subject_stats = {"BraTS21ID": row["BraTS21ID"], "MGMT_value": row["MGMT_value"]}

        for mod in modalities:
            col_name = f"path_{mod}"
            dir_path = os.path.join(INPUT_DIR, row[col_name])

            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                num_slices = len(files)

                # Calculate total size of folder
                total_size_bytes = sum(
                    os.path.getsize(os.path.join(dir_path, f)) for f in files
                )
                avg_file_size = total_size_bytes / num_slices if num_slices > 0 else 0

                subject_stats[f"{mod}_slices"] = num_slices
                subject_stats[f"{mod}_avg_size"] = avg_file_size

                # Pixel sampling (Only if readable and for the first modality to save time/redundancy)
                if can_read_pixels and num_slices > 0:
                    # Sample middle slice
                    mid_slice = files[num_slices // 2]
                    img_path = os.path.join(dir_path, mid_slice)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                    if img is not None:
                        if len(img.shape) == 2:
                            h, w = img.shape
                        else:
                            h, w, c = img.shape

                        # Collect global stats only for FLAIR to avoid skewing distribution with too many points
                        if mod == "FLAIR":
                            widths.append(w)
                            heights.append(h)
                            pixel_means.append(np.mean(img))
                            pixel_stds.append(np.std(img))

                        # Add to subject stats
                        subject_stats[f"{mod}_mean_intensity"] = np.mean(img)
            else:
                subject_stats[f"{mod}_slices"] = 0
                subject_stats[f"{mod}_avg_size"] = 0
                if can_read_pixels:
                    subject_stats[f"{mod}_mean_intensity"] = 0

        meta_features.append(subject_stats)

    df_meta = pd.DataFrame(meta_features)

    # Report Dimensions (if available)
    if can_read_pixels and widths:
        print("\nImage Dimensions (Sampled from FLAIR):")
        print(
            f"Width  - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}, Min: {np.min(widths)}, Max: {np.max(widths)}"
        )
        print(
            f"Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}, Min: {np.min(heights)}, Max: {np.max(heights)}"
        )

        # Aspect Ratio
        ars = np.array(widths) / np.array(heights)
        print(f"Aspect Ratio - Mean: {np.mean(ars):.4f}")

        print("\nPixel Intensity Stats (Global Normalization Info):")
        print(f"Global Mean: {np.mean(pixel_means):.4f}")
        print(f"Global Std:  {np.mean(pixel_stds):.4f}")
    else:
        print("\nImage Dimensions & Pixel Stats:")
        print("Not available (DICOM files could not be read with installed libraries).")

    # Report Slice Counts (Volume Depth)
    print("\nVolume Depth (Slice Counts) per Modality:")
    for mod in modalities:
        col = f"{mod}_slices"
        print(
            f"{mod}: Mean={df_meta[col].mean():.4f}, Std={df_meta[col].std():.4f}, Min={df_meta[col].min()}, Max={df_meta[col].max()}"
        )

    print("-" * 30)

    # --------------------------------------------------------------------------
    # SECTION 4: FEATURE/SIGNAL RELATIONSHIPS
    # --------------------------------------------------------------------------
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Correlation Analysis
    # Select numerical columns
    num_cols = [c for c in df_meta.columns if c not in ["BraTS21ID", "MGMT_value"]]

    # Remove constant columns
    num_cols = [c for c in num_cols if df_meta[c].std() > 0]

    if num_cols:
        # Correlation with Target
        corrs = df_meta[num_cols].apply(lambda x: x.corr(df_meta["MGMT_value"]))
        print("\nTop 5 Correlations with Target (Pearson):")
        print(corrs.abs().sort_values(ascending=False).head(5))

        # Redundancy (Collinearity)
        print("\nRedundant Feature Pairs (Correlation > 0.90):")
        corr_matrix = df_meta[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        pairs_found = False
        for col in to_drop:
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            for row in correlated_with:
                print(f"  {row} <--> {col} : {upper.loc[row, col]:.4f}")
                pairs_found = True
        if not pairs_found:
            print("  None found.")

        # 2. Feature Importance (Random Forest)
        print("\nMultivariate Feature Importance (Random Forest):")
        X = df_meta[num_cols].fillna(0)
        y = df_meta["MGMT_value"]

        rf = RandomForestClassifier(
            n_estimators=100, random_state=SEED, max_depth=5, n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=num_cols)
        print(importances.sort_values(ascending=False).head(5))

        # Meta-Feature Insight
        print("\nInsight:")
        top_feat = importances.idxmax()
        print(f"The most important meta-feature is '{top_feat}'.")
        print(
            "This suggests that "
            + (
                "structural properties (scan depth/size)"
                if "slice" in top_feat or "size" in top_feat
                else "intensity properties"
            )
            + " have "
            + ("weak" if importances.max() < 0.1 else "some")
            + " predictive signal."
        )

    else:
        print("No numerical features extracted for relationship analysis.")

    print("-" * 30)


if __name__ == "__main__":
    perform_eda()
