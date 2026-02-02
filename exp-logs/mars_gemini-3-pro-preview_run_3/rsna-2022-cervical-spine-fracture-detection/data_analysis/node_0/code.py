import os
import glob
import sys
import pandas as pd
import numpy as np
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"


def print_section(title):
    print(f"\n{title.upper()}")
    print("=" * len(title))


def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # 2. Target Variable Analysis
    print_section("Target Variable Analysis")

    target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    # Distribution
    print("--- Class Balance ---")
    for col in target_cols:
        if col in df.columns:
            pos_count = df[col].sum()
            total = len(df)
            ratio = pos_count / total
            print(
                f"{col:<15}: Positive={pos_count:<4} Total={total:<4} Ratio={ratio:.4f}"
            )

    # Label Consistency Check
    # patient_overall should be 1 if any C1-C7 is 1
    if all(c in df.columns for c in target_cols):
        c_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        df["derived_overall"] = df[c_cols].max(axis=1)
        inconsistent = df[df["patient_overall"] != df["derived_overall"]]
        print(
            f"\nLabel Consistency: {len(inconsistent)} inconsistent rows found (patient_overall != max(C1-C7))."
        )

        # Co-occurrence Matrix
        print("\n--- Label Correlation (Pearson) ---")
        corr_matrix = df[target_cols].corr()
        # Print a simplified view (correlation with patient_overall)
        print(corr_matrix["patient_overall"].sort_values(ascending=False).to_string())

    # 3. Input Data Analysis (Image Modality)
    print_section("Input Data Analysis (Image Modality)")

    # We will iterate through the training samples to gather meta-features
    # Since reading DICOM pixels for all might be slow/impossible without pydicom,
    # we focus on file system stats (Depth) and try pydicom on a subset.

    # Check for pydicom availability
    try:
        import pydicom

        has_pydicom = True
    except ImportError:
        has_pydicom = False
        print(
            "Note: 'pydicom' library not found. Pixel-level analysis will be skipped."
        )

    image_stats = []

    # Iterate through all training samples (161 is small enough)
    print(f"Analyzing {len(df)} study directories...")

    for idx, row in df.iterrows():
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Check directory
        if not os.path.exists(full_path):
            continue

        # List files (slices)
        # DICOM files usually end in .dcm, but sometimes no extension
        files = glob.glob(os.path.join(full_path, "*"))
        num_slices = len(files)

        # Calculate average file size (proxy for information content/resolution)
        if num_slices > 0:
            sizes = [os.path.getsize(f) for f in files]
            avg_size_mb = np.mean(sizes) / (1024 * 1024)
        else:
            avg_size_mb = 0

        # DICOM Metadata extraction (Sampled)
        # We only try to read the header of the middle slice for the first few samples to save time
        # or if pydicom is available.
        dcm_rows, dcm_cols = np.nan, np.nan
        pixel_spacing = np.nan

        if has_pydicom and num_slices > 0:
            try:
                # Read middle slice
                mid_file = files[len(files) // 2]
                ds = pydicom.dcmread(mid_file, stop_before_pixels=True)
                dcm_rows = float(ds.Rows)
                dcm_cols = float(ds.Columns)
                if hasattr(ds, "PixelSpacing"):
                    pixel_spacing = float(ds.PixelSpacing[0])
            except Exception:
                pass

        image_stats.append(
            {
                "StudyInstanceUID": row["StudyInstanceUID"],
                "num_slices": num_slices,
                "avg_file_size_mb": avg_size_mb,
                "img_rows": dcm_rows,
                "img_cols": dcm_cols,
                "pixel_spacing": pixel_spacing,
            }
        )

    img_df = pd.DataFrame(image_stats)

    # Report Image Stats
    if not img_df.empty:
        print("\n--- Volume Dimensions (Slices per Scan) ---")
        print(f"Mean: {img_df['num_slices'].mean():.4f}")
        print(f"Std : {img_df['num_slices'].std():.4f}")
        print(f"Min : {img_df['num_slices'].min():.4f}")
        print(f"Max : {img_df['num_slices'].max():.4f}")

        print("\n--- File Sizes (MB per Slice) ---")
        print(f"Mean: {img_df['avg_file_size_mb'].mean():.4f}")

        if has_pydicom:
            print("\n--- Image Dimensions (Sampled) ---")
            # Filter NaNs
            valid_dims = img_df.dropna(subset=["img_rows"])
            if not valid_dims.empty:
                print(
                    f"Common Resolutions: {valid_dims.groupby(['img_rows', 'img_cols']).size().to_dict()}"
                )
                print(
                    f"Mean Pixel Spacing: {valid_dims['pixel_spacing'].mean():.4f} mm"
                )
            else:
                print("Could not extract dimensions from DICOM headers.")
        else:
            print("\nImage dimensions not available (pydicom missing).")

    # 4. Feature/Signal Relationships
    print_section("Feature/Signal Relationships")

    # Merge extracted image features with targets
    merged_df = pd.merge(df, img_df, on="StudyInstanceUID", how="inner")

    if not merged_df.empty:
        # Correlation between Volume Depth and Fracture Presence
        # Hypothesis: Trauma scans might be more detailed (more slices) or focused?
        corr_depth = merged_df["num_slices"].corr(merged_df["patient_overall"])
        corr_size = merged_df["avg_file_size_mb"].corr(merged_df["patient_overall"])

        print("--- Meta-Feature Correlations with Target (patient_overall) ---")
        print(f"Number of Slices (Depth) : {corr_depth:.4f}")
        print(f"Average File Size        : {corr_size:.4f}")

        # Feature Importance using Random Forest
        # We use the extracted meta-features to see if they predict fracture
        print("\n--- Meta-Feature Importance (Random Forest) ---")

        feature_cols = ["num_slices", "avg_file_size_mb"]
        X = merged_df[feature_cols].fillna(0)
        y = merged_df["patient_overall"]

        if len(X) > 10:
            rf = RandomForestClassifier(
                n_estimators=50, random_state=RANDOM_STATE, max_depth=3
            )
            rf.fit(X, y)

            importances = rf.feature_importances_
            for name, imp in zip(feature_cols, importances):
                print(f"{name:<20}: {imp:.4f}")
        else:
            print("Not enough data for Random Forest analysis.")

    # 5. Tabular Data Analysis (Metadata specifics)
    print_section("Tabular Metadata Analysis")

    # Check for missing values in the metadata provided
    missing = df.isnull().sum()
    print("--- Missing Values in Metadata ---")
    if missing.sum() == 0:
        print("No missing values found in training metadata.")
    else:
        print(missing[missing > 0])


if __name__ == "__main__":
    main()
