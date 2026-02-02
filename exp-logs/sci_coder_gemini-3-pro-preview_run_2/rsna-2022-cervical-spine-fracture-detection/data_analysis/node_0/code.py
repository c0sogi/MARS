import os
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
METADATA_FILE = "./metadata/train_metadata.csv"
INPUT_ROOT = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train_images")
SEED = 42

# Set Seeds for Reproducibility
np.random.seed(SEED)


def main():
    # --- 1. Data Integrity ---
    if not os.path.exists(METADATA_FILE):
        print("ERROR: Metadata file not found.")
        return

    df = pd.read_csv(METADATA_FILE)

    print("DATA INTEGRITY")
    print(f"Analysis performed on training set with {len(df)} samples.")
    print(f"Unique Studies: {df['StudyInstanceUID'].nunique()}")
    print("-" * 30)

    # --- 2. Target Variable Analysis ---
    print("\nTARGET VARIABLE ANALYSIS")

    target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    # Distribution / Imbalance
    print("Class Balance (Prevalence):")
    for col in target_cols:
        prevalence = df[col].mean()
        count = df[col].sum()
        print(f"  {col:<15}: {count:5d} / {len(df)} ({prevalence:.4f})")

    # Multi-label analysis
    fracture_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    df["fracture_count"] = df[fracture_cols].sum(axis=1)

    print(f"\nFracture Count per Patient (Distribution):")
    counts = df["fracture_count"].value_counts(normalize=True).sort_index()
    for k, v in counts.items():
        print(f"  {k} fractures: {v:.4f}")

    # Correlation between targets
    print("\nTarget Correlations (Pearson > 0.3):")
    corr = df[target_cols].corr()
    # Mask diagonal and lower triangle to avoid duplicates
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    high_corr = corr.where(mask).stack()
    high_corr = high_corr[abs(high_corr) > 0.3]

    if len(high_corr) > 0:
        for idx, val in high_corr.items():
            print(f"  {idx[0]} vs {idx[1]}: {val:.4f}")
    else:
        print("  No correlations > 0.3 found between targets.")

    # --- 3. Input Data Analysis (Image Modality) ---
    print("\nINPUT DATA ANALYSIS (IMAGE)")

    # Check for pydicom availability
    try:
        import pydicom

        has_pydicom = True
    except ImportError:
        has_pydicom = False
        print("  (Note: pydicom not installed. Analyzing file system metadata only.)")

    # Sampling strategy to save time
    sample_size = min(100, len(df))
    sample_uids = (
        df["StudyInstanceUID"].sample(n=sample_size, random_state=SEED).tolist()
    )

    slice_counts = []
    file_sizes_kb = []
    widths = []
    heights = []
    pixel_means = []

    for uid in sample_uids:
        study_path = os.path.join(TRAIN_IMG_DIR, uid)
        if not os.path.exists(study_path):
            continue

        # Count slices (files)
        try:
            files = os.listdir(study_path)
            dcm_files = [f for f in files if f.endswith(".dcm")]
            slice_counts.append(len(dcm_files))

            if dcm_files:
                # Get file size of the middle slice
                mid_idx = len(dcm_files) // 2
                sample_file_path = os.path.join(study_path, dcm_files[mid_idx])
                file_sizes_kb.append(os.path.getsize(sample_file_path) / 1024)

                # Attempt to read DICOM headers/pixels if library exists
                if has_pydicom:
                    try:
                        ds = pydicom.dcmread(sample_file_path, stop_before_pixels=False)
                        widths.append(ds.Columns)
                        heights.append(ds.Rows)
                        if hasattr(ds, "pixel_array"):
                            pixel_means.append(ds.pixel_array.mean())
                    except Exception:
                        pass
        except OSError:
            pass

    # Report Slice Counts (Z-Depth)
    if slice_counts:
        sc = np.array(slice_counts)
        print(f"Slice Count (Z-Depth) Distribution (Sample N={len(sc)}):")
        print(f"  Mean: {np.mean(sc):.4f}")
        print(f"  Std:  {np.std(sc):.4f}")
        print(f"  Min:  {np.min(sc)}")
        print(f"  Max:  {np.max(sc)}")

    # Report File Sizes
    if file_sizes_kb:
        fs = np.array(file_sizes_kb)
        print(f"File Size (KB) Distribution:")
        print(f"  Mean: {np.mean(fs):.4f}")
        print(f"  Std:  {np.std(fs):.4f}")

    # Report Dimensions (if pydicom worked)
    if has_pydicom and widths:
        w = np.array(widths)
        h = np.array(heights)
        print(f"Image Dimensions (Sample N={len(w)}):")
        print(f"  Width Mean:  {np.mean(w):.4f}")
        print(f"  Height Mean: {np.mean(h):.4f}")

        unique_dims = set(zip(w, h))
        if len(unique_dims) == 1:
            print(
                f"  All sampled images have constant dimensions: {list(unique_dims)[0]}"
            )
        else:
            print(f"  Unique dimensions found: {unique_dims}")

        if pixel_means:
            pm = np.array(pixel_means)
            print(f"Pixel Value Mean (Global Sample): {np.mean(pm):.4f}")
            print(f"Pixel Value Std (Global Sample):  {np.std(pm):.4f}")

    # --- 4. Feature/Signal Relationships ---
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # We analyze if metadata (slice count, file size) correlates with the target.
    # This checks if "fractured" scans are systematically different (e.g. trauma protocol).

    # Create a DataFrame for the sampled data
    meta_features = []
    for uid in sample_uids:
        study_path = os.path.join(TRAIN_IMG_DIR, uid)
        if not os.path.exists(study_path):
            continue
        try:
            files = [f for f in os.listdir(study_path) if f.endswith(".dcm")]
            count = len(files)
            size = 0
            if count > 0:
                size = os.path.getsize(os.path.join(study_path, files[0]))

            meta_features.append(
                {"StudyInstanceUID": uid, "slice_count": count, "file_size_bytes": size}
            )
        except OSError:
            pass

    meta_df = pd.DataFrame(meta_features)

    # Merge with original targets
    analysis_df = pd.merge(df, meta_df, on="StudyInstanceUID", how="inner")

    if not analysis_df.empty and "slice_count" in analysis_df.columns:
        # Correlation: Slice Count vs Patient Overall
        corr_slices = analysis_df["slice_count"].corr(analysis_df["patient_overall"])
        print(f"Correlation (Slice Count vs Target): {corr_slices:.4f}")

        # Correlation: File Size vs Patient Overall
        corr_size = analysis_df["file_size_bytes"].corr(analysis_df["patient_overall"])
        print(f"Correlation (File Size vs Target):   {corr_size:.4f}")

        # Feature Importance (Random Forest)
        # We try to predict 'patient_overall' using only metadata
        X = analysis_df[["slice_count", "file_size_bytes"]]
        y = analysis_df["patient_overall"]

        # Handle NaNs if any
        X = X.fillna(0)

        if len(X) > 10:
            rf = RandomForestClassifier(n_estimators=50, random_state=SEED, max_depth=3)
            rf.fit(X, y)

            print("Meta-Feature Importance (Random Forest):")
            for name, imp in zip(X.columns, rf.feature_importances_):
                print(f"  {name:<20}: {imp:.4f}")
    else:
        print("Insufficient data for relationship analysis.")


if __name__ == "__main__":
    main()
