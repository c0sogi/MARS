import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed():
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(SEED)


def load_data():
    """Loads the training metadata."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    return pd.read_csv(METADATA_PATH)


def analyze_target(df):
    """Analyzes the distribution of the target variable FVC."""
    print("TARGET VARIABLE ANALYSIS")
    target = df["FVC"]

    print(f"Target Variable: FVC")
    print(f"Count: {len(target)}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    skew = stats.skew(target)
    kurt = stats.kurtosis(target)
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")

    # Normality test
    if len(target) > 5000:
        stat, p = stats.jarque_bera(target)
        test_name = "Jarque-Bera"
    else:
        stat, p = stats.shapiro(target)
        test_name = "Shapiro-Wilk"
    print(f"Normality Test ({test_name}) p-value: {p:.4f}")
    print("-" * 30)


def analyze_tabular(df):
    """Analyzes numerical and categorical tabular features."""
    print("INPUT DATA ANALYSIS (TABULAR)")

    # Numerical Analysis
    num_cols = ["Weeks", "Percent", "Age"]
    print("Numerical Columns Analysis:")
    for col in num_cols:
        vals = df[col]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = vals[(vals < lower) | (vals > upper)]

        print(f"Column: {col}")
        print(f"  Mean: {vals.mean():.4f}")
        print(f"  Std: {vals.std():.4f}")
        print(f"  Min: {vals.min():.4f}")
        print(f"  Max: {vals.max():.4f}")
        print(
            f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(vals)*100:.2f}%)"
        )
        print(f"  Missing: {vals.isna().sum()}")

    # Categorical Analysis
    cat_cols = ["Sex", "SmokingStatus"]
    print("\nCategorical Columns Analysis:")
    for col in cat_cols:
        print(f"Column: {col}")
        counts = df[col].value_counts()
        print(f"  Cardinality: {len(counts)}")
        print(f"  Missing: {df[col].isna().sum()}")

        # Rare labels check
        total = len(df)
        ratios = counts / total
        rare = ratios[ratios < 0.01]
        if len(rare) > 0:
            print(f"  Rare labels (<1%): {list(rare.index)}")
        else:
            print(f"  No rare labels.")

    print("-" * 30)


def analyze_images(df):
    """Analyzes image data by sampling DICOM files."""
    print("INPUT DATA ANALYSIS (IMAGE)")

    unique_paths = df["image_path"].unique()
    print(f"Total Image Directories (Patients): {len(unique_paths)}")

    # Check for pydicom availability
    has_pydicom = False
    try:
        import pydicom

        has_pydicom = True
    except ImportError:
        print("Note: 'pydicom' library not found. Skipping pixel-level analysis.")

    # Sample patients to keep runtime low
    sample_size = min(30, len(unique_paths))
    sampled_paths = np.random.choice(unique_paths, sample_size, replace=False)

    slice_counts = []
    widths = []
    heights = []
    pixel_means = []
    pixel_stds = []
    file_sizes = []

    for rel_path in sampled_paths:
        full_dir_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_dir_path):
            continue

        files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]
        slice_counts.append(len(files))

        # Analyze file sizes
        sizes = [os.path.getsize(os.path.join(full_dir_path, f)) for f in files]
        if sizes:
            file_sizes.append(np.mean(sizes))

        # Analyze pixel data if pydicom is available
        if has_pydicom and len(files) > 0:
            # Sample middle slice
            mid_file = files[len(files) // 2]
            try:
                ds = pydicom.dcmread(os.path.join(full_dir_path, mid_file))
                widths.append(ds.Columns)
                heights.append(ds.Rows)
                if hasattr(ds, "pixel_array"):
                    arr = ds.pixel_array
                    pixel_means.append(np.mean(arr))
                    pixel_stds.append(np.std(arr))
            except Exception:
                pass

    # Report Stats
    print("Image Metadata & Stats (Sampled):")
    if slice_counts:
        print(
            f"  Slice Count per Patient: Mean={np.mean(slice_counts):.4f}, Std={np.std(slice_counts):.4f}"
        )
    if file_sizes:
        print(f"  Avg File Size (bytes): Mean={np.mean(file_sizes):.4f}")

    if has_pydicom and widths:
        print(
            f"  Widths: Mean={np.mean(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"  Heights: Mean={np.mean(heights):.4f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(f"  Pixel Value Mean (Global): {np.mean(pixel_means):.4f}")
        print(f"  Pixel Value Std (Global): {np.mean(pixel_stds):.4f}")
    elif not has_pydicom:
        print("  Pixel analysis skipped (pydicom missing).")

    print("-" * 30)


def analyze_relationships(df):
    """Analyzes relationships between features and the target."""
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Prepare Data
    work_df = df.copy()

    # Encode Categoricals
    le_sex = LabelEncoder()
    le_smoke = LabelEncoder()

    work_df["Sex_Code"] = le_sex.fit_transform(work_df["Sex"].astype(str))
    work_df["Smoking_Code"] = le_smoke.fit_transform(
        work_df["SmokingStatus"].astype(str)
    )

    features = ["Weeks", "Percent", "Age", "Sex_Code", "Smoking_Code"]
    target = "FVC"

    # Correlation
    print("Structured Relationships:")
    corr_matrix = work_df[features + [target]].corr(method="pearson")
    print("Correlation with Target (FVC):")
    print(corr_matrix[target].sort_values(ascending=False))

    # Redundancy Check
    print("\nRedundancy (Collinear Pairs > 0.90):")
    found_collinear = False
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.90:
                print(f"  {features[i]} - {features[j]}: {c:.4f}")
                found_collinear = True
    if not found_collinear:
        print("  No highly collinear pairs found.")

    # Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest):")
    X = work_df[features].fillna(0)
    y = work_df[target]

    rf = RandomForestRegressor(
        n_estimators=50, random_state=SEED, n_jobs=-1, max_depth=10, verbose=0
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(min(5, len(features))):
        print(f"  {features[indices[f]]}: {importances[indices[f]]:.4f}")

    # Unstructured / Meta-feature Relationships
    print("\nUnstructured (Meta-Feature) Relationships:")
    # Group by patient to get baseline stats
    patient_stats = (
        df.groupby("Patient")
        .agg({"FVC": "mean", "Weeks": "count", "Age": "first"})
        .rename(columns={"Weeks": "Num_Visits", "FVC": "Mean_FVC"})
    )

    # Correlation between number of visits and mean FVC
    corr_visits_fvc = patient_stats["Num_Visits"].corr(patient_stats["Mean_FVC"])
    print(f"  Correlation (Num Visits vs Mean FVC): {corr_visits_fvc:.4f}")

    # Correlation between Age and Mean FVC (Patient level)
    corr_age_fvc = patient_stats["Age"].corr(patient_stats["Mean_FVC"])
    print(f"  Correlation (Age vs Mean FVC): {corr_age_fvc:.4f}")

    print("-" * 30)


def main():
    set_seed()
    try:
        df = load_data()
        analyze_target(df)
        analyze_tabular(df)
        analyze_images(df)
        analyze_relationships(df)
    except Exception as e:
        print(f"An error occurred during EDA: {e}")


if __name__ == "__main__":
    main()
