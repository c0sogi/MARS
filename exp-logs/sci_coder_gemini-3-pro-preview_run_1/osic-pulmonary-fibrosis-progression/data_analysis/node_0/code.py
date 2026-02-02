import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_header(title):
    print("\n" + "=" * 40)
    print(title.upper())
    print("=" * 40)


def analyze_target(df, target_col):
    print_header("2. Target Variable Analysis")

    target = df[target_col]

    # Distribution stats
    mean_val = target.mean()
    std_val = target.std()
    min_val = target.min()
    max_val = target.max()

    print(f"Target: {target_col}")
    print(f"Count: {len(target)}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")

    # Normality check (Skewness and Kurtosis)
    skew = target.skew()
    kurt = target.kurtosis()

    print(f"Skewness: {skew:.4f} (Pos > 0, Neg < 0)")
    print(f"Kurtosis: {kurt:.4f} (Normal ~ 0)")

    if abs(skew) > 1:
        print("-> Target distribution is highly skewed.")
    else:
        print("-> Target distribution is approximately symmetric.")


def analyze_tabular(df, num_cols, cat_cols):
    print_header("3. Input Data Analysis (Tabular)")

    # Numerical Analysis
    print("--- Numerical Features ---")
    for col in num_cols:
        if col not in df.columns:
            continue
        series = df[col]

        # Outlier detection (IQR)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        print(f"Feature: {col}")
        print(f"  Mean: {series.mean():.4f} | Std: {series.std():.4f}")
        print(f"  Min: {series.min():.4f} | Max: {series.max():.4f}")
        print(
            f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)"
        )

    # Categorical Analysis
    print("\n--- Categorical Features ---")
    for col in cat_cols:
        if col not in df.columns:
            continue
        series = df[col]
        unique_vals = series.unique()
        n_unique = len(unique_vals)

        print(f"Feature: {col}")
        print(f"  Cardinality: {n_unique}")

        # Check for high cardinality
        if n_unique > 50:
            print("  -> High cardinality flag (> 50 categories)")

        # Check for rare labels
        counts = series.value_counts(normalize=True)
        rare_labels = counts[counts < 0.01].index.tolist()
        if rare_labels:
            print(f"  -> Rare labels (< 1%): {rare_labels}")
        else:
            print("  -> No rare labels found.")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"{col}: {count} NaNs ({count/len(df)*100:.2f}%)")
    else:
        print("No missing values detected in the training set.")


def analyze_image_metadata(df, input_root):
    """
    Analyzes image data based on file system metadata (slice counts, file sizes)
    since specific DICOM libraries might not be available in the restricted environment.
    """
    print_header("3. Input Data Analysis (Image Metadata)")

    # We aggregate by Patient since images are stored in patient folders
    unique_patients = df["Patient"].unique()

    slice_counts = []
    file_sizes = []

    # Sample a subset of patients to keep within time limits if dataset is huge,
    # but here N=126 is small enough to process all.
    print(f"Analyzing DICOM directories for {len(unique_patients)} patients...")

    for patient in unique_patients:
        # Get directory from the first entry of this patient
        # The metadata 'dicom_dir' is relative to input root, e.g., "train/ID..."
        rel_path = df[df["Patient"] == patient]["dicom_dir"].iloc[0]
        full_path = os.path.join(input_root, rel_path)

        if os.path.exists(full_path):
            files = os.listdir(full_path)
            # Filter for .dcm files just in case
            dcm_files = [f for f in files if f.endswith(".dcm")]

            count = len(dcm_files)
            slice_counts.append(count)

            # Calculate average file size for this patient (proxy for resolution/depth)
            if count > 0:
                sizes = [os.path.getsize(os.path.join(full_path, f)) for f in dcm_files]
                avg_size = np.mean(sizes) / 1024.0  # KB
                file_sizes.append(avg_size)
            else:
                file_sizes.append(0)
        else:
            slice_counts.append(0)
            file_sizes.append(0)

    # Convert to arrays
    slice_counts = np.array(slice_counts)
    file_sizes = np.array(file_sizes)

    print("\n--- Image Dimensions (Slice Depth) ---")
    print(f"Distribution of Slice Counts per Patient:")
    print(f"  Mean: {np.mean(slice_counts):.4f}")
    print(f"  Std:  {np.std(slice_counts):.4f}")
    print(f"  Min:  {np.min(slice_counts):.4f}")
    print(f"  Max:  {np.max(slice_counts):.4f}")

    print("\n--- Image Signal (File Sizes) ---")
    print(f"Distribution of Avg File Size per Patient (KB):")
    print(f"  Mean: {np.mean(file_sizes):.4f}")
    print(f"  Std:  {np.std(file_sizes):.4f}")

    # Return these patient-level stats for relationship analysis
    return pd.DataFrame(
        {
            "Patient": unique_patients,
            "Slice_Count": slice_counts,
            "Avg_File_Size_KB": file_sizes,
        }
    )


def analyze_relationships(df, image_stats, target_col, num_cols, cat_cols):
    print_header("4. Feature/Signal Relationships")

    # Merge image stats back to main dataframe
    # Note: df has multiple rows per patient, image_stats has 1 row per patient.
    # We broadcast image stats to all patient visits.
    full_df = pd.merge(df, image_stats, on="Patient", how="left")

    # Prepare data for correlation and importance
    # Encode categorical variables
    analysis_df = full_df.copy()
    le = LabelEncoder()

    encoded_cats = []
    for col in cat_cols:
        if col in analysis_df.columns:
            analysis_df[f"{col}_enc"] = le.fit_transform(analysis_df[col].astype(str))
            encoded_cats.append(f"{col}_enc")

    # Define features for analysis
    # We include numerical features + encoded categorical + image meta features
    features = (
        [c for c in num_cols if c in analysis_df.columns]
        + encoded_cats
        + ["Slice_Count", "Avg_File_Size_KB"]
    )

    # Filter out target from features if present
    features = [f for f in features if f != target_col]

    # 1. Correlation Analysis
    print("--- Correlation with Target (Top 5) ---")
    # Calculate correlation of features with target
    correlations = (
        analysis_df[features + [target_col]].corr()[target_col].drop(target_col)
    )
    sorted_corr = correlations.abs().sort_values(ascending=False)

    for feat in sorted_corr.head(5).index:
        val = correlations[feat]
        print(f"{feat}: {val:.4f}")

    # Redundancy Check
    print("\n--- Collinearity Check (Corr > 0.90) ---")
    corr_matrix = analysis_df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

    if high_corr:
        for col in high_corr:
            correlated_feats = upper.index[upper[col] > 0.90].tolist()
            print(f"{col} is highly correlated with: {correlated_feats}")
    else:
        print("No highly collinear pairs found.")

    # 2. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    X = analysis_df[features].fillna(0)  # Simple impute for analysis
    y = analysis_df[target_col]

    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )

    for feat, imp in importances.head(5).items():
        print(f"{feat}: {imp:.4f}")

    # 3. Meta-Feature Relationships
    print("\n--- Meta-Feature Relationships ---")
    # Check correlation between Slice Count and Target
    slice_corr = correlations.get("Slice_Count", 0)
    print(f"Correlation between Slice Count (Depth) and FVC: {slice_corr:.4f}")
    if abs(slice_corr) > 0.1:
        print(
            "-> Weak but noticeable relationship between CT scan depth and lung capacity."
        )
    else:
        print(
            "-> No significant linear relationship between CT scan depth and lung capacity."
        )


def main():
    # 1. Setup
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    set_seed(42)

    # Load Data
    try:
        train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    except FileNotFoundError:
        print(
            "Error: Metadata files not found. Please ensure metadata generation was successful."
        )
        return

    # Define Column Types
    target_col = "FVC"
    num_cols = ["Weeks", "Percent", "Age"]
    cat_cols = ["Sex", "SmokingStatus"]

    # 2. Target Analysis
    analyze_target(train_df, target_col)

    # 3. Tabular Data Analysis
    analyze_tabular(train_df, num_cols, cat_cols)

    # 4. Image Data Analysis (Metadata Proxy)
    image_stats = analyze_image_metadata(train_df, INPUT_DIR)

    # 5. Relationship Analysis
    analyze_relationships(train_df, image_stats, target_col, num_cols, cat_cols)

    print("\n" + "=" * 40)
    print("EDA COMPLETE")
    print("=" * 40)


if __name__ == "__main__":
    main()
