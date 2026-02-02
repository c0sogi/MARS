import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

# --- Configuration ---
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"


def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # ---------------------------------------------------------
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("SECTION 1: TARGET VARIABLE ANALYSIS")
    target = "FVC"
    fvc_data = df[target]

    print(f"Target Variable: {target}")
    print(f"Mean: {fvc_data.mean():.4f}")
    print(f"Std Dev: {fvc_data.std():.4f}")
    print(f"Min: {fvc_data.min():.4f}")
    print(f"Max: {fvc_data.max():.4f}")

    # Normality Check
    fvc_skew = skew(fvc_data)
    fvc_kurt = kurtosis(fvc_data)
    print(
        f"Skewness: {fvc_skew:.4f} ({'Right Skewed' if fvc_skew > 0 else 'Left Skewed'})"
    )
    print(f"Kurtosis: {fvc_kurt:.4f}")
    print("-" * 30)

    # ---------------------------------------------------------
    # SECTION 2: TABULAR DATA ANALYSIS
    # ---------------------------------------------------------
    print("\nSECTION 2: TABULAR INPUT DATA ANALYSIS")

    # Numerical Features
    num_cols = ["Weeks", "Age", "Percent"]
    print("--- Numerical Features ---")
    for col in num_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        print(f"Feature: {col}")
        print(f"  Mean: {series.mean():.4f} | Std: {series.std():.4f}")
        print(f"  Min: {series.min():.4f} | Max: {series.max():.4f}")
        print(f"  Outliers (IQR method): {outliers} ({outliers/len(df)*100:.2f}%)")

    # Categorical Features
    cat_cols = ["Sex", "SmokingStatus"]
    print("\n--- Categorical Features ---")
    for col in cat_cols:
        print(f"Feature: {col}")
        counts = df[col].value_counts()
        print(f"  Cardinality: {df[col].nunique()}")
        print(f"  Top Category: {counts.index[0]} ({counts.iloc[0]} samples)")
        if len(counts) > 1 and counts.iloc[-1] / len(df) < 0.01:
            print(f"  Warning: Rare label detected '{counts.index[-1]}'")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values detected.")
    else:
        print(missing[missing > 0])
    print("-" * 30)

    # ---------------------------------------------------------
    # SECTION 3: IMAGE DATA ANALYSIS (METADATA & VOLUMETRICS)
    # ---------------------------------------------------------
    print("\nSECTION 3: IMAGE DATA ANALYSIS")
    # Note: Direct pixel analysis requires pydicom which is not in the allowed list.
    # We will analyze the physical properties of the scan (Scan Depth) via file counts.

    scan_depths = {}
    unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()

    print(f"Analyzing DICOM directories for {len(unique_patients)} unique patients...")

    for _, row in unique_patients.iterrows():
        patient_id = row["Patient"]
        rel_path = row["dcm_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            # Count files ending in .dcm
            num_slices = len(
                [
                    name
                    for name in os.listdir(full_path)
                    if name.lower().endswith(".dcm")
                ]
            )
            scan_depths[patient_id] = num_slices
        else:
            scan_depths[patient_id] = 0

    # Map scan depths back to dataframe
    df["ScanDepth"] = df["Patient"].map(scan_depths)

    # Analyze Scan Depth Distribution
    sd = df["ScanDepth"]
    print("\n--- Scan Volume (Slices per Patient) ---")
    print(f"Mean Slices: {sd.mean():.4f}")
    print(f"Std Slices: {sd.std():.4f}")
    print(f"Min Slices: {sd.min()} | Max Slices: {sd.max()}")
    print("-" * 30)

    # ---------------------------------------------------------
    # SECTION 4: FEATURE RELATIONSHIPS
    # ---------------------------------------------------------
    print("\nSECTION 4: FEATURE RELATIONSHIPS")

    # Prepare data for Correlation and RF
    eda_df = df.copy()

    # Encode Categoricals
    le = LabelEncoder()
    for col in cat_cols:
        eda_df[col] = le.fit_transform(eda_df[col].astype(str))

    # 1. Correlation Matrix
    print("--- Correlation with Target (FVC) ---")
    # Select numerical columns + encoded categoricals + ScanDepth
    corr_cols = num_cols + cat_cols + ["ScanDepth", "FVC"]
    corr_matrix = eda_df[corr_cols].corr()
    target_corr = corr_matrix["FVC"].sort_values(ascending=False)

    # Drop self-correlation
    target_corr = target_corr.drop("FVC")

    for feat, corr in target_corr.items():
        print(f"{feat}: {corr:.4f}")

    # Check for Collinearity (Redundancy)
    print("\n--- Feature Redundancy (Corr > 0.90) ---")
    high_corr_pairs = []
    features = corr_cols[:-1]  # Exclude target
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.loc[features[i], features[j]]
            if abs(c) > 0.90:
                high_corr_pairs.append((features[i], features[j], c))

    if high_corr_pairs:
        for f1, f2, c in high_corr_pairs:
            print(f"High Collinearity: {f1} & {f2} (Corr: {c:.4f})")
    else:
        print("No highly collinear features found.")

    # 2. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    X = eda_df[features]
    y = eda_df["FVC"]

    rf = RandomForestRegressor(
        n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=10
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )

    print("Top 5 Features predicting FVC:")
    for i, (feat, imp) in enumerate(importances.head(5).items(), 1):
        print(f"{i}. {feat}: {imp:.4f}")

    # Meta-feature insight
    print("\n--- Meta-Feature Insight ---")
    scan_corr = corr_matrix.loc["ScanDepth", "FVC"]
    print(f"Correlation between Scan Depth (Image Count) and FVC: {scan_corr:.4f}")
    if abs(scan_corr) < 0.1:
        print(
            "Insight: The number of CT slices does not appear to be strongly linearly related to lung capacity."
        )
    else:
        print(
            "Insight: There is a notable linear relationship between scan volume and lung capacity."
        )


if __name__ == "__main__":
    main()
