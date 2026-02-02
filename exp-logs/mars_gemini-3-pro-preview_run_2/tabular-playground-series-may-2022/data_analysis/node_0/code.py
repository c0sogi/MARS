import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


def main():
    print("SECTION 1: DATA LOADING AND INTEGRITY")

    # Load Metadata
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        print(f"Error: Metadata file not found at {train_meta_path}")
        return

    df_meta = pd.read_csv(train_meta_path)
    train_ids = set(df_meta["id"].values)

    print(f"Metadata loaded. Training set size: {len(train_ids)}")

    # Load Raw Data
    raw_train_path = os.path.join(INPUT_DIR, "train.csv")
    if not os.path.exists(raw_train_path):
        print(f"Error: Raw data file not found at {raw_train_path}")
        return

    # Read full train csv
    df_raw = pd.read_csv(raw_train_path)

    # Filter for training set only (Data Integrity)
    df = df_raw[df_raw["id"].isin(train_ids)].copy()

    # Verify alignment
    if len(df) != len(train_ids):
        print("Warning: Mismatch between metadata IDs and found IDs in raw file.")

    # Drop ID as it is not a feature
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    print(f"Data loaded and filtered. Shape: {df.shape}")
    print("-" * 30)

    # --------------------------------------------------------------------------
    # Target Variable Analysis
    # --------------------------------------------------------------------------
    print("\nSECTION 2: TARGET VARIABLE ANALYSIS")

    target_col = "target"
    if target_col not in df.columns:
        print("Error: Target column 'target' not found in dataset.")
        return

    # Distribution
    target_counts = df[target_col].value_counts()
    target_ratios = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: '{target_col}'")
    for label, count in target_counts.items():
        ratio = target_ratios[label]
        print(f"Class {label}: {count} samples ({ratio:.4f})")

    # Imbalance Check
    if len(target_ratios) == 2:
        # Binary classification
        ratio_0 = target_ratios.get(0, 0)
        ratio_1 = target_ratios.get(1, 0)
        print(f"Class Balance Ratio (0:1): {ratio_0:.4f} : {ratio_1:.4f}")
    else:
        print(f"Multiclass distribution found with {len(target_ratios)} classes.")

    print("-" * 30)

    # --------------------------------------------------------------------------
    # Input Data Analysis (Tabular)
    # --------------------------------------------------------------------------
    print("\nSECTION 3: INPUT DATA ANALYSIS (TABULAR)")

    # Separate features from target
    X = df.drop(columns=[target_col])

    # Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numerical Columns: {len(num_cols)}")
    print(f"Categorical Columns: {len(cat_cols)}")

    # 3.1 Numerical Analysis
    if num_cols:
        print("\n-- Numerical Features Statistics --")
        # We compute stats manually or via describe to format output
        desc = X[num_cols].describe().T

        # Outlier Analysis (IQR)
        outlier_report = []
        for col in num_cols:
            series = X[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = series[(series < lower_bound) | (series > upper_bound)]
            n_outliers = len(outliers)
            outlier_report.append(n_outliers)

        desc["outliers_iqr"] = outlier_report

        # Print a summary of the first 10 columns to avoid flooding output if many cols
        print(
            f"{'Feature':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers':<10}"
        )
        for idx, row in desc.head(10).iterrows():
            print(
                f"{idx:<20} {row['mean']:<10.4f} {row['std']:<10.4f} {row['min']:<10.4f} {row['max']:<10.4f} {int(row['outliers_iqr']):<10}"
            )
        if len(desc) > 10:
            print(f"... and {len(desc) - 10} more numerical features.")

    # 3.2 Categorical Analysis
    if cat_cols:
        print("\n-- Categorical Features Statistics --")
        print(
            f"{'Feature':<20} {'Cardinality':<12} {'High Card.':<10} {'Rare Labels':<10}"
        )

        for col in cat_cols:
            series = X[col].astype(str)  # Handle mixed types if any
            counts = series.value_counts()
            ratios = series.value_counts(normalize=True)

            cardinality = len(counts)
            high_card = "YES" if cardinality > 50 else "NO"

            # Rare labels < 1%
            rare_count = (ratios < 0.01).sum()
            has_rare = "YES" if rare_count > 0 else "NO"

            print(f"{col:<20} {cardinality:<12} {high_card:<10} {has_rare:<10}")

    # 3.3 Missing Values
    print("\n-- Missing Values --")
    missing_series = X.isnull().mean()
    missing_cols = missing_series[missing_series > 0]

    if len(missing_cols) == 0:
        print("No missing values found in features.")
    else:
        print(f"{'Feature':<20} {'Missing %':<10}")
        for col, ratio in missing_cols.items():
            print(f"{col:<20} {ratio:.4f}")

    print("-" * 30)

    # --------------------------------------------------------------------------
    # Feature Relationships
    # --------------------------------------------------------------------------
    print("\nSECTION 4: FEATURE/SIGNAL RELATIONSHIPS")

    # Sampling for expensive operations if dataset is huge
    # 640k rows is manageable, but for correlation/RF on many columns, let's cap at 100k for speed
    MAX_SAMPLES_EDA = 100000
    if len(df) > MAX_SAMPLES_EDA:
        print(
            f"Subsampling {MAX_SAMPLES_EDA} rows for relationship analysis to ensure runtime constraints..."
        )
        df_sample = df.sample(n=MAX_SAMPLES_EDA, random_state=RANDOM_STATE)
    else:
        df_sample = df

    X_sample = df_sample.drop(columns=[target_col])
    y_sample = df_sample[target_col]

    # 4.1 Redundancy (Correlation)
    # Only for numerical columns
    if num_cols:
        print("\n-- Redundancy Check (Collinear Pairs > 0.90) --")
        # Calculate correlation matrix
        corr_matrix = X_sample[num_cols].corr().abs()

        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        high_corr_pairs = []
        for col in to_drop:
            # Find the feature it correlates with
            correlated_feats = upper.index[upper[col] > 0.90].tolist()
            for feat in correlated_feats:
                val = upper.loc[feat, col]
                high_corr_pairs.append((feat, col, val))

        if not high_corr_pairs:
            print("No collinear pairs found (Correlation > 0.90).")
        else:
            print(f"Found {len(high_corr_pairs)} collinear pairs:")
            # Print top 10
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            for f1, f2, val in high_corr_pairs[:10]:
                print(f"{f1} - {f2}: {val:.4f}")
            if len(high_corr_pairs) > 10:
                print(f"... and {len(high_corr_pairs) - 10} more.")

    # 4.2 Feature Importance (Random Forest)
    print("\n-- Feature Importance (Lightweight Random Forest) --")

    # Preprocessing for RF
    # 1. Impute Missing
    # 2. Encode Categorical

    X_rf = X_sample.copy()

    # Handle Categoricals: Label Encoding
    # Handle Numerics: Simple Mean Imputation

    # Impute Numerics
    if num_cols:
        imp_num = SimpleImputer(strategy="mean")
        X_rf[num_cols] = imp_num.fit_transform(X_rf[num_cols])

    # Process Categoricals
    if cat_cols:
        # Fill missing categoricals with 'Missing'
        X_rf[cat_cols] = X_rf[cat_cols].fillna("Missing")
        le = LabelEncoder()
        for col in cat_cols:
            # Convert to string to handle mixed types safely
            X_rf[col] = le.fit_transform(X_rf[col].astype(str))

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )

    rf.fit(X_rf, y_sample)

    # Get Importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_rf.columns

    print("Top 5 Features by Importance:")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    print("-" * 30)
    print("EDA Complete.")


if __name__ == "__main__":
    main()
