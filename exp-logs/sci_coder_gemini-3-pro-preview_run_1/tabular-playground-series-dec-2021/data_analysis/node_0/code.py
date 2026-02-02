import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    # --- Configuration ---
    TRAIN_PATH = "./metadata/train.csv"
    TARGET_COL = "Cover_Type"

    # --- Load Data ---
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    df = pd.read_csv(TRAIN_PATH)

    # Drop Id column if present as it's not a feature
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # --- Modality Detection ---
    # Based on metadata, we expect tabular. We verify by checking for object columns that look like paths.
    # If mostly numeric, treat as tabular.
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(include=["object"]).columns

    # Heuristic: If > 80% columns are numeric, treat as Tabular.
    # The metadata script already confirmed no file paths, so we proceed with Tabular logic.
    is_tabular = len(num_cols) / len(df.columns) > 0.8

    # --- Section 1: Target Variable Analysis ---
    print("SECTION 1: TARGET VARIABLE ANALYSIS")

    if TARGET_COL not in df.columns:
        print(f"Target column '{TARGET_COL}' not found.")
        return

    target_data = df[TARGET_COL]
    n_total = len(df)

    # Determine if Classification or Regression based on target type and unique values
    # Cover_Type is typically classification (integers)
    is_classification = False
    if pd.api.types.is_integer_dtype(target_data) or pd.api.types.is_object_dtype(
        target_data
    ):
        unique_count = target_data.nunique()
        if unique_count < 50:  # Heuristic for classification
            is_classification = True

    if is_classification:
        print(f"Task Type: Classification")
        print(f"Unique Classes: {target_data.nunique()}")

        counts = target_data.value_counts()
        props = target_data.value_counts(normalize=True)

        print("\nClass Distribution:")
        for cls, count in counts.items():
            prop = props[cls]
            print(f"Class {cls}: {count} samples ({prop:.4f})")

        # Imbalance
        max_cls = counts.max()
        min_cls = counts.min()
        ratio = max_cls / min_cls if min_cls > 0 else float("inf")
        print(f"\nClass Imbalance Ratio (Max/Min): {ratio:.4f}")

    else:
        print(f"Task Type: Regression")
        print(f"Mean: {target_data.mean():.4f}")
        print(f"Std:  {target_data.std():.4f}")
        print(f"Skewness: {target_data.skew():.4f}")
        print(f"Kurtosis: {target_data.kurtosis():.4f}")

    # --- Section 2: Input Data Analysis (Tabular) ---
    print("\nSECTION 2: INPUT DATA ANALYSIS (TABULAR)")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    df_features = df[feature_cols]

    # Missing Values
    print("\nMissing Values:")
    missing = df_features.isnull().sum()
    missing_pct = (missing / n_total) * 100
    missing_cols = missing[missing > 0]

    if len(missing_cols) == 0:
        print("No missing values detected.")
    else:
        for col, count in missing_cols.items():
            pct = missing_pct[col]
            print(f"{col}: {count} missing ({pct:.4f}%)")

    # Numerical Analysis
    num_features = df_features.select_dtypes(include=[np.number]).columns
    if len(num_features) > 0:
        print(f"\nNumerical Features ({len(num_features)}):")
        # Compute stats in chunks or efficiently
        stats = df_features[num_features].describe().T

        # Outlier detection (IQR)
        # For large datasets, we can do this vectorized
        Q1 = df_features[num_features].quantile(0.25)
        Q3 = df_features[num_features].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (
            (df_features[num_features] < lower_bound)
            | (df_features[num_features] > upper_bound)
        ).sum()

        # Display top 5 numerical features by outlier count to keep output concise, or summary
        print(
            f"{'Feature':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Outliers':<10}"
        )
        print("-" * 90)
        for col in num_features[:10]:  # Limit to first 10 to avoid huge output
            row = stats.loc[col]
            n_out = outliers[col]
            print(
                f"{col:<30} {row['mean']:<12.4f} {row['std']:<12.4f} {row['min']:<12.4f} {row['max']:<12.4f} {n_out:<10}"
            )
        if len(num_features) > 10:
            print(f"... and {len(num_features) - 10} more numerical features.")

    # Categorical Analysis
    cat_features = df_features.select_dtypes(exclude=[np.number]).columns
    if len(cat_features) > 0:
        print(f"\nCategorical Features ({len(cat_features)}):")
        for col in cat_features:
            unique_vals = df_features[col].nunique()
            print(f"{col}: {unique_vals} categories")

            # Check for rare labels (< 1%)
            val_counts = df_features[col].value_counts(normalize=True)
            rare = val_counts[val_counts < 0.01]
            if not rare.empty:
                print(f"  > Flagged: {len(rare)} rare labels (< 1% freq)")

            if unique_vals > 50:
                print(f"  > Flagged: High cardinality (> 50 categories)")
    else:
        print("\nCategorical Features: None detected.")

    # --- Section 3: Feature Relationships ---
    print("\nSECTION 3: FEATURE RELATIONSHIPS")

    # Correlation (Numerical)
    if len(num_features) > 1:
        print("\nRedundancy Check (Correlation > 0.90):")
        # Calculate correlation matrix
        # If dataset is huge, this is still usually fine for <100 cols.
        # If cols > 1000, might need subsampling. Here cols=54.
        corr_matrix = df_features[num_features].corr().abs()

        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        high_corr_pairs = []
        for col in to_drop:
            # Find the row that correlates
            correlated_rows = upper.index[upper[col] > 0.90].tolist()
            for row in correlated_rows:
                val = upper.loc[row, col]
                high_corr_pairs.append((row, col, val))

        if high_corr_pairs:
            # Sort by correlation strength
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            for f1, f2, val in high_corr_pairs[:10]:  # Show top 10
                print(f"{f1} - {f2}: {val:.4f}")
            if len(high_corr_pairs) > 10:
                print(f"... and {len(high_corr_pairs) - 10} more pairs.")
        else:
            print("No highly collinear pairs found.")

    # Feature Importance (Random Forest)
    print("\nTop 5 Important Features (Random Forest):")

    # Subsample for speed if dataset is large
    MAX_SAMPLES_RF = 100000
    if len(df) > MAX_SAMPLES_RF:
        # Stratified subsample
        if is_classification:
            try:
                # Handle rare classes by grouping small classes or simple random sample if stratify fails
                # Simple random sample is robust enough for EDA importance
                df_sample = df.sample(n=MAX_SAMPLES_RF, random_state=42)
            except:
                df_sample = df.sample(n=MAX_SAMPLES_RF, random_state=42)
        else:
            df_sample = df.sample(n=MAX_SAMPLES_RF, random_state=42)
    else:
        df_sample = df

    X_sample = df_sample[feature_cols]
    y_sample = df_sample[TARGET_COL]

    # Preprocess categorical for RF (Label Encode)
    X_sample_enc = X_sample.copy()
    for col in cat_features:
        le = LabelEncoder()
        # Handle unknown labels in sample by converting to string
        X_sample_enc[col] = le.fit_transform(X_sample_enc[col].astype(str))

    # Fill NaNs for RF
    X_sample_enc = X_sample_enc.fillna(0)

    # Train RF
    if is_classification:
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=42, verbose=0
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=42, verbose=0
        )

    rf.fit(X_sample_enc, y_sample)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(5, len(feature_cols))):
        feat_name = feature_cols[indices[i]]
        imp_val = importances[indices[i]]
        print(f"{i+1}. {feat_name}: {imp_val:.4f}")


if __name__ == "__main__":
    main()
