import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Constants & Configuration ---
METADATA_PATH = "./metadata/train.parquet"
TARGET_COL = "Cover_Type"
SEED = 42
SAMPLE_SIZE_RF = 100000  # Downsample for RF feature importance to save time


# --- Seeding ---
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()

# --- Helper Functions ---


def detect_modality(df):
    """
    Heuristic to detect data modality.
    Prioritizes: Image/Audio (paths) > Text (long strings) > Tabular.
    """
    # Check for file paths (Image/Audio)
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(object_cols) > 0:
        sample = (
            df[object_cols[0]].dropna().astype(str).iloc[0]
            if not df[object_cols[0]].dropna().empty
            else ""
        )
        if sample.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")):
            return "Image"
        if sample.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            return "Audio"

        # Check for Text (high average length)
        mean_len = df[object_cols[0]].dropna().astype(str).str.len().mean()
        if mean_len > 50:  # Arbitrary threshold for "Text" vs "Categorical"
            return "Text"

    return "Tabular"


def analyze_target(df, target_col):
    print("=" * 30)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return None

    y = df[target_col]

    # Determine Task Type
    is_numeric = pd.api.types.is_numeric_dtype(y)
    num_unique = y.nunique()

    # Heuristic: If numeric and < 20 unique values, treat as Classification
    # Or if object type, treat as Classification
    if (is_numeric and num_unique < 20) or not is_numeric:
        task_type = "Classification"
        print(f"Task Type: Classification")
        print(f"Unique Classes: {num_unique}")

        counts = y.value_counts()
        ratios = y.value_counts(normalize=True)

        print("\nClass Balance:")
        for cls, count in counts.items():
            ratio = ratios[cls]
            print(f"Class {cls}: {count} samples ({ratio:.4f})")

        # Check for imbalance
        min_cls = ratios.min()
        max_cls = ratios.max()
        print(f"\nImbalance Ratio (Max/Min): {(max_cls/min_cls):.4f}")

    else:
        task_type = "Regression"
        print(f"Task Type: Regression")

        print(f"\nDistribution Stats:")
        print(f"Mean: {y.mean():.4f}")
        print(f"Std:  {y.std():.4f}")
        print(f"Min:  {y.min():.4f}")
        print(f"Max:  {y.max():.4f}")

        print(f"\nNormality Check:")
        print(f"Skewness: {y.skew():.4f}")
        print(f"Kurtosis: {y.kurtosis():.4f}")

    return task_type


def analyze_tabular_inputs(df, target_col):
    print("\n" + "=" * 30)
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("=" * 30)

    feature_cols = [c for c in df.columns if c != target_col and c != "Id"]
    df_feats = df[feature_cols]

    # Numerical Analysis
    num_cols = df_feats.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"\n--- Numerical Features ({len(num_cols)}) ---")
        stats = df_feats[num_cols].describe().T

        # Outlier Detection (IQR)
        Q1 = df_feats[num_cols].quantile(0.25)
        Q3 = df_feats[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (
            (df_feats[num_cols] < (Q1 - 1.5 * IQR))
            | (df_feats[num_cols] > (Q3 + 1.5 * IQR))
        ).sum()

        # Combine info
        stats["outliers"] = outliers
        stats["outlier_pct"] = (outliers / len(df)) * 100

        # Display top 10 by outlier count to keep output concise
        print("Top 10 features by outlier count:")
        print(
            stats[["mean", "std", "min", "max", "outliers", "outlier_pct"]]
            .sort_values("outliers", ascending=False)
            .head(10)
            .to_string(float_format="{:.4f}".format)
        )

    # Categorical Analysis
    cat_cols = df_feats.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        print(f"\n--- Categorical Features ({len(cat_cols)}) ---")
        for col in cat_cols:
            n_unique = df_feats[col].nunique()
            print(f"Feature '{col}': {n_unique} unique values")

            if n_unique > 50:
                print(f"  -> High Cardinality (>50 categories)")

            # Rare labels
            counts = df_feats[col].value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(f"  -> Rare labels (<1% freq): {len(rare)} categories")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df_feats.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False).to_string())


def analyze_relationships(df, target_col, task_type):
    print("\n" + "=" * 30)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    feature_cols = [c for c in df.columns if c != target_col and c != "Id"]

    # 1. Correlation (Numerical)
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        print("\n--- Redundancy (Collinearity > 0.90) ---")
        # Compute correlation matrix
        corr_matrix = df[num_cols].corr().abs()

        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        collinear_pairs = []
        for col in to_drop:
            # Find the feature it correlates with
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            for cw in correlated_with:
                collinear_pairs.append((cw, col, upper.loc[cw, col]))

        if not collinear_pairs:
            print("No highly collinear pairs found.")
        else:
            for f1, f2, val in collinear_pairs[:20]:  # Limit output
                print(f"{f1} <-> {f2}: {val:.4f}")
            if len(collinear_pairs) > 20:
                print(f"... and {len(collinear_pairs) - 20} more.")

    # 2. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")

    # Prepare data
    # Drop NaNs for RF simplicity or fill with median
    df_rf = df[feature_cols + [target_col]].dropna()

    if df_rf.empty:
        print("Cannot run RF: Dataset empty after dropping NaNs.")
        return

    # Encode categoricals
    cat_cols = df_rf.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_rf[col] = le.fit_transform(df_rf[col].astype(str))

    # Downsample if too large
    if len(df_rf) > SAMPLE_SIZE_RF:
        print(
            f"Downsampling to {SAMPLE_SIZE_RF} rows for Feature Importance analysis..."
        )
        if task_type == "Classification":
            # Stratified sample
            try:
                _, df_sample = train_test_split(
                    df_rf,
                    test_size=SAMPLE_SIZE_RF,
                    stratify=df_rf[target_col],
                    random_state=SEED,
                )
            except ValueError:
                # Fallback if stratification fails (e.g. rare classes)
                df_sample = df_rf.sample(n=SAMPLE_SIZE_RF, random_state=SEED)
        else:
            df_sample = df_rf.sample(n=SAMPLE_SIZE_RF, random_state=SEED)
    else:
        df_sample = df_rf

    X = df_sample[feature_cols]
    y = df_sample[target_col]

    if task_type == "Classification":
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED, verbose=0
        )
    else:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED, verbose=0
        )

    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("Top 5 Important Features:")
    print(
        importances.sort_values(ascending=False)
        .head(5)
        .to_string(float_format="{:.4f}".format)
    )


# --- Main Execution ---


def main():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    print(f"Loading data from {METADATA_PATH}...")
    df = pd.read_parquet(METADATA_PATH)
    print(f"Data Loaded. Shape: {df.shape}")

    # 1. Detect Modality
    modality = detect_modality(df)
    print(f"Detected Modality: {modality}")

    # 2. Target Analysis
    task_type = analyze_target(df, TARGET_COL)

    if task_type is None:
        print("Skipping further analysis due to missing target.")
        return

    # 3. Input Analysis
    if modality == "Tabular":
        analyze_tabular_inputs(df, TARGET_COL)
    else:
        # Placeholder for other modalities if they were detected
        # (Based on provided metadata, this path is unlikely, but required by prompt)
        print(
            f"Detailed analysis for {modality} modality is not implemented in this tabular-focused script."
        )
        # Fallback to tabular analysis of metadata features
        analyze_tabular_inputs(df, TARGET_COL)

    # 4. Relationships
    analyze_relationships(df, TARGET_COL, task_type)

    print("\nEDA Completed.")


if __name__ == "__main__":
    main()
