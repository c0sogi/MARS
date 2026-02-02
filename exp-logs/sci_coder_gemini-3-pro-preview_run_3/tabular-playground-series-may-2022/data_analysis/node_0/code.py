import os
import sys
import numpy as np
import pandas as pd
import warnings
import random
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
TRAIN_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df, target_col):
    print("=" * 30)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    target_data = df[target_col].dropna()
    unique_vals = target_data.nunique()

    # Heuristic: < 20 unique values -> Classification, else Regression
    is_classification = unique_vals < 20

    print(f"Target Column: '{target_col}'")
    print(f"Type: {'Classification' if is_classification else 'Regression'}")

    if is_classification:
        counts = target_data.value_counts()
        ratios = target_data.value_counts(normalize=True)
        print("\nClass Balance:")
        for label, ratio in ratios.items():
            count = counts[label]
            print(f"Class {label}: {count} samples ({ratio:.4f})")
    else:
        # Regression metrics
        t_mean = target_data.mean()
        t_std = target_data.std()
        t_skew = skew(target_data)
        t_kurt = kurtosis(target_data)

        print(f"\nDistribution Stats:")
        print(f"Mean: {t_mean:.4f}")
        print(f"Std Dev: {t_std:.4f}")
        print(f"Skewness: {t_skew:.4f}")
        print(f"Kurtosis: {t_kurt:.4f}")

    return is_classification


def analyze_tabular(df, target_col):
    print("\n" + "=" * 30)
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("=" * 30)

    # Drop non-feature columns for analysis
    cols_to_exclude = [target_col, "id", "source_path"]
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]

    # Separate numerical and categorical
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    categorical_df = df[feature_cols].select_dtypes(exclude=[np.number])

    # 1. Missing Values
    print("MISSING VALUES:")
    missing = df[feature_cols].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_cols = missing[missing > 0]
    if len(missing_cols) == 0:
        print("No missing values found in features.")
    else:
        for col, count in missing_cols.items():
            print(f"{col}: {count} ({missing_pct[col]:.4f}%)")

    # 2. Numerical Analysis
    if not numeric_df.empty:
        print("\nNUMERICAL FEATURES:")
        stats = numeric_df.describe().T

        # Outlier detection (IQR)
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (
            (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        ).sum()

        print(
            f"{'Feature':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers':<10}"
        )
        print("-" * 80)
        for col in numeric_df.columns:
            mean_val = stats.loc[col, "mean"]
            std_val = stats.loc[col, "std"]
            min_val = stats.loc[col, "min"]
            max_val = stats.loc[col, "max"]
            out_count = outliers[col]
            print(
                f"{col:<25} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f} {out_count:<10}"
            )

    # 3. Categorical Analysis
    if not categorical_df.empty:
        print("\nCATEGORICAL FEATURES:")
        print(f"{'Feature':<25} {'Cardinality':<15} {'Notes'}")
        print("-" * 80)
        for col in categorical_df.columns:
            cardinality = categorical_df[col].nunique()
            counts = categorical_df[col].value_counts(normalize=True)
            rare_labels = counts[counts < 0.01]

            notes = []
            if cardinality > 50:
                notes.append(">50 Categories")
            if not rare_labels.empty:
                notes.append(f"Rare Labels: {len(rare_labels)}")

            note_str = ", ".join(notes) if notes else "OK"
            print(f"{col:<25} {cardinality:<15} {note_str}")
    else:
        print("\nCATEGORICAL FEATURES: None detected.")

    return numeric_df, categorical_df


def analyze_relationships(
    df, target_col, is_classification, numeric_df, categorical_df
):
    print("\n" + "=" * 30)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # 1. Correlation (Numerical)
    if not numeric_df.empty:
        print("REDUNDANCY (Collinear Pairs > 0.90):")
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        found_collinear = False
        for col in to_drop:
            # Find the features it correlates with
            correlated_feats = upper.index[upper[col] > 0.90].tolist()
            for feat in correlated_feats:
                print(f"{col} - {feat}: {upper.loc[feat, col]:.4f}")
                found_collinear = True

        if not found_collinear:
            print("No highly collinear pairs found.")

    # 2. Feature Importance (Random Forest)
    print("\nFEATURE IMPORTANCE (Top 5):")

    # Prepare data for RF
    # Downsample for speed if dataset is huge
    MAX_SAMPLES = 50000
    if len(df) > MAX_SAMPLES:
        df_sample = df.sample(n=MAX_SAMPLES, random_state=SEED).copy()
    else:
        df_sample = df.copy()

    y = df_sample[target_col]
    X = df_sample.drop(columns=[target_col, "id", "source_path"], errors="ignore")

    # Simple preprocessing
    # Fill NaNs
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].mean())

    for col in X.select_dtypes(exclude=[np.number]).columns:
        # Fill NaN with mode or 'Missing'
        if X[col].isnull().any():
            X[col] = X[col].fillna(
                X[col].mode()[0] if not X[col].mode().empty else "Missing"
            )
        # Label Encode
        le = LabelEncoder()
        # Convert to string to handle mixed types safely
        X[col] = le.fit_transform(X[col].astype(str))

    # Train Model
    if is_classification:
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
        )
    else:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
        )

    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = 5
    for i in range(min(top_n, X.shape[1])):
        feat_name = X.columns[indices[i]]
        score = importances[indices[i]]
        print(f"{i+1}. {feat_name}: {score:.4f}")


def main():
    set_seed(SEED)

    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    # Load Data
    # Read only the first few rows to check columns if needed, but here we read all
    # since we need stats. 640k rows is manageable in memory.
    try:
        df = pd.read_csv(TRAIN_PATH)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    target_col = "target"
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in dataset.")
        return

    # 1. Target Analysis
    is_classification = analyze_target(df, target_col)

    # 2. Modality Detection & Input Analysis
    # The prompt implies we should detect modality.
    # However, standard CSVs are usually Tabular unless they contain paths to images/audio.
    # We check if there are columns that look like file paths (excluding source_path).

    feature_cols = [c for c in df.columns if c not in [target_col, "id", "source_path"]]
    sample_val = df[feature_cols[0]].iloc[0] if feature_cols else 0

    # Simple heuristic: If features are mostly numeric or short strings, it's tabular.
    # If a feature column contains extensions like .jpg, .wav, it's unstructured reference.
    # Given the dataset description ("normalized continuous data"), it is Tabular.

    numeric_df, categorical_df = analyze_tabular(df, target_col)

    # 3. Feature Relationships
    analyze_relationships(df, target_col, is_classification, numeric_df, categorical_df)


if __name__ == "__main__":
    main()
