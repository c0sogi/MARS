import pandas as pd
import numpy as np
import os
import random
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Configuration & Seeding
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def main():
    # ==========================================
    # 1. Data Loading & Integrity
    # ==========================================
    # Load Metadata to identify training breath_ids
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: Metadata file not found at {TRAIN_META_PATH}")
        return

    df_meta = pd.read_csv(TRAIN_META_PATH)
    train_breath_ids = set(df_meta["breath_id"].unique())

    # Load Raw Data
    # Optimization: Read only necessary columns if dataset is massive,
    # but 5M rows fits in memory (approx 1GB).
    df_raw = pd.read_csv(TRAIN_DATA_PATH)

    # Filter for Training Set only
    df_train = df_raw[df_raw["breath_id"].isin(train_breath_ids)].copy()

    # Verify Split
    # Check if any breath_id in df_train is NOT in train_breath_ids (Should be 0)
    # This is implicitly handled by the filter, but conceptually important.

    print("DATA INTEGRITY CHECK")
    print(f"Raw rows: {len(df_raw)}")
    print(f"Training rows (filtered): {len(df_train)}")
    print("-" * 30)

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    target_col = "pressure"
    y = df_train[target_col]

    print("\nTARGET VARIABLE ANALYSIS")

    # Distribution Stats
    mean_val = y.mean()
    std_val = y.std()
    min_val = y.min()
    max_val = y.max()

    print(f"Target: {target_col}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")

    # Normality Check (Regression)
    skewness = skew(y)
    kurt = kurtosis(y)

    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurt:.4f}")
    if abs(skewness) > 1:
        print("Note: Target distribution is highly skewed.")
    else:
        print("Note: Target distribution is approximately symmetric.")

    # ==========================================
    # 3. Input Data Analysis (Tabular)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TABULAR)")

    # Identify Column Types
    # Based on dataset description:
    # Numerical: time_step, u_in, pressure (target), R, C (physical constants)
    # Categorical/Binary: u_out
    # ID columns: id, breath_id (excluded from feature stats usually)

    feature_cols = [
        c for c in df_train.columns if c not in ["id", "breath_id", "pressure"]
    ]

    # Heuristic for categorical vs numerical
    # u_out is binary (0, 1). R and C have few unique values but are numeric.
    # We will treat R and C as numerical for stats, but note cardinality.

    numerical_cols = ["time_step", "u_in", "R", "C"]
    categorical_cols = ["u_out"]  # Binary

    # --- Numerical Analysis ---
    print("--- Numerical Features ---")
    for col in numerical_cols:
        series = df_train[col]
        mu = series.mean()
        sigma = series.std()
        mn = series.min()
        mx = series.max()

        # Outlier detection (IQR method)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)

        print(f"Feature: {col}")
        print(f"  Mean: {mu:.4f}, Std: {sigma:.4f}, Min: {mn:.4f}, Max: {mx:.4f}")
        print(
            f"  Outliers (IQR): {outlier_count} ({outlier_count/len(series)*100:.2f}%)"
        )

    # --- Categorical Analysis ---
    print("\n--- Categorical Features ---")
    # Also checking R and C for cardinality as they are discrete settings
    potential_cats = categorical_cols + ["R", "C"]

    for col in potential_cats:
        unique_vals = df_train[col].unique()
        cardinality = len(unique_vals)
        print(f"Feature: {col}")
        print(f"  Cardinality: {cardinality}")

        # Check for rare labels
        val_counts = df_train[col].value_counts(normalize=True)
        rare_labels = val_counts[val_counts < 0.01].index.tolist()
        if rare_labels:
            print(f"  Rare labels (<1%): {rare_labels}")
        else:
            print(f"  Rare labels (<1%): None")

    # --- Missing Values ---
    print("\n--- Missing Values ---")
    missing = df_train[feature_cols].isnull().sum()
    missing_pct = (missing / len(df_train)) * 100

    if missing.sum() == 0:
        print("No missing values found in features.")
    else:
        for col in feature_cols:
            if missing[col] > 0:
                print(f"{col}: {missing[col]} missing ({missing_pct[col]:.4f}%)")

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # --- Correlation ---
    # Using Pearson for all numerical-like columns including u_out, R, C
    corr_cols = numerical_cols + categorical_cols + [target_col]
    corr_matrix = df_train[corr_cols].corr(method="pearson")

    print("--- Top Correlations with Target (Pressure) ---")
    target_corr = (
        corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    )
    for feat, val in target_corr.items():
        print(f"{feat}: {val:.4f}")

    # --- Redundancy (Collinearity) ---
    print("\n--- Feature Redundancy (Corr > 0.90) ---")
    redundant_pairs = []
    features_to_check = numerical_cols + categorical_cols
    for i in range(len(features_to_check)):
        for j in range(i + 1, len(features_to_check)):
            f1 = features_to_check[i]
            f2 = features_to_check[j]
            c = corr_matrix.loc[f1, f2]
            if abs(c) > 0.90:
                redundant_pairs.append((f1, f2, c))

    if not redundant_pairs:
        print("No collinear pairs found.")
    else:
        for f1, f2, c in redundant_pairs:
            print(f"{f1} - {f2}: {c:.4f}")

    # --- Feature Importance (Random Forest) ---
    print("\n--- Feature Importance (Random Forest) ---")
    # Sample data for speed (100k rows)
    sample_size = min(100000, len(df_train))
    df_sample = df_train.sample(n=sample_size, random_state=SEED)

    X_sample = df_sample[numerical_cols + categorical_cols]
    y_sample = df_sample[target_col]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED, verbose=0
    )
    rf.fit(X_sample, y_sample)

    importances = pd.Series(rf.feature_importances_, index=X_sample.columns)
    top_5 = importances.sort_values(ascending=False).head(5)

    for feat, imp in top_5.items():
        print(f"{feat}: {imp:.4f}")

    # --- Unstructured / Meta-Feature Relationships ---
    print("\n--- Meta-Feature Analysis (Breath Level) ---")
    # Aggregating by breath_id to see if breath-level properties correlate with pressure
    # e.g., Does a specific R/C setting lead to higher average pressure?

    breath_groups = df_train.groupby("breath_id")
    breath_meta = breath_groups.agg(
        {
            "pressure": "mean",
            "u_in": "mean",
            "R": "first",
            "C": "first",
            "time_step": "count",  # breath duration in steps
        }
    ).rename(columns={"time_step": "breath_len"})

    # Correlation of meta-features with mean pressure
    meta_corr = breath_meta.corr()["pressure"].drop("pressure")

    print("Correlation of Breath-Level Aggregates with Mean Pressure:")
    for feat, val in meta_corr.items():
        print(f"Mean/First {feat}: {val:.4f}")


if __name__ == "__main__":
    main()
