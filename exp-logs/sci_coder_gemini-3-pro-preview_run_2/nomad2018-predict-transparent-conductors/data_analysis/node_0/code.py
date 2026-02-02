import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
import random

# Constants
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_csv(path)


def analyze_targets(df, target_cols):
    print("=" * 40)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 40)

    for target in target_cols:
        print(f"Target: {target}")
        data = df[target]

        # Distribution stats
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()

        # Normality checks
        skew_val = skew(data)
        kurt_val = kurtosis(data)

        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Skewness: {skew_val:.4f} (Normal ~ 0)")
        print(f"  Kurtosis: {kurt_val:.4f} (Normal ~ 0)")
        print("-" * 20)


def analyze_tabular_inputs(df, feature_cols):
    print("=" * 40)
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("=" * 40)

    numerical_cols = []
    categorical_cols = []

    # Heuristic to distinguish categorical from numerical
    # We treat 'spacegroup' as categorical for cardinality check, but it might be ordinal
    for col in feature_cols:
        if df[col].dtype == "object" or col == "spacegroup":
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    # Numerical Analysis
    if numerical_cols:
        print("Numerical Features:")
        print(
            f"{'Feature':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers (IQR)':<10}"
        )
        print("-" * 85)
        for col in numerical_cols:
            data = df[col]
            mean_v = data.mean()
            std_v = data.std()
            min_v = data.min()
            max_v = data.max()

            # IQR Outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

            print(
                f"{col:<30} {mean_v:<10.4f} {std_v:<10.4f} {min_v:<10.4f} {max_v:<10.4f} {outliers:<10}"
            )
        print()

    # Categorical Analysis
    if categorical_cols:
        print("Categorical Features:")
        for col in categorical_cols:
            cardinality = df[col].nunique()
            print(f"  {col}: {cardinality} unique values")
            if cardinality > 50:
                print(f"    [FLAG] High cardinality (>50)")

            # Check for rare labels
            counts = df[col].value_counts(normalize=True)
            rare_labels = counts[counts < 0.01]
            if not rare_labels.empty:
                print(f"    [FLAG] {len(rare_labels)} rare labels (<1% freq)")
        print()

    # Missing Values
    print("Missing Values:")
    missing = df[feature_cols].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    if missing.sum() == 0:
        print("  No missing values found in features.")
    else:
        for col in feature_cols:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")


def analyze_unstructured_inputs(df):
    print("=" * 40)
    print("INPUT DATA ANALYSIS (GEOMETRY FILES)")
    print("=" * 40)

    # Analyze geometry.xyz files
    # We will check file sizes and basic atom counts if easily parsable

    file_sizes = []
    atom_counts = []

    # Sample a subset to keep runtime low if dataset is huge, but here 2k is fine
    print(f"Analyzing {len(df)} geometry files...")

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            # File size
            file_sizes.append(os.path.getsize(full_path))

            # Atom count (number of lines - 2 header lines usually in XYZ,
            # but let's just count lines for a rough proxy or parse if standard)
            # Standard XYZ: Line 1 = atom count.
            try:
                with open(full_path, "r") as f:
                    first_line = f.readline()
                    if first_line.strip().isdigit():
                        atom_counts.append(int(first_line.strip()))
            except:
                pass

    if file_sizes:
        print(f"Geometry File Size (bytes):")
        print(f"  Mean: {np.mean(file_sizes):.2f}")
        print(f"  Std:  {np.std(file_sizes):.2f}")
        print(f"  Min:  {np.min(file_sizes)}")
        print(f"  Max:  {np.max(file_sizes)}")

    if atom_counts:
        print(f"Atom Counts (from .xyz header):")
        print(f"  Mean: {np.mean(atom_counts):.2f}")
        print(f"  Min:  {np.min(atom_counts)}")
        print(f"  Max:  {np.max(atom_counts)}")

        # Check consistency with tabular data if available
        if "number_of_total_atoms" in df.columns:
            tabular_counts = df["number_of_total_atoms"]
            diff = np.abs(
                np.array(atom_counts) - tabular_counts.values[: len(atom_counts)]
            )
            if np.sum(diff) == 0:
                print("  [OK] Atom counts match 'number_of_total_atoms' in CSV.")
            else:
                print(
                    f"  [WARN] Atom counts mismatch for {np.count_nonzero(diff)} records."
                )


def analyze_feature_relationships(df, feature_cols, target_cols):
    print("=" * 40)
    print("FEATURE RELATIONSHIPS")
    print("=" * 40)

    # Prepare data for correlation and importance
    # Handle categorical for correlation (Label Encoding)
    df_encoded = df.copy()
    for col in feature_cols:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # 1. Correlation
    print("Top Correlations with Targets (Pearson):")
    corr_matrix = df_encoded[feature_cols + target_cols].corr(method="pearson")

    for target in target_cols:
        print(f"  Target: {target}")
        target_corrs = corr_matrix[target].drop(target_cols)  # Drop targets themselves
        # Sort by absolute value
        top_corrs = target_corrs.abs().sort_values(ascending=False).head(5)
        for feat, val in top_corrs.items():
            sign = "+" if target_corrs[feat] > 0 else "-"
            print(f"    {feat:<30}: {sign}{val:.4f}")
    print()

    # 2. Feature Importance (Random Forest)
    print("Feature Importance (Random Forest):")
    X = df_encoded[feature_cols].fillna(0)  # Simple impute for importance check

    for target in target_cols:
        y = df_encoded[target]
        rf = RandomForestRegressor(
            n_estimators=50, random_state=SEED, n_jobs=-1, max_depth=10
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"  Target: {target}")
        for i in range(min(5, len(feature_cols))):
            print(f"    {feature_cols[indices[i]]:<30}: {importances[indices[i]]:.4f}")
    print()

    # 3. Redundancy (Collinearity)
    print("Feature Redundancy (Correlation > 0.90):")
    high_corr_pairs = []
    feat_corr = df_encoded[feature_cols].corr().abs()
    # Iterate over upper triangle
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if feat_corr.iloc[i, j] > 0.90:
                high_corr_pairs.append(
                    (feature_cols[i], feature_cols[j], feat_corr.iloc[i, j])
                )

    if high_corr_pairs:
        for f1, f2, val in high_corr_pairs:
            print(f"  {f1} <-> {f2}: {val:.4f}")
    else:
        print("  No highly collinear features found.")

    # 4. Meta-Feature Relationship
    # Correlation between file size and target
    # We need to re-calculate file sizes aligned with the dataframe
    file_sizes = []
    valid_indices = []
    for idx, row in df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
            valid_indices.append(idx)

    if file_sizes:
        print("\nMeta-Feature Relationships:")
        # Create a temporary series
        size_series = pd.Series(file_sizes, index=valid_indices)
        for target in target_cols:
            # Align target with valid file sizes
            target_subset = df.loc[valid_indices, target]
            corr = size_series.corr(target_subset)
            print(f"  Correlation (File Size vs {target}): {corr:.4f}")


def main():
    set_seed(SEED)

    # Load Data
    print("Loading data...")
    try:
        df = load_data(METADATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define Columns
    # Exclude IDs, file paths, and targets from features
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    exclude_cols = ["id", "file_path"] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 1. Target Analysis
    analyze_targets(df, target_cols)

    # 2. Input Data Analysis (Tabular)
    analyze_tabular_inputs(df, feature_cols)

    # 3. Input Data Analysis (Unstructured/Geometry)
    analyze_unstructured_inputs(df)

    # 4. Feature Relationships
    analyze_feature_relationships(df, feature_cols, target_cols)


if __name__ == "__main__":
    main()
