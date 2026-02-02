import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    # Load Train Metadata
    train_path = "./metadata/train_metadata.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        sys.exit(1)

    # Load specific columns to save memory
    df_train = pd.read_csv(train_path)

    # Load Structures
    struct_path = "./input/structures.csv"
    if not os.path.exists(struct_path):
        print(f"Error: {struct_path} not found.")
        sys.exit(1)

    df_struct = pd.read_csv(struct_path)

    return df_train, df_struct


def engineer_distance(df_train, df_struct):
    # Merge structure info for atom_index_0
    # Rename columns in struct to avoid collision
    df_struct_0 = df_struct.rename(
        columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
    )
    df_struct_1 = df_struct.rename(
        columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
    )

    # We need to merge on molecule_name and atom_index
    # Optimization: Filter structures to only those in train if needed, but train is large so just merge

    df_merged = df_train.merge(
        df_struct_0[["molecule_name", "atom_index_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )

    df_merged = df_merged.merge(
        df_struct_1[["molecule_name", "atom_index_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    # Calculate Euclidean Distance
    df_merged["dist_x"] = df_merged["x0"] - df_merged["x1"]
    df_merged["dist_y"] = df_merged["y0"] - df_merged["y1"]
    df_merged["dist_z"] = df_merged["z0"] - df_merged["z1"]
    df_merged["distance"] = np.sqrt(
        df_merged["dist_x"] ** 2 + df_merged["dist_y"] ** 2 + df_merged["dist_z"] ** 2
    )

    return df_merged


def analyze_target(df):
    target = df["scalar_coupling_constant"]

    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Distribution stats
    mean_val = target.mean()
    std_val = target.std()
    min_val = target.min()
    max_val = target.max()

    print(f"Mean: {mean_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"Max: {max_val:.4f}")

    # Normality
    skew = stats.skew(target)
    kurt = stats.kurtosis(target)
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")

    # Quantiles
    print(f"25th Percentile: {target.quantile(0.25):.4f}")
    print(f"50th Percentile (Median): {target.median():.4f}")
    print(f"75th Percentile: {target.quantile(0.75):.4f}")
    print("")


def analyze_tabular_inputs(df):
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("-" * 30)

    # Numerical Columns
    num_cols = ["atom_index_0", "atom_index_1"]
    print("Numerical Features:")
    for col in num_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        print(f"  Column: {col}")
        print(f"    Mean: {series.mean():.4f}, Std: {series.std():.4f}")
        print(f"    Min: {series.min():.4f}, Max: {series.max():.4f}")
        print(f"    Outliers (IQR method): {outliers}")

    # Categorical Columns
    cat_cols = ["type"]  # molecule_name is high cardinality ID
    print("\nCategorical Features:")
    for col in cat_cols:
        series = df[col]
        n_unique = series.nunique()
        print(f"  Column: {col}")
        print(f"    Cardinality: {n_unique}")

        if n_unique <= 50:
            counts = series.value_counts(normalize=True)
            print(f"    Distribution: {counts.to_dict()}")
        else:
            print("    Distribution: > 50 categories, skipping detail.")

        # Rare labels
        counts = series.value_counts(normalize=True)
        rare = counts[counts < 0.01]
        if not rare.empty:
            print(f"    Rare Labels (<1%): {list(rare.index)}")
        else:
            print("    Rare Labels (<1%): None")

    # Missing Values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.4f}%)")
    if missing.sum() == 0:
        print("  No missing values found.")
    print("")


def analyze_meta_features(df):
    print("UNSTRUCTURED (META-FEATURE) RELATIONSHIPS")
    print("-" * 30)
    print("Derived Geometric Feature: Inter-atomic Distance (Angstroms)")

    dist = df["distance"]
    print(f"  Mean Distance: {dist.mean():.4f}")
    print(f"  Std Distance: {dist.std():.4f}")
    print(f"  Min Distance: {dist.min():.4f}")
    print(f"  Max Distance: {dist.max():.4f}")

    # Correlation with Target
    corr_pearson = df["distance"].corr(df["scalar_coupling_constant"], method="pearson")
    corr_spearman = df["distance"].corr(
        df["scalar_coupling_constant"], method="spearman"
    )

    print(f"  Correlation with Target (Pearson): {corr_pearson:.4f}")
    print(f"  Correlation with Target (Spearman): {corr_spearman:.4f}")
    print("")


def analyze_relationships(df):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Prepare data for relationships
    # Encode type
    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    features = ["atom_index_0", "atom_index_1", "distance", "type_enc"]
    target = "scalar_coupling_constant"

    # Correlation Matrix
    print("Correlation Matrix (Pearson):")
    corr_mat = df[features + [target]].corr()
    print(corr_mat.round(4))

    # Redundancy
    print("\nRedundant Features (Correlation > 0.90):")
    redundant_found = False
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_mat.iloc[i, j]
            if abs(c) > 0.90:
                print(f"  {features[i]} - {features[j]}: {c:.4f}")
                redundant_found = True
    if not redundant_found:
        print("  None")

    # Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest):")
    # Sample data for speed if necessary, but 3M is okay for a shallow tree
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    X = df_sample[features]
    y = df_sample[target]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42, verbose=0
    )
    rf.fit(X, y)

    importances = pd.DataFrame(
        {"feature": features, "importance": rf.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    for idx, row in importances.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


def main():
    set_seed(42)

    # Load Data
    df_train, df_struct = load_data()

    # Engineer Features (Distance)
    # This transforms the raw tabular data into physically meaningful data
    df_eda = engineer_distance(df_train, df_struct)

    # 1. Target Analysis
    analyze_target(df_eda)

    # 2. Input Data Analysis (Tabular)
    analyze_tabular_inputs(df_eda)

    # 3. Unstructured/Meta Analysis (Geometric)
    analyze_meta_features(df_eda)

    # 4. Relationships
    analyze_relationships(df_eda)


if __name__ == "__main__":
    main()
