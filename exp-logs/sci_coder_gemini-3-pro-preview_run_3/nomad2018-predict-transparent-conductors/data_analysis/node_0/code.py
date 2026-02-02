import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ase.io

# --- Constants & Configuration ---
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title.upper()}")
    print(f"{'='*60}")


def analyze_targets(df, target_cols):
    print_section("Target Variable Analysis")

    for col in target_cols:
        print(f"\n--- Target: {col} ---")
        series = df[col]

        # Distribution stats
        desc = series.describe()
        print(f"Count: {desc['count']:.0f}")
        print(f"Mean:  {desc['mean']:.4f}")
        print(f"Std:   {desc['std']:.4f}")
        print(f"Min:   {desc['min']:.4f}")
        print(f"Max:   {desc['max']:.4f}")

        # Normality
        skew = series.skew()
        kurt = series.kurtosis()
        print(
            f"Skewness: {skew:.4f} ({'Highly Skewed' if abs(skew) > 1 else 'Moderately Skewed' if abs(skew) > 0.5 else 'Approx. Symmetric'})"
        )
        print(f"Kurtosis: {kurt:.4f}")

        # Normality Test (Shapiro-Wilk for small sample, D'Agostino's K^2 for larger)
        if len(series) >= 3:
            stat, p = stats.normaltest(series)
            print(
                f"Normality Test (D'Agostino K^2): p-value={p:.4e} -> {'Likely Normal' if p > 0.05 else 'Not Normal'}"
            )


def analyze_tabular(df, numerical_cols, categorical_cols):
    print_section("Tabular Input Data Analysis")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found in tabular data.")
    else:
        for col, count in missing.items():
            print(f"{col}: {count} missing ({count/len(df)*100:.2f}%)")

    # Numerical Features
    if numerical_cols:
        print("\n--- Numerical Features ---")
        stats_df = df[numerical_cols].describe().T
        stats_df["IQR"] = df[numerical_cols].quantile(0.75) - df[
            numerical_cols
        ].quantile(0.25)

        # Outlier detection (1.5 * IQR)
        outlier_counts = {}
        for col in numerical_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)

        print(
            f"{'Feature':<30} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Outliers':<8}"
        )
        print("-" * 95)
        for col in numerical_cols:
            row = stats_df.loc[col]
            print(
                f"{col:<30} | {row['mean']:<10.4f} | {row['std']:<10.4f} | {row['min']:<10.4f} | {row['max']:<10.4f} | {outlier_counts[col]:<8}"
            )

    # Categorical Features
    if categorical_cols:
        print("\n--- Categorical Features ---")
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            print(f"\nFeature: {col}")
            print(f"Cardinality: {unique_vals}")

            # Check for rare labels (< 1%)
            value_counts = df[col].value_counts(normalize=True)
            rare = value_counts[value_counts < 0.01]
            if not rare.empty:
                print(
                    f"Rare labels count: {len(rare)} (Total freq: {rare.sum()*100:.2f}%)"
                )
                print(f"Top 5 most common: \n{value_counts.head(5).to_string()}")
            else:
                print("No rare labels (< 1%) detected.")


def extract_geometry_features(df):
    """
    Reads .xyz files using ASE and extracts physical properties.
    """
    print_section("Geometry Data Analysis (XYZ Files)")

    volumes = []
    densities = []
    num_atoms_list = []

    # Limit processing if dataset is huge, but 1700 is manageable
    print(f"Processing geometry files for {len(df)} samples...")

    valid_indices = []

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        try:
            # ASE read
            atoms = ase.io.read(full_path)

            # Volume (Angstrom^3)
            vol = atoms.get_volume()

            # Number of atoms
            n_atoms = len(atoms)

            # Density (Atomic Mass Units / Angstrom^3 is proportional to g/cm^3)
            # Just using sum of masses / volume
            mass = sum(atoms.get_masses())
            density = mass / vol if vol > 0 else 0

            volumes.append(vol)
            densities.append(density)
            num_atoms_list.append(n_atoms)
            valid_indices.append(idx)

        except Exception as e:
            # In case of read failure, we skip or fill NaN.
            # For EDA, we'll just skip and report if many fail.
            pass

    if not valid_indices:
        print("Failed to read any geometry files.")
        return df

    # Create a temporary dataframe for these features
    geo_df = pd.DataFrame(
        {
            "geo_volume": volumes,
            "geo_density": densities,
            "geo_num_atoms": num_atoms_list,
        },
        index=valid_indices,
    )

    # Merge back
    df_merged = df.join(geo_df)

    # Analyze the new features
    print("\n--- Extracted Geometry Features ---")
    print(df_merged[["geo_volume", "geo_density", "geo_num_atoms"]].describe())

    return df_merged


def analyze_relationships(df, target_cols, num_cols, cat_cols):
    print_section("Feature Relationships")

    # 1. Correlations
    print("\n--- Top Correlations with Targets (Pearson) ---")
    # Select only numeric columns including extracted geometry ones
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    for target in target_cols:
        print(f"\nTarget: {target}")
        corrs = (
            corr_matrix[target]
            .drop(target_cols, errors="ignore")
            .sort_values(ascending=False, key=abs)
        )
        print(corrs.head(5))

    # 2. Redundancy (Collinearity)
    print("\n--- High Feature Collinearity (> 0.90) ---")
    # Exclude targets from this check
    feature_corr = numeric_df.drop(columns=target_cols, errors="ignore").corr().abs()
    # Upper triangle
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    high_corr_pairs = [
        (column, index, upper.loc[index, column])
        for index in upper.index
        for column in upper.columns
        if upper.loc[index, column] > 0.90
    ]

    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}")
    else:
        print("No highly collinear feature pairs found.")

    # 3. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")

    # Prepare data
    # Handle NaNs if any (simple fill for EDA purposes)
    X = numeric_df.drop(columns=target_cols, errors="ignore").fillna(0)

    # Encode categorical if low cardinality and not already in numeric_df
    # For this dataset, 'spacegroup' is numeric (int), so it's likely handled.
    # If there were string categoricals, we'd encode them here.

    for target in target_cols:
        y = df[target]

        # Train simple RF
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )

        print(f"\nTop 5 Features for {target}:")
        print(importances.head(5))


def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded training metadata with shape: {df.shape}")

    # Define Column Groups based on dataset description
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Identify numerical columns based on types
    # Exclude targets and file_path and id
    exclude_cols = target_cols + ["id", "file_path"]
    input_cols = [c for c in df.columns if c not in exclude_cols]

    numerical_cols = []
    categorical_cols = []

    for col in input_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Heuristic: if integer and low cardinality, treat as categorical?
            # Spacegroup is technically categorical (nominal), but often encoded as int.
            # Let's treat spacegroup as categorical for analysis purposes if cardinality is low relative to data size,
            # but here spacegroups can be many. Let's treat it as categorical.
            if col == "spacegroup":
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    # 2. Target Analysis
    analyze_targets(df, target_cols)

    # 3. Tabular Input Analysis
    analyze_tabular(df, numerical_cols, categorical_cols)

    # 4. Geometry Feature Extraction & Analysis
    # This adds 'geo_volume', 'geo_density', etc. to df
    df = extract_geometry_features(df)

    # Update numerical columns list with new features
    new_geo_cols = ["geo_volume", "geo_density", "geo_num_atoms"]
    numerical_cols.extend([c for c in new_geo_cols if c in df.columns])

    # 5. Relationships
    analyze_relationships(df, target_cols, numerical_cols, categorical_cols)


if __name__ == "__main__":
    main()
