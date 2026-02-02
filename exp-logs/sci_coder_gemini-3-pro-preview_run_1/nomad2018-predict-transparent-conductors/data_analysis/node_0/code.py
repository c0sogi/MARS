import os
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)


def analyze_targets(df, target_cols):
    print_header("Target Variable Analysis")

    for target in target_cols:
        print(f"\nAnalysis for Target: {target}")
        data = df[target]

        # Distribution stats
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()

        print(
            f"  Distribution: Mean={mean_val:.4f}, Std={std_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}"
        )

        # Normality check
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        print(f"  Normality: Skewness={skew:.4f}, Kurtosis={kurt:.4f}")

        if abs(skew) > 1:
            print("  -> The distribution is highly skewed.")
        else:
            print("  -> The distribution is approximately symmetric.")


def analyze_tabular_features(df, feature_cols):
    print_header("Input Data Analysis (Tabular)")

    # Numerical Analysis
    print("\n[Numerical Features]")
    numerical_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if numerical_cols:
        stats_df = df[numerical_cols].describe().T
        stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]

        # Outlier detection (1.5 * IQR rule)
        outliers = {}
        for col in numerical_cols:
            q1 = stats_df.loc[col, "25%"]
            q3 = stats_df.loc[col, "75%"]
            iqr = stats_df.loc[col, "IQR"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[
                0
            ]
            outliers[col] = outlier_count

        print(
            f"{'Feature':<30} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Outliers':<8}"
        )
        print("-" * 95)
        for col in numerical_cols:
            print(
                f"{col:<30} | {stats_df.loc[col, 'mean']:<10.4f} | {stats_df.loc[col, 'std']:<10.4f} | "
                f"{stats_df.loc[col, 'min']:<10.4f} | {stats_df.loc[col, 'max']:<10.4f} | {outliers[col]:<8}"
            )
    else:
        print("No numerical features found.")

    # Categorical Analysis
    print("\n[Categorical Features]")
    # Heuristic: treat 'spacegroup' as categorical even if int
    cat_cols = [
        c
        for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c]) or c == "spacegroup"
    ]

    if cat_cols:
        for col in cat_cols:
            unique_vals = df[col].nunique()
            print(f"  Feature: {col}")
            print(f"    Cardinality: {unique_vals}")
            if unique_vals > 50:
                print("    -> High cardinality (> 50 categories).")

            # Check for rare labels
            counts = df[col].value_counts(normalize=True)
            rare_labels = counts[counts < 0.01]
            if not rare_labels.empty:
                print(f"    -> Found {len(rare_labels)} rare labels (< 1% frequency).")
    else:
        print("No categorical features found.")

    # Missing Values
    print("\n[Missing Values]")
    missing = df[feature_cols].isnull().sum()
    if missing.sum() == 0:
        print("  No missing values detected.")
    else:
        for col in feature_cols:
            if missing[col] > 0:
                pct = (missing[col] / len(df)) * 100
                print(f"  {col}: {missing[col]} missing ({pct:.2f}%)")


def calculate_cell_volume(row):
    """Calculates unit cell volume from lattice lengths and angles."""
    a = row["lattice_vector_1_ang"]
    b = row["lattice_vector_2_ang"]
    c = row["lattice_vector_3_ang"]
    alpha_rad = np.radians(row["lattice_angle_alpha_degree"])
    beta_rad = np.radians(row["lattice_angle_beta_degree"])
    gamma_rad = np.radians(row["lattice_angle_gamma_degree"])

    term = (
        1
        - np.cos(alpha_rad) ** 2
        - np.cos(beta_rad) ** 2
        - np.cos(gamma_rad) ** 2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )
    return (
        a * b * c * np.sqrt(max(0, term))
    )  # max(0, term) to avoid numerical errors near 0


def analyze_feature_relationships(df, feature_cols, target_cols):
    print_header("Feature Relationships")

    # 1. Correlation Analysis
    print("\n[Correlation Analysis]")
    # Include targets in correlation matrix
    analysis_df = df[feature_cols + target_cols].copy()

    # Encode categorical if any (e.g. spacegroup)
    if "spacegroup" in analysis_df.columns:
        le = LabelEncoder()
        analysis_df["spacegroup"] = le.fit_transform(
            analysis_df["spacegroup"].astype(str)
        )

    corr_matrix = analysis_df.corr(method="pearson")

    # Report correlations with targets
    for target in target_cols:
        print(f"\nTop correlations with {target}:")
        target_corr = corr_matrix[target].drop(
            target_cols
        )  # drop targets to see features
        # Sort by absolute value
        sorted_corr = target_corr.abs().sort_values(ascending=False)
        for feat in sorted_corr.index[:5]:
            print(f"  {feat:<30}: {target_corr[feat]:.4f}")

    # Report Collinearity
    print("\n[Collinearity Check (Corr > 0.90)]")
    collinear_pairs = []
    features = [c for c in feature_cols if c in analysis_df.columns]
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c1 = features[i]
            c2 = features[j]
            corr_val = corr_matrix.loc[c1, c2]
            if abs(corr_val) > 0.90:
                collinear_pairs.append((c1, c2, corr_val))

    if collinear_pairs:
        for c1, c2, val in collinear_pairs:
            print(f"  {c1} <--> {c2}: {val:.4f}")
    else:
        print("  No highly collinear feature pairs found.")

    # 2. Feature Importance (Random Forest)
    print("\n[Feature Importance - Random Forest]")

    # Prepare X and y
    X = analysis_df[features].fillna(0)  # Simple imputation for importance check

    for target in target_cols:
        y = analysis_df[target]
        rf = RandomForestRegressor(
            n_estimators=50, random_state=SEED, n_jobs=-1, verbose=0
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=features).sort_values(
            ascending=False
        )
        print(f"\nTop 5 Important Features for {target}:")
        for feat, imp in importances.head(5).items():
            print(f"  {feat:<30}: {imp:.4f}")


def analyze_meta_relationships(df, target_cols):
    print_header("Unstructured (Meta-Feature) Relationships")

    # Calculate derived physical properties
    # 1. Cell Volume
    df["cell_volume"] = df.apply(calculate_cell_volume, axis=1)

    # 2. Atomic Density (Atoms / Volume)
    df["atomic_density"] = df["number_of_total_atoms"] / df["cell_volume"]

    # 3. Average Atomic Mass Proxy (weighted by composition)
    # Approx masses: Al=26.98, Ga=69.72, In=114.82.
    # Note: The percent columns sum to 1.0 for the cation sites.
    # This is a rough proxy for "heaviness" of the material.
    df["avg_cation_mass"] = (
        df["percent_atom_al"] * 26.98
        + df["percent_atom_ga"] * 69.72
        + df["percent_atom_in"] * 114.82
    )

    meta_features = ["cell_volume", "atomic_density", "avg_cation_mass"]

    print("Derived Meta-Features Statistics:")
    print(df[meta_features].describe().T[["mean", "std", "min", "max"]])

    print("\nCorrelations with Targets:")
    corr_matrix = df[meta_features + target_cols].corr()

    for target in target_cols:
        print(f"\nTarget: {target}")
        for feat in meta_features:
            print(f"  {feat:<20}: {corr_matrix.loc[feat, target]:.4f}")


def main():
    # Define paths
    METADATA_PATH = "./metadata/train.csv"

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded training data with {len(df)} samples.")

    # Define Column Groups
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Identify feature columns (exclude IDs, targets, and file paths)
    exclude = ["id", "file_path"] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude]

    # 1. Target Analysis
    analyze_targets(df, target_cols)

    # 2. Input Data Analysis
    analyze_tabular_features(df, feature_cols)

    # 3. Feature Relationships
    analyze_feature_relationships(df, feature_cols, target_cols)

    # 4. Meta-Feature Analysis (Physics-based derived features)
    analyze_meta_relationships(df, target_cols)


if __name__ == "__main__":
    main()
