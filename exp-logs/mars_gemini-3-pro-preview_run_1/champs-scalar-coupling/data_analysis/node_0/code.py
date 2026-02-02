import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
METADATA_TRAIN = "./metadata/train.csv"
INPUT_STRUCTURES = "./input/structures.csv"
INPUT_POTENTIAL = "./input/potential_energy.csv"
INPUT_DIPOLE = "./input/dipole_moments.csv"


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    """Loads training metadata and structures."""
    # Load Train Data
    df_train = pd.read_csv(METADATA_TRAIN)

    # Load Structures
    # structures.csv: molecule_name, atom_index, atom, x, y, z
    df_structures = pd.read_csv(INPUT_STRUCTURES)

    return df_train, df_structures


def analyze_target(df):
    """Analyzes the target variable distribution."""
    target = df["scalar_coupling_constant"]

    print("DATA INTEGRITY")
    print(f"Analysis performed on Training Set with shape: {df.shape}")
    print("-" * 30)

    print("TARGET VARIABLE ANALYSIS")
    print(f"Target Name: scalar_coupling_constant")
    print(f"Type: Regression")

    # Global Stats
    mean_val = target.mean()
    std_val = target.std()
    min_val = target.min()
    max_val = target.max()
    skew_val = skew(target)
    kurt_val = kurtosis(target)

    print(f"Mean: {mean_val:.4f}")
    print(f"Std:  {std_val:.4f}")
    print(f"Min:  {min_val:.4f}")
    print(f"Max:  {max_val:.4f}")
    print(f"Skewness: {skew_val:.4f} (Normal=0)")
    print(f"Kurtosis: {kurt_val:.4f} (Normal=3)")

    # Analysis by Type
    print("\nTarget Statistics by Coupling Type:")
    type_stats = df.groupby("type")["scalar_coupling_constant"].agg(
        ["mean", "std", "count"]
    )
    print(type_stats.to_string(float_format="{:.4f}".format))
    print("-" * 30)


def analyze_tabular_inputs(df):
    """Analyzes tabular features."""
    print("INPUT DATA ANALYSIS (TABULAR)")

    # Numerical Columns (excluding target and id)
    num_cols = ["atom_index_0", "atom_index_1"]
    print("\nNumerical Features:")
    for col in num_cols:
        col_data = df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        outliers = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()

        print(f"Feature: {col}")
        print(f"  Mean: {col_data.mean():.4f}, Std: {col_data.std():.4f}")
        print(f"  Min: {col_data.min():.4f}, Max: {col_data.max():.4f}")
        print(f"  Outliers (IQR method): {outliers} ({outliers/len(df)*100:.2f}%)")

    # Categorical Columns
    cat_cols = ["type", "molecule_name"]
    print("\nCategorical Features:")
    for col in cat_cols:
        unique_count = df[col].nunique()
        print(f"Feature: {col}")
        print(f"  Cardinality: {unique_count}")
        if unique_count > 50:
            print(f"  High Cardinality Flag: Yes (>50 categories)")

        # Check for rare labels (only for low cardinality to avoid massive computation on molecule_name)
        if unique_count < 100:
            counts = df[col].value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(f"  Rare Labels (<1%): {list(rare.index)}")
            else:
                print(f"  Rare Labels (<1%): None")

    # Missing Values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found.")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    print("-" * 30)


def engineer_features_and_analyze_relationships(df_train, df_structures):
    """Merges structure data to calculate distances and analyzes relationships."""
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Merge Structure Data
    # We need coordinates for atom_0 and atom_1
    # Optimization: Drop unnecessary columns from structures for merge
    structures_map = df_structures[
        ["molecule_name", "atom_index", "x", "y", "z", "atom"]
    ]

    # Merge for Atom 0
    df_merged = df_train.merge(
        structures_map,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    ).rename(columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom_0"})

    # Merge for Atom 1
    df_merged = df_merged.merge(
        structures_map,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
        suffixes=("_0", "_1"),
    ).rename(columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom_1"})

    # 2. Calculate Distance
    df_merged["dist_x"] = (df_merged["x0"] - df_merged["x1"]) ** 2
    df_merged["dist_y"] = (df_merged["y0"] - df_merged["y1"]) ** 2
    df_merged["dist_z"] = (df_merged["z0"] - df_merged["z1"]) ** 2
    df_merged["dist"] = np.sqrt(
        df_merged["dist_x"] + df_merged["dist_y"] + df_merged["dist_z"]
    )

    # 3. Correlations
    print("\nCorrelations with Target (scalar_coupling_constant):")
    # Select numerical features for correlation
    corr_cols = ["atom_index_0", "atom_index_1", "dist", "scalar_coupling_constant"]
    corr_matrix = df_merged[corr_cols].corr(method="pearson")
    target_corr = corr_matrix["scalar_coupling_constant"].drop(
        "scalar_coupling_constant"
    )

    for feat, corr in target_corr.items():
        print(f"  {feat}: {corr:.4f}")

    # Check for Collinearity
    print("\nRedundancy Check (Correlation > 0.90):")
    # Check between features only
    feat_corr = df_merged[["atom_index_0", "atom_index_1", "dist"]].corr().abs()
    high_corr_pairs = []
    for i in range(len(feat_corr.columns)):
        for j in range(i + 1, len(feat_corr.columns)):
            if feat_corr.iloc[i, j] > 0.9:
                high_corr_pairs.append(
                    (feat_corr.columns[i], feat_corr.columns[j], feat_corr.iloc[i, j])
                )

    if high_corr_pairs:
        for f1, f2, val in high_corr_pairs:
            print(f"  {f1} - {f2}: {val:.4f}")
    else:
        print("  No highly collinear numerical features found.")

    # 4. Feature Importance (Random Forest)
    print("\nFeature Importance (Random Forest on 50k sample):")
    # Sample data for speed
    sample_size = min(50000, len(df_merged))
    df_sample = df_merged.sample(n=sample_size, random_state=RANDOM_STATE).copy()

    # Preprocessing for RF
    le_type = LabelEncoder()
    df_sample["type_enc"] = le_type.fit_transform(df_sample["type"])

    le_atom0 = LabelEncoder()
    df_sample["atom_0_enc"] = le_atom0.fit_transform(df_sample["atom_0"])

    le_atom1 = LabelEncoder()
    df_sample["atom_1_enc"] = le_atom1.fit_transform(df_sample["atom_1"])

    features = [
        "atom_index_0",
        "atom_index_1",
        "dist",
        "type_enc",
        "atom_0_enc",
        "atom_1_enc",
    ]
    X = df_sample[features]
    y = df_sample["scalar_coupling_constant"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    # 5. Unstructured/Meta-Feature Relationships (Molecule Level)
    print("\nMolecule-Level Meta-Feature Analysis:")
    try:
        # Load auxiliary
        df_pot = pd.read_csv(INPUT_POTENTIAL)
        df_dip = pd.read_csv(INPUT_DIPOLE)

        # Calculate Dipole Magnitude
        df_dip["dipole_mag"] = np.sqrt(
            df_dip["X"] ** 2 + df_dip["Y"] ** 2 + df_dip["Z"] ** 2
        )

        # Merge with sample
        df_meta = df_sample.merge(df_pot, on="molecule_name", how="left")
        df_meta = df_meta.merge(
            df_dip[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
        )

        # Correlation
        meta_corr_pot = df_meta["scalar_coupling_constant"].corr(
            df_meta["potential_energy"]
        )
        meta_corr_dip = df_meta["scalar_coupling_constant"].corr(df_meta["dipole_mag"])

        print(f"  Correlation Target vs Potential Energy: {meta_corr_pot:.4f}")
        print(f"  Correlation Target vs Dipole Magnitude: {meta_corr_dip:.4f}")
        print(
            "  (Note: Weak correlation expected as these are global molecule properties vs local atom-pair target)"
        )

    except Exception as e:
        print(f"  Could not perform meta-feature analysis: {e}")

    print("-" * 30)


def main():
    set_seed(RANDOM_STATE)

    # Load Data
    try:
        df_train, df_structures = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Run Analysis
    analyze_target(df_train)
    analyze_tabular_inputs(df_train)
    engineer_features_and_analyze_relationships(df_train, df_structures)


if __name__ == "__main__":
    main()
