import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Configuration & Setup
# ==========================================
# Suppress warnings
warnings.filterwarnings("ignore")

# Set Random Seeds for Reproducibility
SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# File Paths
TRAIN_META_PATH = "./metadata/train_metadata.csv"
STRUCTURES_PATH = "./input/structures.csv"
POTENTIAL_ENERGY_PATH = "./input/potential_energy.csv"
DIPOLE_MOMENTS_PATH = "./input/dipole_moments.csv"


def run_eda():
    # ==========================================
    # 1. Data Loading & Preprocessing
    # ==========================================
    # Load primary training data
    df_train = pd.read_csv(TRAIN_META_PATH)

    # Load auxiliary data
    df_struct = pd.read_csv(STRUCTURES_PATH)
    df_pot = pd.read_csv(POTENTIAL_ENERGY_PATH)
    df_dip = pd.read_csv(DIPOLE_MOMENTS_PATH)

    # --- Feature Engineering for EDA ---
    # We need to calculate distances to make the analysis meaningful.
    # Merge structure info for atom_index_0
    df_train = pd.merge(
        df_train,
        df_struct,
        how="left",
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
    )
    df_train.rename(
        columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom_0"}, inplace=True
    )
    df_train.drop(columns=["atom_index"], inplace=True)

    # Merge structure info for atom_index_1
    df_train = pd.merge(
        df_train,
        df_struct,
        how="left",
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
    )
    df_train.rename(
        columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom_1"}, inplace=True
    )
    df_train.drop(columns=["atom_index"], inplace=True)

    # Calculate Euclidean Distance
    df_train["dist"] = np.sqrt(
        (df_train["x0"] - df_train["x1"]) ** 2
        + (df_train["y0"] - df_train["y1"]) ** 2
        + (df_train["z0"] - df_train["z1"]) ** 2
    )

    # Merge Potential Energy
    df_train = pd.merge(df_train, df_pot, on="molecule_name", how="left")

    # Merge Dipole Moments and calculate magnitude
    df_dip["dipole_mag"] = np.sqrt(
        df_dip["X"] ** 2 + df_dip["Y"] ** 2 + df_dip["Z"] ** 2
    )
    df_train = pd.merge(
        df_train,
        df_dip[["molecule_name", "dipole_mag"]],
        on="molecule_name",
        how="left",
    )

    # Define Numerical and Categorical columns for analysis
    # We exclude IDs and raw coordinates from direct distribution analysis to focus on derived features
    num_cols = ["dist", "potential_energy", "dipole_mag"]
    cat_cols = ["type", "atom_0", "atom_1"]
    target_col = "scalar_coupling_constant"

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    target_vals = df_train[target_col]

    print(f"Target Name: {target_col}")
    print(f"Count: {len(target_vals)}")
    print(f"Mean: {target_vals.mean():.4f}")
    print(f"Std Dev: {target_vals.std():.4f}")
    print(f"Min: {target_vals.min():.4f}")
    print(f"Max: {target_vals.max():.4f}")

    # Normality Check
    t_skew = skew(target_vals)
    t_kurt = kurtosis(target_vals)
    print(f"Skewness: {t_skew:.4f}")
    print(f"Kurtosis: {t_kurt:.4f}")
    if abs(t_skew) > 1 or abs(t_kurt) > 1:
        print("Note: Target distribution deviates significantly from normal.")

    print("\nTarget Stats by Coupling Type:")
    type_stats = df_train.groupby("type")[target_col].agg(["mean", "std", "count"])
    for idx, row in type_stats.iterrows():
        print(
            f"  {idx}: Mean={row['mean']:.4f}, Std={row['std']:.4f}, Count={int(row['count'])}"
        )

    # ==========================================
    # 3. Input Data Analysis (Tabular)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TABULAR)")

    # --- Numerical Features ---
    print("\n-- Numerical Features --")
    for col in num_cols:
        vals = df_train[col]
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((vals < lower_bound) | (vals > upper_bound)).sum()

        print(f"Feature: {col}")
        print(f"  Mean: {vals.mean():.4f}")
        print(f"  Std: {vals.std():.4f}")
        print(f"  Min: {vals.min():.4f}")
        print(f"  Max: {vals.max():.4f}")
        print(f"  Outliers (IQR method): {outliers} ({outliers/len(vals)*100:.2f}%)")

    # --- Categorical Features ---
    print("\n-- Categorical Features --")
    for col in cat_cols:
        print(f"Feature: {col}")
        unique_vals = df_train[col].nunique()
        print(f"  Cardinality: {unique_vals}")

        # Check for rare labels
        counts = df_train[col].value_counts(normalize=True)
        rare = counts[counts < 0.01]
        if len(rare) > 0:
            print(f"  Rare labels (<1%): {list(rare.index)}")
        else:
            print("  Rare labels (<1%): None")

    # --- Missing Values ---
    print("\n-- Missing Values --")
    missing = df_train.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df_train)*100:.4f}%)")
    else:
        print("  No missing values found in the training set.")

    # ==========================================
    # 4. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # --- Correlation ---
    print("\n-- Correlations (Pearson) --")
    # Calculate correlation matrix for numerical features + target
    corr_cols = num_cols + [target_col]
    corr_matrix = df_train[corr_cols].corr()

    # Print correlations with target
    print(f"Correlation with {target_col}:")
    for col in num_cols:
        print(f"  {col}: {corr_matrix.loc[col, target_col]:.4f}")

    # --- Redundancy ---
    print("\n-- Redundancy Check (Collinear Pairs > 0.90) --")
    redundant_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1 = num_cols[i]
            c2 = num_cols[j]
            corr_val = abs(corr_matrix.loc[c1, c2])
            if corr_val > 0.90:
                redundant_pairs.append(f"{c1} & {c2} ({corr_val:.4f})")

    if redundant_pairs:
        for pair in redundant_pairs:
            print(f"  {pair}")
    else:
        print("  No highly collinear pairs found among numerical features.")

    # --- Feature Importance (Random Forest) ---
    print("\n-- Feature Importance (Lightweight Random Forest) --")

    # Prepare data for RF
    # Sample data to keep it lightweight and fast (100k samples)
    sample_size = min(100000, len(df_train))
    df_sample = df_train.sample(n=sample_size, random_state=SEED).copy()

    # Encode categoricals
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_sample[col] = le.fit_transform(df_sample[col].astype(str))
        le_dict[col] = le

    X_rf = df_sample[num_cols + cat_cols]
    y_rf = df_sample[target_col]

    # Train RF
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
    )
    rf.fit(X_rf, y_rf)

    # Get importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"Top 5 Features (trained on {sample_size} samples):")
    for i in range(min(5, len(indices))):
        feat_name = X_rf.columns[indices[i]]
        score = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {score:.4f}")

    # --- Metadata Relationship ---
    print("\n-- Metadata Relationships --")
    # Check if larger molecules (more atoms) have different target distributions
    # We can infer molecule size by counting atoms in structures.csv for the sampled molecules
    # But simpler: check correlation between potential energy (proxy for size/complexity) and target
    pot_corr = df_train["potential_energy"].corr(df_train[target_col])
    print(
        f"Correlation between Potential Energy (proxy for complexity) and Target: {pot_corr:.4f}"
    )

    # Check if distance is the dominant factor per type
    print("Correlation between Distance and Target per Type:")
    for t in df_train["type"].unique():
        subset = df_train[df_train["type"] == t]
        c = subset["dist"].corr(subset[target_col])
        print(f"  {t}: {c:.4f}")


if __name__ == "__main__":
    run_eda()
