import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

# Set fixed random seeds for reproducibility
import random

random.seed(42)
np.random.seed(42)


def run_eda():
    # ==========================================
    # 1. DATA INTEGRITY & LOADING
    # ==========================================
    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Load metadata to identify training split
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        print(f"Error: Metadata file not found at {train_meta_path}")
        return

    df_meta_train = pd.read_csv(train_meta_path)
    train_ids = set(df_meta_train["id"].values)

    # Load raw JSON data
    # We only load train.json as per instructions to analyze training data
    json_path = os.path.join(INPUT_DIR, "train.json")
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Filter raw data to keep only those in the training split
    train_data = [item for item in raw_data if item["id"] in train_ids]

    # Convert to DataFrame for easier handling of tabular components
    # Note: band_1 and band_2 are lists in the json
    df_train = pd.DataFrame(train_data)

    # Ensure strict alignment with metadata (though filter should handle it)
    df_train = df_train[df_train["id"].isin(train_ids)].reset_index(drop=True)

    # Parse 'inc_angle' to numeric, coercing 'na' to NaN
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # ==========================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================
    print("\nTARGET VARIABLE ANALYSIS")
    print("------------------------")

    target_col = "is_iceberg"
    counts = df_train[target_col].value_counts()
    proportions = df_train[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Class 0 (Ship):    {counts.get(0, 0)} ({proportions.get(0, 0):.4f})")
    print(f"Class 1 (Iceberg): {counts.get(1, 0)} ({proportions.get(1, 0):.4f})")

    # Check for imbalance
    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    print(f"Class Balance Ratio (0:1): {ratio:.4f}")

    # ==========================================
    # 3. INPUT DATA ANALYSIS (IMAGE MODALITY)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (IMAGE)")
    print("---------------------------")

    # Extract bands
    # Stack them into numpy arrays: (N_samples, 5625)
    band_1_stack = np.stack(df_train["band_1"].values)
    band_2_stack = np.stack(df_train["band_2"].values)

    # Check dimensions
    # Expected 75x75 = 5625
    n_pixels = band_1_stack.shape[1]
    side_length = int(np.sqrt(n_pixels))

    print(f"Image Structure:")
    print(f"  Flattened Length: {n_pixels}")
    if side_length * side_length == n_pixels:
        print(f"  Inferred Dimensions: {side_length} x {side_length}")
        print(f"  Aspect Ratio: 1.0000")
    else:
        print(f"  Inferred Dimensions: Non-square or irregular ({n_pixels} pixels)")

    print(f"  Channels: 2 (Band 1: HH, Band 2: HV)")

    # Pixel Stats (Global)
    # Band 1
    b1_mean = np.mean(band_1_stack)
    b1_std = np.std(band_1_stack)
    b1_min = np.min(band_1_stack)
    b1_max = np.max(band_1_stack)

    # Band 2
    b2_mean = np.mean(band_2_stack)
    b2_std = np.std(band_2_stack)
    b2_min = np.min(band_2_stack)
    b2_max = np.max(band_2_stack)

    print("\nPixel Value Statistics (dB):")
    print(
        f"  Band 1 (HH) - Mean: {b1_mean:.4f}, Std: {b1_std:.4f}, Min: {b1_min:.4f}, Max: {b1_max:.4f}"
    )
    print(
        f"  Band 2 (HV) - Mean: {b2_mean:.4f}, Std: {b2_std:.4f}, Min: {b2_min:.4f}, Max: {b2_max:.4f}"
    )

    # ==========================================
    # 4. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # --- A. Structured Relationships (Derived Features) ---
    print("A. Structured (Tabular) Relationships")

    # Feature Engineering for Analysis:
    # Create summary statistics for each image to treat as tabular features
    features_df = pd.DataFrame()
    features_df["inc_angle"] = df_train["inc_angle"]

    # Band 1 stats per image
    features_df["b1_mean"] = np.mean(band_1_stack, axis=1)
    features_df["b1_std"] = np.std(band_1_stack, axis=1)
    features_df["b1_min"] = np.min(band_1_stack, axis=1)
    features_df["b1_max"] = np.max(band_1_stack, axis=1)

    # Band 2 stats per image
    features_df["b2_mean"] = np.mean(band_2_stack, axis=1)
    features_df["b2_std"] = np.std(band_2_stack, axis=1)
    features_df["b2_min"] = np.min(band_2_stack, axis=1)
    features_df["b2_max"] = np.max(band_2_stack, axis=1)

    # Add target for correlation
    features_df["target"] = df_train["is_iceberg"]

    # 1. Correlation
    # Drop rows with NaN in inc_angle for correlation calculation
    corr_df = features_df.dropna()
    correlations = corr_df.corr(method="pearson")["target"].sort_values(ascending=False)

    print("\n  Top 5 Correlations with Target (Pearson):")
    # Filter out the target itself
    top_corr = correlations.drop("target").head(5)
    for name, val in top_corr.items():
        print(f"    {name}: {val:.4f}")

    print("\n  Bottom 5 Correlations with Target (Pearson):")
    bot_corr = correlations.drop("target").tail(5)
    for name, val in bot_corr.items():
        print(f"    {name}: {val:.4f}")

    # Check for Redundancy (Collinearity > 0.90)
    print("\n  Redundant Feature Pairs (Correlation > 0.90):")
    # Compute correlation matrix of features only
    feat_corr = corr_df.drop(columns=["target"]).corr().abs()
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    found_redundancy = False
    for col in to_drop:
        correlated_cols = upper.index[upper[col] > 0.90].tolist()
        for c in correlated_cols:
            print(f"    {col} <-> {c} (Corr: {upper.loc[c, col]:.4f})")
            found_redundancy = True
    if not found_redundancy:
        print("    None found among derived features.")

    # 2. Feature Importance (Random Forest)
    # Handle Missing Values in inc_angle for RF
    imputer = SimpleImputer(strategy="mean")
    X = features_df.drop(columns=["target"])
    y = features_df["target"]

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=5
    )
    rf.fit(X_imputed, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    print("\n  Random Forest Feature Importance (Top 5):")
    for name, val in importances.head(5).items():
        print(f"    {name}: {val:.4f}")

    # --- B. Unstructured (Meta-Feature) Relationships ---
    print("\nB. Unstructured (Meta-Feature) Relationships")

    # Analyze 'inc_angle' specifically
    # Report missing values
    missing_inc = df_train["inc_angle"].isna().sum()
    total_rows = len(df_train)
    print(
        f"  Incidence Angle Missing Values: {missing_inc} / {total_rows} ({missing_inc/total_rows:.2%})"
    )

    # Compare distributions for Ship vs Iceberg
    # We use the non-missing data
    valid_inc = df_train.dropna(subset=["inc_angle"])

    mean_inc_ship = valid_inc[valid_inc["is_iceberg"] == 0]["inc_angle"].mean()
    std_inc_ship = valid_inc[valid_inc["is_iceberg"] == 0]["inc_angle"].std()

    mean_inc_ice = valid_inc[valid_inc["is_iceberg"] == 1]["inc_angle"].mean()
    std_inc_ice = valid_inc[valid_inc["is_iceberg"] == 1]["inc_angle"].std()

    print("\n  Incidence Angle by Class:")
    print(f"    Ship (0)    - Mean: {mean_inc_ship:.4f}, Std: {std_inc_ship:.4f}")
    print(f"    Iceberg (1) - Mean: {mean_inc_ice:.4f}, Std: {std_inc_ice:.4f}")

    # Point-biserial correlation is essentially Pearson between binary and continuous
    # We already calculated this in the correlation section, let's reiterate contextually
    inc_corr = correlations.get("inc_angle", 0)
    print(f"    Correlation (Inc Angle vs Target): {inc_corr:.4f}")
    if abs(inc_corr) < 0.1:
        print("    -> Weak linear relationship detected.")
    else:
        print("    -> Moderate/Significant linear relationship detected.")


if __name__ == "__main__":
    run_eda()
