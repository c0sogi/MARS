import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
import warnings

# Configuration
DATA_PATH = "./metadata/train.csv"
SEED = 42
SAMPLE_SIZE_RF = 100000  # Sample size for Random Forest to ensure speed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    np.random.seed(seed)


def calculate_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
        return

    # Define roles
    target_col = "pressure"
    id_cols = ["id", "breath_id"]
    feature_cols = [c for c in df.columns if c not in [target_col] + id_cols]

    # 3. Target Variable Analysis
    print("TARGET VARIABLE ANALYSIS")
    target = df[target_col]

    # Distribution Stats
    print(f"Target: {target_col}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std:  {target.std():.4f}")
    print(f"Min:  {target.min():.4f}")
    print(f"Max:  {target.max():.4f}")

    # Normality
    target_skew = skew(target)
    target_kurt = kurtosis(target)
    print(f"Skewness: {target_skew:.4f}")
    print(f"Kurtosis: {target_kurt:.4f}")
    print("-" * 30)

    # 4. Input Data Analysis (Tabular)
    print("INPUT DATA ANALYSIS")

    # Missing Values
    print("Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        for col in df.columns:
            if missing[col] > 0:
                print(f"{col}: {missing[col]} ({missing_pct[col]:.4f}%)")
    print("")

    # Numerical Analysis
    # In this dataset, R, C, u_out are technically discrete/categorical but encoded as numbers.
    # u_in and time_step are continuous.
    # We will analyze all feature columns as numerical first as per standard tabular EDA for regression.
    print("Numerical Statistics:")
    stats_df = df[feature_cols].describe().T
    stats_df["outliers_iqr"] = df[feature_cols].apply(calculate_outliers_iqr)

    print(
        f"{'Column':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers':<10}"
    )
    for idx, row in stats_df.iterrows():
        print(
            f"{idx:<15} {row['mean']:.4f}     {row['std']:.4f}     {row['min']:.4f}     {row['max']:.4f}     {int(row['outliers_iqr']):<10}"
        )
    print("")

    # Categorical/Cardinality Analysis
    # Checking for columns that might be effectively categorical
    print("Cardinality Analysis:")
    for col in feature_cols:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
        if unique_count > 50:
            pass  # High cardinality, likely continuous
        else:
            # Check for rare labels (< 1%)
            counts = df[col].value_counts(normalize=True)
            rare_labels = counts[counts < 0.01]
            if not rare_labels.empty:
                print(f"  -> Flag: Contains {len(rare_labels)} rare labels (< 1% freq)")
    print("-" * 30)

    # 5. Feature/Signal Relationships
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Correlation
    print("Correlation with Target (Pearson):")
    # Calculate correlation of features with target
    correlations = df[feature_cols].corrwith(df[target_col])
    for col, corr in correlations.items():
        print(f"{col}: {corr:.4f}")
    print("")

    # Redundancy (Collinearity)
    print("Redundancy Check (Correlation > 0.90):")
    corr_matrix = df[feature_cols].corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    redundant_pairs = []
    for col in to_drop:
        # Find the row index (feature) that correlates highly with this column
        correlated_feats = upper.index[upper[col] > 0.90].tolist()
        for feat in correlated_feats:
            redundant_pairs.append((feat, col))

    if not redundant_pairs:
        print("No collinear pairs found (> 0.90).")
    else:
        for pair in redundant_pairs:
            val = corr_matrix.loc[pair[0], pair[1]]
            print(f"{pair[0]} - {pair[1]}: {val:.4f}")
    print("")

    # Feature Importance (Random Forest)
    print("Feature Importance (Random Forest):")
    # Sampling for speed
    if len(df) > SAMPLE_SIZE_RF:
        sample_df = df.sample(n=SAMPLE_SIZE_RF, random_state=SEED)
    else:
        sample_df = df

    X_sample = sample_df[feature_cols]
    y_sample = sample_df[target_col]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED, verbose=0
    )
    rf.fit(X_sample, y_sample)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Features:")
    for i in range(min(5, len(feature_cols))):
        feat_name = feature_cols[indices[i]]
        importance = importances[indices[i]]
        print(f"{i+1}. {feat_name}: {importance:.4f}")


if __name__ == "__main__":
    main()
