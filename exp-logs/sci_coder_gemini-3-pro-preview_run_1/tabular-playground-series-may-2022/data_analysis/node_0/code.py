import os
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score

# 1. Configuration and Setup
SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def main():
    # Path to the training metadata file
    train_path = "./metadata/train.csv"

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    # Load Data
    # Using low_memory=False to ensure dtypes are inferred correctly for mixed types if any
    df = pd.read_csv(train_path)

    # Identify special columns
    id_col = "id"
    target_col = "target"
    ignore_cols = ["source_path", id_col]

    # Filter out metadata columns from features
    feature_cols = [c for c in df.columns if c not in [target_col] + ignore_cols]

    # ---------------------------------------------------------
    # 2. Target Variable Analysis
    # ---------------------------------------------------------
    print("TARGET VARIABLE ANALYSIS")

    # Check if classification or regression based on unique values
    # The task description says "predict whether the machine is in state 0 or state 1", implying binary classification.
    # We verify this by checking the number of unique values.
    unique_targets = df[target_col].dropna().unique()
    is_classification = False

    if len(unique_targets) <= 20:  # Heuristic for classification
        is_classification = True
        counts = df[target_col].value_counts()
        total = len(df)
        print(f"Type: Classification")
        print(f"Unique Classes: {sorted(unique_targets)}")
        print("Class Balance:")
        for cls, count in counts.items():
            ratio = count / total
            print(f"  Class {cls}: {count} ({ratio:.4f})")
    else:
        # Regression
        print(f"Type: Regression")
        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        skew = df[target_col].skew()
        kurt = df[target_col].kurtosis()
        print(f"Mean: {target_mean:.4f}")
        print(f"Std Dev: {target_std:.4f}")
        print(f"Skewness: {skew:.4f}")
        print(f"Kurtosis: {kurt:.4f}")

    print("-" * 30)

    # ---------------------------------------------------------
    # 3. Input Data Analysis (Tabular)
    # ---------------------------------------------------------
    print("INPUT DATA ANALYSIS (TABULAR)")

    # Separate Numerical and Categorical Features
    # We assume standard pandas inference.
    # Objects and Categories are categorical. Numbers are numerical.

    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Numerical Analysis ---
    if num_cols:
        print(f"Numerical Features: {len(num_cols)}")
        print(
            f"{'Feature':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers(IQR)':<15} {'NaNs(%)':<10}"
        )

        for col in num_cols:
            series = df[col]
            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()

            # Outliers via IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()

            # NaNs
            nan_count = series.isna().sum()
            nan_pct = (nan_count / len(df)) * 100

            print(
                f"{col[:19]:<20} {mean_val:.4f}     {std_val:.4f}     {min_val:.4f}     {max_val:.4f}     {outliers:<15} {nan_pct:.4f}%"
            )
    else:
        print("Numerical Features: 0")

    print("")

    # --- Categorical Analysis ---
    if cat_cols:
        print(f"Categorical Features: {len(cat_cols)}")
        print(
            f"{'Feature':<20} {'Cardinality':<12} {'NaNs(%)':<10} {'Rare Labels (>1%)':<20}"
        )

        for col in cat_cols:
            series = df[col].astype(str)  # Handle mixed types by converting to string
            cardinality = series.nunique()
            nan_count = df[col].isna().sum()
            nan_pct = (nan_count / len(df)) * 100

            # Rare labels check
            value_counts = series.value_counts(normalize=True)
            rare_labels = (value_counts < 0.01).sum()
            has_high_cardinality = cardinality > 50

            flag_str = ""
            if has_high_cardinality:
                flag_str += "High Card. "
            if rare_labels > 0:
                flag_str += f"{rare_labels} Rare"
            if flag_str == "":
                flag_str = "-"

            print(f"{col[:19]:<20} {cardinality:<12} {nan_pct:.4f}%    {flag_str:<20}")
    else:
        print("Categorical Features: 0")

    print("-" * 30)

    # ---------------------------------------------------------
    # 4. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # --- Correlation (Numerical) ---
    if len(num_cols) > 1:
        print("Top Correlated Pairs (Pearson > 0.90):")
        # Compute correlation matrix
        corr_matrix = df[num_cols].corr().abs()

        # Select upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation > 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        high_corr_pairs = []
        for col in upper.columns:
            high_corr_cols = upper.index[upper[col] > 0.90].tolist()
            for row in high_corr_cols:
                val = upper.loc[row, col]
                high_corr_pairs.append((row, col, val))

        if high_corr_pairs:
            for p in high_corr_pairs:
                print(f"  {p[0]} - {p[1]}: {p[2]:.4f}")
        else:
            print("  None found.")

    print("")

    # --- Feature Importance (Random Forest) ---
    print("Top 5 Features (Random Forest Importance):")

    # Prepare data for RF
    # We use a subset to speed up if data is very large, but 640k is manageable.
    # To be safe on time, we sample 50k rows for importance analysis.
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=SEED).copy()

    X_sample = df_sample[feature_cols]
    y_sample = df_sample[target_col]

    # Preprocessing for RF
    # 1. Impute Numerical
    if num_cols:
        num_imputer = SimpleImputer(strategy="mean")
        X_sample[num_cols] = num_imputer.fit_transform(X_sample[num_cols])

    # 2. Encode Categorical
    if cat_cols:
        # Fill NaNs with 'missing' before encoding
        X_sample[cat_cols] = X_sample[cat_cols].fillna("missing").astype(str)
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_sample[cat_cols] = ord_enc.fit_transform(X_sample[cat_cols])

    # Train RF
    if is_classification:
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=SEED
        )

    rf.fit(X_sample, y_sample)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(5, len(feature_cols))):
        feat_name = feature_cols[indices[i]]
        imp_val = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {imp_val:.4f}")


if __name__ == "__main__":
    main()
