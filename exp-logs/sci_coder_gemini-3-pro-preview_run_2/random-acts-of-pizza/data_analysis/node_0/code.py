import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    # Load Metadata
    train_meta_path = "./metadata/train.csv"
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {train_meta_path}")

    df_meta = pd.read_csv(train_meta_path)

    # Load Raw Train Data
    train_json_path = "./input/train.json"
    with open(train_json_path, "r") as f:
        raw_train_data = json.load(f)

    # Create DataFrame from raw data
    df_raw = pd.DataFrame(raw_train_data)

    # Merge with metadata to keep only training set and valid labels
    # We map using request_id to ensure alignment
    df_train = df_meta.merge(df_raw, on="request_id", how="left")

    # Handle target column name conflict if any (metadata has requester_received_pizza_x)
    if "requester_received_pizza_x" in df_train.columns:
        df_train.rename(
            columns={"requester_received_pizza_x": "requester_received_pizza"},
            inplace=True,
        )
        df_train.drop(
            columns=["requester_received_pizza_y"], inplace=True, errors="ignore"
        )

    # Load Test Data Sample to identify valid features (prevent leakage from future/retrieval fields)
    test_json_path = "./input/test.json"
    with open(test_json_path, "r") as f:
        test_sample = json.load(f)

    valid_features = list(test_sample[0].keys())

    # Ensure target is kept
    if "requester_received_pizza" not in valid_features:
        valid_features.append("requester_received_pizza")

    # Filter training data to only valid features + target
    # We intersect columns because some test features might be missing in train or vice versa (unlikely but safe)
    cols_to_keep = [c for c in valid_features if c in df_train.columns]
    df_train = df_train[cols_to_keep]

    return df_train


def analyze_target(df, target_col):
    print("=" * 30)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: {target_col}")
    print(f"Class 0 (Failure): {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"Class 1 (Success): {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    print(f"Imbalance Ratio (Neg/Pos): {ratio:.4f}")
    print("")


def analyze_tabular(df, target_col):
    print("=" * 30)
    print("INPUT DATA ANALYSIS (TABULAR)")
    print("=" * 30)

    # Identify column types
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target from input analysis
    if target_col in numerics:
        numerics.remove(target_col)

    objects = df.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Filter out text columns for tabular analysis (heuristically identified by name or length)
    text_cols = ["request_text", "request_text_edit_aware", "request_title"]
    cat_cols = [c for c in objects if c not in text_cols]

    # --- Numerical Analysis ---
    print("--- Numerical Features ---")
    if not numerics:
        print("No numerical features found.")
    else:
        stats = df[numerics].describe().T
        stats["IQR"] = df[numerics].quantile(0.75) - df[numerics].quantile(0.25)

        # Outlier detection (1.5 * IQR)
        outlier_counts = {}
        for col in numerics:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_counts[col] = len(outliers)

        print(
            f"{'Feature':<50} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Outliers':<10}"
        )
        print("-" * 105)
        for col in numerics:
            row = stats.loc[col]
            print(
                f"{col:<50} {row['mean']:<10.4f} {row['std']:<10.4f} {row['min']:<10.4f} {row['max']:<10.4f} {outlier_counts[col]:<10}"
            )
    print("")

    # --- Categorical Analysis ---
    print("--- Categorical Features ---")
    # Handle list columns (e.g. subreddits) separately or exclude
    clean_cat_cols = []
    for col in cat_cols:
        # Check if column contains lists
        if df[col].apply(lambda x: isinstance(x, list)).any():
            continue  # Skip list columns for standard categorical analysis
        clean_cat_cols.append(col)

    if not clean_cat_cols:
        print("No standard categorical features found (lists excluded).")
    else:
        print(f"{'Feature':<40} {'Cardinality':<12} {'Rare Labels (<1%)':<20}")
        print("-" * 80)
        for col in clean_cat_cols:
            cardinality = df[col].nunique()
            counts = df[col].value_counts(normalize=True)
            rare_count = (counts < 0.01).sum()
            print(f"{col:<40} {cardinality:<12} {rare_count:<20}")
    print("")

    # --- Missing Values ---
    print("--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(f"{'Feature':<40} {'Count':<10} {'Percentage':<10}")
        for col, count in missing.items():
            print(f"{col:<40} {count:<10} {count/len(df)*100:.4f}%")
    print("")


def analyze_text(df):
    print("=" * 30)
    print("INPUT DATA ANALYSIS (TEXT)")
    print("=" * 30)

    text_cols = ["request_text_edit_aware", "request_title"]
    # Fallback if edit_aware not present
    if "request_text_edit_aware" not in df.columns and "request_text" in df.columns:
        text_cols = ["request_text", "request_title"]

    valid_text_cols = [c for c in text_cols if c in df.columns]

    for col in valid_text_cols:
        print(f"Analyzing Text Column: {col}")

        # Fill NaNs with empty string
        texts = df[col].fillna("").astype(str)

        # Lengths
        char_lengths = texts.apply(len)
        word_lengths = texts.apply(lambda x: len(x.split()))

        print(
            f"  Character Lengths -> Mean: {char_lengths.mean():.4f}, Std: {char_lengths.std():.4f}, Max: {char_lengths.max()}"
        )
        print(
            f"  Word Lengths      -> Mean: {word_lengths.mean():.4f}, Std: {word_lengths.std():.4f}, Max: {word_lengths.max()}"
        )

        # Vocabulary
        all_words = " ".join(texts).split()
        unique_words = set(all_words)
        print(f"  Vocabulary Size (approx): {len(unique_words)}")
        print("-" * 30)
    print("")


def analyze_relationships(df, target_col):
    print("=" * 30)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Prepare data for correlation and importance
    # 1. Numerics
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerics:
        numerics.remove(target_col)

    # 2. Add Meta-features (Text Lengths)
    df_rel = df[numerics].copy()

    text_col = (
        "request_text_edit_aware"
        if "request_text_edit_aware" in df.columns
        else "request_text"
    )
    if text_col in df.columns:
        df_rel["text_len_char"] = df[text_col].fillna("").astype(str).apply(len)
        df_rel["text_len_word"] = (
            df[text_col].fillna("").astype(str).apply(lambda x: len(x.split()))
        )

    title_col = "request_title"
    if title_col in df.columns:
        df_rel["title_len_char"] = df[title_col].fillna("").astype(str).apply(len)

    # Fill missing values in numerics
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df_rel), columns=df_rel.columns)
    y = df[target_col]

    # --- Correlation ---
    print("--- Correlation with Target (Top 5) ---")
    # Add target back for correlation calculation
    X_corr = X.copy()
    X_corr["target"] = y.values
    corrs = X_corr.corr()["target"].drop("target").sort_values(key=abs, ascending=False)

    for feat, val in corrs.head(5).items():
        print(f"{feat:<40} {val:.4f}")
    print("")

    # --- Redundancy ---
    print("--- High Collinearity (Correlation > 0.90) ---")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if not to_drop:
        print("No highly collinear pairs found.")
    else:
        for col in to_drop:
            # Find the feature it correlates with
            correlated_feats = upper.index[upper[col] > 0.90].tolist()
            print(f"{col} correlates with: {correlated_feats}")
    print("")

    # --- Feature Importance (Random Forest) ---
    print("--- Feature Importance (Random Forest) ---")
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    for feat, val in importances.head(5).items():
        print(f"{feat:<40} {val:.4f}")
    print("")

    # --- Meta-Feature Insight ---
    print("--- Meta-Feature Insights ---")
    if "text_len_word" in X.columns:
        mean_len_success = X.loc[y == 1, "text_len_word"].mean()
        mean_len_fail = X.loc[y == 0, "text_len_word"].mean()
        print(f"Mean Word Count (Success): {mean_len_success:.4f}")
        print(f"Mean Word Count (Failure): {mean_len_fail:.4f}")
        if mean_len_success > mean_len_fail:
            print("-> Successful requests tend to be longer.")
        else:
            print("-> Successful requests tend to be shorter.")


def main():
    set_seed(42)

    try:
        df_train = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    target_col = "requester_received_pizza"

    # Ensure target is numeric (0/1)
    df_train[target_col] = df_train[target_col].astype(int)

    analyze_target(df_train, target_col)
    analyze_tabular(df_train, target_col)
    analyze_text(df_train)
    analyze_relationships(df_train, target_col)


if __name__ == "__main__":
    main()
