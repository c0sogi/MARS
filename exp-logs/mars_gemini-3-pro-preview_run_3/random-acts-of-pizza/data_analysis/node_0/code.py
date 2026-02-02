import os
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
METADATA_PATH = "./metadata/train.parquet"
SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    df = pd.read_parquet(METADATA_PATH)
    return df


def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")


def analyze_target(df, target_col):
    print_header("TARGET VARIABLE ANALYSIS")

    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: '{target_col}'")
    print(f"Type: Classification (Binary)")
    print(f"Class Distribution:")
    for label, count in counts.items():
        prop = proportions[label]
        print(f"  Class {label}: {count} samples ({prop:.4%})")

    # Imbalance check
    majority_class_prop = proportions.max()
    if majority_class_prop > 0.6:
        print(f"Imbalance: Yes (Majority class is {majority_class_prop:.2%} of data)")
    else:
        print(f"Imbalance: No")


def analyze_tabular(df, numerical_cols, categorical_cols):
    print_header("INPUT DATA ANALYSIS: TABULAR")

    # --- Numerical Analysis ---
    print("--- Numerical Features ---")
    if not numerical_cols:
        print("No numerical columns found.")
    else:
        stats = []
        for col in numerical_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()

            # IQR Outliers
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_count = len(outliers)

            stats.append(
                {
                    "Feature": col,
                    "Mean": mean_val,
                    "Std": std_val,
                    "Min": min_val,
                    "Max": max_val,
                    "Outliers": outlier_count,
                }
            )

        stats_df = pd.DataFrame(stats)
        # Print formatted
        for _, row in stats_df.iterrows():
            print(
                f"{row['Feature']}: Mean={row['Mean']:.4f}, Std={row['Std']:.4f}, "
                f"Min={row['Min']:.4f}, Max={row['Max']:.4f}, Outliers={int(row['Outliers'])}"
            )

    # --- Categorical Analysis ---
    print("\n--- Categorical Features ---")
    if not categorical_cols:
        print("No categorical columns found.")
    else:
        for col in categorical_cols:
            # Skip if column contains lists (like subreddits list)
            if df[col].apply(lambda x: isinstance(x, list)).any():
                print(f"{col}: Contains lists (Skipping cardinality check)")
                continue

            series = df[col].astype(str)
            unique_count = series.nunique()
            print(f"{col}: Cardinality = {unique_count}")

            if unique_count > 50:
                print(f"  [FLAG] High cardinality (> 50 categories)")

            # Rare labels check
            counts = series.value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(f"  [FLAG] {len(rare)} labels have < 1% frequency")

    # --- Missing Values ---
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found in the selected columns.")
    else:
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            print(f"{col}: {count} NaNs ({pct:.4f}%)")


def analyze_text(df, text_cols):
    print_header("INPUT DATA ANALYSIS: TEXT")

    for col in text_cols:
        print(f"\nAnalysis for column: '{col}'")

        # Ensure string
        texts = df[col].astype(str).fillna("")

        # Lengths
        char_lengths = texts.apply(len)
        word_lengths = texts.apply(lambda x: len(x.split()))

        print(
            f"  Character Length: Mean={char_lengths.mean():.4f}, Std={char_lengths.std():.4f}, Max={char_lengths.max()}"
        )
        print(
            f"  Word Count:       Mean={word_lengths.mean():.4f}, Std={word_lengths.std():.4f}, Max={word_lengths.max()}"
        )

        # Vocabulary
        try:
            vectorizer = CountVectorizer(max_features=100000, stop_words="english")
            vectorizer.fit(texts)
            vocab_size = len(vectorizer.vocabulary_)
            print(f"  Vocabulary Size (approx, stop_words removed): {vocab_size}")
        except ValueError:
            # Handle empty vocabulary case
            print(f"  Vocabulary Size: 0 (Empty or stop-words only)")


def analyze_relationships(df, numerical_cols, text_cols, target_col):
    print_header("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Structured Relationships (Correlation)
    print("--- Numerical Correlation (Redundancy) ---")
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

        if not high_corr:
            print("No pairs with correlation > 0.90 found.")
        else:
            print("Pairs with Correlation > 0.90:")
            # Find specific pairs
            for col in high_corr:
                correlated_rows = upper.index[upper[col] > 0.90].tolist()
                for row in correlated_rows:
                    val = upper.loc[row, col]
                    print(f"  {row} <--> {col}: {val:.4f}")
    else:
        print("Not enough numerical columns for correlation analysis.")

    # 2. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Top 5) ---")

    # Prepare data for RF
    # We will use numerical columns + derived text length columns
    rf_df = df[numerical_cols].copy()

    # Impute numericals
    imputer = SimpleImputer(strategy="median")
    if not rf_df.empty:
        rf_df_imputed = pd.DataFrame(
            imputer.fit_transform(rf_df), columns=rf_df.columns, index=rf_df.index
        )
    else:
        rf_df_imputed = pd.DataFrame(index=df.index)

    # Add text meta-features
    for t_col in text_cols:
        rf_df_imputed[f"{t_col}_len_char"] = df[t_col].astype(str).fillna("").apply(len)
        rf_df_imputed[f"{t_col}_len_word"] = (
            df[t_col].astype(str).fillna("").apply(lambda x: len(x.split()))
        )

    y = df[target_col]
    X = rf_df_imputed

    if X.shape[1] > 0:
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        for f in range(min(5, X.shape[1])):
            print(f"  {f+1}. {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")
    else:
        print("No features available for importance analysis.")

    # 3. Unstructured (Meta-Feature) Relationships
    print("\n--- Meta-Feature Relationships ---")
    # Check if longer text correlates with success
    for t_col in text_cols:
        len_col = f"{t_col}_len_word"
        if len_col in X.columns:
            # Point Biserial correlation is Pearson when one var is binary
            corr, _ = pearsonr(X[len_col], y)
            print(f"Correlation between '{t_col}' word count and Target: {corr:.4f}")

            # Compare means
            mean_pos = X.loc[y == 1, len_col].mean()
            mean_neg = X.loc[y == 0, len_col].mean()
            print(f"  Avg Word Count (Pizza Received): {mean_pos:.2f}")
            print(f"  Avg Word Count (No Pizza):       {mean_neg:.2f}")


def main():
    set_seed(SEED)

    # Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define Column Groups based on dataset description
    target_col = "requester_received_pizza"

    # Text columns
    text_cols = ["request_text", "request_title"]

    # Identify Numerical columns automatically
    # Exclude target, text, and ID-like columns
    exclude_cols = (
        [target_col]
        + text_cols
        + ["request_id", "requester_username", "source_file", "request_text_edit_aware"]
    )

    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in exclude_cols]

    # Identify Categorical columns
    # We treat object cols that are not text/id as categorical
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    # Execute Analysis
    analyze_target(df, target_col)
    analyze_tabular(df, numerical_cols, categorical_cols)
    analyze_text(df, text_cols)
    analyze_relationships(df, numerical_cols, text_cols, target_col)


if __name__ == "__main__":
    main()
