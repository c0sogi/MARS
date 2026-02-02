import os
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_FILE = "./metadata/train.csv"
RANDOM_STATE = 42

# Set random seeds
np.random.seed(RANDOM_STATE)


def run_eda():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    # Identify Column Types
    target_col = "requester_received_pizza"

    # Potential leakage columns to exclude from predictive analysis
    leakage_cols = ["giver_username_if_known", "request_id", "source_file"]

    # Text columns
    text_cols = ["request_text", "request_title", "request_text_edit_aware"]

    # Identify numerical and categorical columns automatically
    # We exclude the target, leakage, and text columns from this automatic selection
    exclude_cols = [target_col] + leakage_cols + text_cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    print("\n" + "=" * 30)
    print("2. TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    # Target Distribution
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        proportions = df[target_col].value_counts(normalize=True)
        print(f"Target: {target_col}")
        print(f"Class Distribution:\n{counts}")
        print(f"Class Proportions:\n{proportions.apply(lambda x: f'{x:.4f}')}")

        # Imbalance check
        minority_class_prop = proportions.min()
        print(f"Minority Class Proportion: {minority_class_prop:.4f}")
        if minority_class_prop < 0.1:
            print("Status: Highly Imbalanced")
        elif minority_class_prop < 0.4:
            print("Status: Moderately Imbalanced")
        else:
            print("Status: Balanced")
    else:
        print(f"Target column '{target_col}' not found.")

    print("\n" + "=" * 30)
    print("3. INPUT DATA ANALYSIS (TABULAR)")
    print("=" * 30)

    # Numerical Analysis
    print("--- Numerical Data ---")
    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats["iqr"] = stats["75%"] - stats["25%"]

        # Outlier detection (1.5 * IQR)
        outlier_counts = {}
        for col in numeric_cols:
            q1 = stats.loc[col, "25%"]
            q3 = stats.loc[col, "75%"]
            iqr = stats.loc[col, "iqr"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)

        print(
            f"{'Feature':<50} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Outliers':<8}"
        )
        print("-" * 115)
        for col in numeric_cols:
            print(
                f"{col:<50} | {stats.loc[col, 'mean']:<10.4f} | {stats.loc[col, 'std']:<10.4f} | "
                f"{stats.loc[col, 'min']:<10.4f} | {stats.loc[col, 'max']:<10.4f} | {outlier_counts[col]:<8}"
            )
    else:
        print("No numerical columns found.")

    # Categorical Analysis
    print("\n--- Categorical Data ---")
    if categorical_cols:
        print(
            f"{'Feature':<40} | {'Cardinality':<12} | {'Rare Labels (<1%)':<20} | {'Missing':<10}"
        )
        print("-" * 90)
        for col in categorical_cols:
            cardinality = df[col].nunique()
            missing = df[col].isnull().sum()

            # Check for rare labels
            if cardinality > 0:
                value_counts = df[col].value_counts(normalize=True)
                rare_count = (value_counts < 0.01).sum()
                rare_flag = "Yes" if rare_count > 0 else "No"
            else:
                rare_flag = "N/A"

            # Truncate name if too long
            col_name = (col[:37] + "..") if len(col) > 37 else col
            print(
                f"{col_name:<40} | {cardinality:<12} | {rare_flag:<20} | {missing:<10}"
            )

            if cardinality > 50:
                print(f"  -> Warning: High cardinality for '{col}'.")
    else:
        print("No categorical columns found.")

    # Missing Values
    print("\n--- Missing Values ---")
    missing_total = df.isnull().sum()
    missing_cols = missing_total[missing_total > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"{col}: {count} missing ({pct:.2f}%)")
    else:
        print("No missing values found in the dataset.")

    print("\n" + "=" * 30)
    print("3. INPUT DATA ANALYSIS (TEXT)")
    print("=" * 30)

    # Analyze request_text_edit_aware (preferred over raw text to avoid edit leakage)
    text_target = "request_text_edit_aware"
    if text_target in df.columns:
        print(f"Analyzing text column: {text_target}")

        # Fill NaNs with empty string for analysis
        texts = df[text_target].fillna("").astype(str)

        # Length Analysis
        char_lengths = texts.apply(len)
        word_lengths = texts.apply(lambda x: len(x.split()))

        print(
            f"Character Lengths -> Mean: {char_lengths.mean():.4f}, Std: {char_lengths.std():.4f}, Max: {char_lengths.max()}"
        )
        print(
            f"Word Counts       -> Mean: {word_lengths.mean():.4f}, Std: {word_lengths.std():.4f}, Max: {word_lengths.max()}"
        )

        # Vocabulary Analysis
        try:
            # Limit features to avoid OOM on large datasets, but enough to estimate vocab
            vectorizer = CountVectorizer(stop_words="english", max_features=10000)
            vectorizer.fit(texts)
            vocab_size = len(vectorizer.vocabulary_)
            print(f"Vocabulary Size (approx, top 10k max): {vocab_size}")
        except Exception as e:
            print(f"Could not calculate vocabulary size: {e}")

    else:
        print(f"Text column '{text_target}' not found.")

    print("\n" + "=" * 30)
    print("4. FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Correlation Analysis (Numerical)
    print("--- Numerical Correlations with Target ---")
    if numeric_cols and target_col in df.columns:
        # Encode target as int for correlation
        y = df[target_col].astype(int)
        correlations = {}
        for col in numeric_cols:
            # Handle NaNs for correlation calculation
            if df[col].isnull().any():
                corr = df[col].fillna(df[col].mean()).corr(y)
            else:
                corr = df[col].corr(y)
            correlations[col] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("Top 10 Numerical Features by Pearson Correlation:")
        for name, score in sorted_corr[:10]:
            print(f"{name:<50}: {score:.4f}")

        # Collinearity Check
        print("\n--- High Collinearity (>0.90) ---")
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

        if to_drop:
            for col in to_drop:
                correlated_with = upper.index[upper[col] > 0.90].tolist()
                print(f"{col} is highly correlated with: {correlated_with}")
        else:
            print("No highly collinear pairs detected.")

    # Meta-Feature Relationships (Text Length vs Target)
    print("\n--- Meta-Feature Relationships ---")
    if text_target in df.columns and target_col in df.columns:
        df["temp_word_count"] = (
            df[text_target].fillna("").astype(str).apply(lambda x: len(x.split()))
        )

        mean_len_pos = df[df[target_col] == True]["temp_word_count"].mean()
        mean_len_neg = df[df[target_col] == False]["temp_word_count"].mean()

        print(f"Mean Word Count (Pizza Received):     {mean_len_pos:.4f}")
        print(f"Mean Word Count (No Pizza Received):  {mean_len_neg:.4f}")
        if mean_len_pos > mean_len_neg:
            print("Observation: Successful requests tend to be longer.")
        else:
            print("Observation: Successful requests tend to be shorter.")

    # Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    try:
        # Prepare data for RF
        # Use numeric cols + simple text stats
        rf_df = df[numeric_cols].copy()
        rf_df = rf_df.fillna(0)  # Simple imputation for importance check

        # Add meta features
        if text_target in df.columns:
            rf_df["text_len_chars"] = df[text_target].fillna("").astype(str).apply(len)

        # Target
        y = df[target_col].astype(int)

        # Train Model
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(rf_df, y)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("Top 5 Features by Random Forest Importance:")
        for f in range(min(5, len(numeric_cols))):
            print(f"{rf_df.columns[indices[f]]:<50}: {importances[indices[f]]:.4f}")

    except Exception as e:
        print(f"Could not run Feature Importance: {e}")


if __name__ == "__main__":
    run_eda()
