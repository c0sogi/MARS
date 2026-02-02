import os
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def analyze_target(df, target_col):
    print("=" * 30)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    scores = df[target_col]

    # Distribution stats
    print(f"Target Column: '{target_col}'")
    print(f"Count: {len(scores)}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std Dev: {scores.std():.4f}")
    print(f"Min: {scores.min():.4f}")
    print(f"Max: {scores.max():.4f}")

    # Normality check (Skew/Kurtosis)
    skew = stats.skew(scores)
    kurt = stats.kurtosis(scores)
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurt:.4f}")

    # Class balance (since scores are discrete 0, 0.25, etc.)
    print("\n[Value Counts & Ratios]")
    counts = scores.value_counts().sort_index()
    ratios = scores.value_counts(normalize=True).sort_index()
    for val, count in counts.items():
        print(f"Score {val:.2f}: {count} ({ratios[val]*100:.2f}%)")


def analyze_text_inputs(df, text_cols, cat_cols):
    print("\n" + "=" * 30)
    print("INPUT DATA ANALYSIS (TEXT & CATEGORICAL)")
    print("=" * 30)

    # 1. Categorical Analysis (Context)
    for col in cat_cols:
        print(f"\n[Categorical Column: {col}]")
        unique_vals = df[col].nunique()
        print(f"Cardinality: {unique_vals}")

        # Check for rare labels
        counts = df[col].value_counts(normalize=True)
        rare_labels = counts[counts < 0.01]
        print(f"Rare labels (<1% freq): {len(rare_labels)} out of {unique_vals}")

        # Most common
        top_5 = df[col].value_counts().head(5)
        print(f"Top 5 categories: {top_5.to_dict()}")

        # Missing values
        nans = df[col].isna().sum()
        print(f"Missing values: {nans} ({nans/len(df)*100:.2f}%)")

    # 2. Text Analysis (Anchor & Target)
    for col in text_cols:
        print(f"\n[Text Column: {col}]")

        # Missing
        nans = df[col].isna().sum()
        if nans > 0:
            print(f"Missing values: {nans} ({nans/len(df)*100:.2f}%)")
            # Fill NA for length calc
            series = df[col].fillna("")
        else:
            print(f"Missing values: 0 (0.0000%)")
            series = df[col]

        # Lengths (Characters)
        char_lens = series.astype(str).apply(len)
        print(
            f"Char Length -> Mean: {char_lens.mean():.4f}, Std: {char_lens.std():.4f}, Max: {char_lens.max()}"
        )

        # Lengths (Words - simple split)
        word_lens = series.astype(str).apply(lambda x: len(x.split()))
        print(
            f"Word Count  -> Mean: {word_lens.mean():.4f}, Std: {word_lens.std():.4f}, Max: {word_lens.max()}"
        )

        # Vocabulary
        # We use a simple CountVectorizer to get unique tokens
        vec = CountVectorizer(stop_words="english")
        try:
            vec.fit(series.astype(str))
            vocab_size = len(vec.vocabulary_)
            print(f"Vocabulary Size (approx, no stopwords): {vocab_size}")
        except ValueError:
            print("Vocabulary Size: Unable to calculate (possibly empty text).")


def get_jaccard_sim(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    if len(a) + len(b) - len(c) == 0:
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_levenshtein_ratio(s1, s2):
    # Simple implementation or use library if available.
    # Since we can't rely on python-Levenshtein being installed, we use a basic ratio based on difflib or simple logic.
    # However, for speed and standard library usage, we can use SequenceMatcher
    from difflib import SequenceMatcher

    return SequenceMatcher(None, str(s1), str(s2)).ratio()


def analyze_relationships(df, target_col):
    print("\n" + "=" * 30)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Create Meta-Features
    print("Generating meta-features for relationship analysis...")
    meta_df = pd.DataFrame()
    meta_df["target"] = df[target_col]

    # Length features
    meta_df["len_anchor"] = df["anchor"].astype(str).apply(len)
    meta_df["len_target"] = df["target"].astype(str).apply(len)
    meta_df["len_diff"] = abs(meta_df["len_anchor"] - meta_df["len_target"])

    # Word count features
    meta_df["wc_anchor"] = df["anchor"].astype(str).apply(lambda x: len(x.split()))
    meta_df["wc_target"] = df["target"].astype(str).apply(lambda x: len(x.split()))

    # Similarity features
    meta_df["jaccard"] = df.apply(
        lambda x: get_jaccard_sim(x["anchor"], x["target"]), axis=1
    )
    meta_df["levenshtein"] = df.apply(
        lambda x: get_levenshtein_ratio(x["anchor"], x["target"]), axis=1
    )

    # Context encoding (Ordinal for correlation check)
    enc = OrdinalEncoder()
    meta_df["context_enc"] = enc.fit_transform(df[["context"]])

    # 1. Correlations
    print("\n[Correlations with Target]")
    corrs = meta_df.corr(method="pearson")["target"].sort_values(ascending=False)
    print(corrs.drop("target"))

    print("\n[Collinearity Check (Corr > 0.90)]")
    # Check feature-feature correlations
    features = meta_df.drop(columns=["target"])
    feat_corr = features.corr().abs()
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    if high_corr:
        for col in high_corr:
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"Feature '{col}' is highly correlated with: {correlated_with}")
    else:
        print("No highly collinear features found.")

    # 2. Feature Importance (Random Forest)
    print("\n[Feature Importance (Random Forest Regressor)]")
    X = features.fillna(0)
    y = meta_df["target"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("Top 5 Features:")
    print(importances.head(5))

    # 3. Meta-Feature Insights
    print("\n[Meta-Feature Insights]")
    # Check if longer shared words correlate with score
    # We bin the Levenshtein ratio
    meta_df["lev_bin"] = pd.cut(meta_df["levenshtein"], bins=5)
    mean_score_by_lev = meta_df.groupby("lev_bin", observed=True)["target"].mean()
    print("Mean Score by Levenshtein Ratio Bins:")
    print(mean_score_by_lev)


def main():
    set_seed(42)

    # Configuration
    TRAIN_PATH = "./metadata/train.csv"

    print("Starting EDA Script...")
    print(f"Reading data from {TRAIN_PATH}")

    try:
        df_train = load_data(TRAIN_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. Target Analysis
    analyze_target(df_train, "score")

    # 2. Input Analysis
    # Anchor and Target are text, Context is categorical
    analyze_text_inputs(df_train, text_cols=["anchor", "target"], cat_cols=["context"])

    # 3. Relationship Analysis
    analyze_relationships(df_train, "score")

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
