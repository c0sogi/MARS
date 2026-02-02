import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_eda():
    # 1. Setup
    set_seed(42)
    warnings.filterwarnings("ignore")

    data_path = "./metadata/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    print("========================================")
    print("      EXPLORATORY DATA ANALYSIS         ")
    print("========================================")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("")

    # 2. Target Variable Analysis
    print("TARGET VARIABLE ANALYSIS")
    print("------------------------")
    target_col = "score"

    # Distribution
    score_counts = df[target_col].value_counts().sort_index()
    print("Score Distribution:")
    for score, count in score_counts.items():
        percent = (count / len(df)) * 100
        print(f"  Score {score}: {count} ({percent:.2f}%)")

    # Skewness and Kurtosis
    target_skew = skew(df[target_col])
    target_kurt = kurtosis(df[target_col])
    print(f"Skewness: {target_skew:.4f}")
    print(f"Kurtosis: {target_kurt:.4f}")
    print("")

    # 3. Input Data Analysis (Text & Categorical)
    print("INPUT DATA ANALYSIS")
    print("-------------------")

    # Text Analysis
    # Anchor Analysis
    df["anchor_char_len"] = df["anchor"].astype(str).apply(len)
    df["anchor_word_len"] = df["anchor"].astype(str).apply(lambda x: len(x.split()))

    # Target Analysis
    df["target_char_len"] = df["target"].astype(str).apply(len)
    df["target_word_len"] = df["target"].astype(str).apply(lambda x: len(x.split()))

    print("Text Length Statistics (Anchor):")
    print(
        f"  Char Length: Mean={df['anchor_char_len'].mean():.4f}, Std={df['anchor_char_len'].std():.4f}, Max={df['anchor_char_len'].max()}"
    )
    print(
        f"  Word Length: Mean={df['anchor_word_len'].mean():.4f}, Std={df['anchor_word_len'].std():.4f}, Max={df['anchor_word_len'].max()}"
    )

    print("Text Length Statistics (Target):")
    print(
        f"  Char Length: Mean={df['target_char_len'].mean():.4f}, Std={df['target_char_len'].std():.4f}, Max={df['target_char_len'].max()}"
    )
    print(
        f"  Word Length: Mean={df['target_word_len'].mean():.4f}, Std={df['target_word_len'].std():.4f}, Max={df['target_word_len'].max()}"
    )

    # Vocabulary Analysis
    all_text = pd.concat([df["anchor"], df["target"]]).astype(str)
    # Simple whitespace tokenization
    vocab = set()
    for text in all_text:
        vocab.update(text.lower().split())

    print(f"Vocabulary Size (Unique Tokens): {len(vocab)}")

    # Context Analysis (Categorical)
    print("\nContext (CPC Code) Analysis:")
    context_counts = df["context"].value_counts()
    print(f"  Cardinality: {len(context_counts)}")
    print(
        f"  Most Frequent: {context_counts.index[0]} ({context_counts.iloc[0]} occurrences)"
    )

    # Check for rare labels (< 1%)
    rare_threshold = len(df) * 0.01
    rare_contexts = context_counts[context_counts < rare_threshold]
    print(
        f"  Rare Categories (<1% freq): {len(rare_contexts)} out of {len(context_counts)}"
    )
    print("")

    # 4. Feature/Signal Relationships
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # Construct Meta-Features for Relationship Analysis
    # 1. Length Difference (abs)
    df["len_diff"] = (df["anchor_char_len"] - df["target_char_len"]).abs()

    # 2. Jaccard Similarity & Common Words
    def get_jaccard_and_overlap(row):
        set_a = set(str(row["anchor"]).lower().split())
        set_b = set(str(row["target"]).lower().split())
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        jaccard = intersection / union if union > 0 else 0.0
        return pd.Series([intersection, jaccard])

    df[["common_words", "jaccard_sim"]] = df.apply(get_jaccard_and_overlap, axis=1)

    # 3. Context Frequency Encoding (to treat as numeric for correlation)
    context_freq = df["context"].value_counts().to_dict()
    df["context_freq"] = df["context"].map(context_freq)

    # Correlation Analysis
    numeric_feats = [
        "anchor_char_len",
        "target_char_len",
        "len_diff",
        "common_words",
        "jaccard_sim",
        "context_freq",
    ]
    correlations = (
        df[numeric_feats + ["score"]].corr(method="pearson")["score"].drop("score")
    )

    print("Correlation with Target (Score):")
    for feat, corr in correlations.sort_values(ascending=False).items():
        print(f"  {feat}: {corr:.4f}")

    # Feature Importance (Random Forest)
    # We use the meta-features to see which structural property explains the score best
    X = df[numeric_feats].fillna(0)
    y = df["score"]

    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=numeric_feats).sort_values(
        ascending=False
    )

    print("\nTop 5 Meta-Feature Importance (Random Forest):")
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    # Redundancy Check
    print("\nRedundancy Check (Correlation > 0.90):")
    corr_matrix = df[numeric_feats].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

    if high_corr:
        for col in high_corr:
            correlated_cols = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} is highly correlated with: {correlated_cols}")
    else:
        print("  No highly collinear meta-features found.")


if __name__ == "__main__":
    perform_eda()
