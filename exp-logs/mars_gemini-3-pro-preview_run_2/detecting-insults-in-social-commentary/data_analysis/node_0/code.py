import pandas as pd
import numpy as np
import os
import sys
import warnings
import random
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Configuration
METADATA_PATH = "./metadata/train.csv"
SEED = 42

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    return pd.read_csv(METADATA_PATH)


def clean_text(text):
    """
    Cleans the text column.
    The description says: unicode-escaped text, surrounded by double-quotes.
    Example: "You are an idiot."
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes
    try:
        # If it starts and ends with quotes, it might be a string literal
        if text.startswith('"') and text.endswith('"'):
            # This handles escaped characters like \n, \xe2, etc.
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Basic unicode unescape if needed, though ast.literal_eval usually handles it
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except:
        pass

    return text


def parse_date(date_str):
    """
    Parses date format: YYYYMMDDhhmmssZ
    Returns datetime object or NaT
    """
    if pd.isna(date_str) or date_str == "":
        return pd.NaT

    try:
        # Remove 'Z' at the end
        clean_str = str(date_str).replace("Z", "")
        return datetime.strptime(clean_str, "%Y%m%d%H%M%S")
    except ValueError:
        return pd.NaT


def analyze_target(df):
    print("=" * 30)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 30)

    target_col = "Insult"
    counts = df[target_col].value_counts()
    props = df[target_col].value_counts(normalize=True)

    print(f"Target: {target_col}")
    print(f"Class 0 (Neutral):   {counts.get(0, 0)} ({props.get(0, 0):.4f})")
    print(f"Class 1 (Insulting): {counts.get(1, 0)} ({props.get(1, 0):.4f})")

    ratio = counts.get(0, 0) / max(1, counts.get(1, 0))
    print(f"Imbalance Ratio (0:1): {ratio:.4f}")
    print("")


def analyze_text_modality(df):
    print("=" * 30)
    print("INPUT DATA ANALYSIS (TEXT)")
    print("=" * 30)

    # 1. Clean Text
    # We create a temporary list to avoid modifying the original dataframe slice warnings
    cleaned_texts = df["Comment"].apply(clean_text).tolist()

    # 2. Length Analysis
    char_lens = [len(t) for t in cleaned_texts]
    word_lens = [len(t.split()) for t in cleaned_texts]

    print("Sequence Lengths (Characters):")
    print(f"  Mean: {np.mean(char_lens):.4f}")
    print(f"  Std:  {np.std(char_lens):.4f}")
    print(f"  Min:  {np.min(char_lens):.4f}")
    print(f"  Max:  {np.max(char_lens):.4f}")

    print("Sequence Lengths (Words):")
    print(f"  Mean: {np.mean(word_lens):.4f}")
    print(f"  Std:  {np.std(word_lens):.4f}")
    print(f"  Min:  {np.min(word_lens):.4f}")
    print(f"  Max:  {np.max(word_lens):.4f}")

    # 3. Vocabulary Analysis
    # Use a basic CountVectorizer to estimate vocab size
    try:
        vec = CountVectorizer(min_df=2, stop_words="english")
        vec.fit(cleaned_texts)
        vocab_size = len(vec.vocabulary_)
        print(f"Vocabulary Size (min_df=2, stop_words='english'): {vocab_size}")
    except ValueError:
        print(
            "Vocabulary Size: Unable to calculate (dataset might be empty or contain only stop words)."
        )

    # Return features for relationship analysis
    return char_lens, word_lens, cleaned_texts


def analyze_relationships(df, char_lens, word_lens, cleaned_texts):
    print("=" * 30)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 30)

    # Create a feature dataframe for analysis
    features = pd.DataFrame()
    features["target"] = df["Insult"].values
    features["char_len"] = char_lens
    features["word_len"] = word_lens

    # Process Date Metadata
    # Date format: YYYYMMDDhhmmssZ
    dates = df["Date"].apply(parse_date)

    # Extract numerical features from date
    features["has_date"] = dates.notna().astype(int)
    features["hour"] = dates.apply(lambda x: x.hour if pd.notna(x) else -1)
    features["day_of_week"] = dates.apply(lambda x: x.weekday() if pd.notna(x) else -1)

    # 1. Meta-Feature Correlations
    print("Meta-Feature Correlations (Pearson with Target):")
    corr = features.corr()["target"].drop("target")
    for name, val in corr.items():
        print(f"  {name}: {val:.4f}")

    # 2. Feature Importance (Structured/Meta Features)
    print("\nMeta-Feature Importance (Random Forest):")
    X_meta = features.drop(columns=["target"])
    y = features["target"]

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X_meta, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(min(5, len(X_meta.columns))):
        feat_name = X_meta.columns[indices[i]]
        score = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {score:.4f}")

    # 3. Unstructured Relationships (Text Content)
    # Check if longer comments are more likely to be insults
    print("\nUnstructured Relationships (Length vs Class):")
    mean_len_0 = features[features["target"] == 0]["char_len"].mean()
    mean_len_1 = features[features["target"] == 1]["char_len"].mean()
    print(f"  Avg Char Length (Neutral):   {mean_len_0:.4f}")
    print(f"  Avg Char Length (Insulting): {mean_len_1:.4f}")

    # Top words for Insulting class (simple frequency diff)
    print("\nTop Indicative Words (TF-IDF approach):")
    # We use a small vectorizer to find words with highest coefficient in a linear model
    # or just highest difference in frequency. Let's use a quick Logistic Regression on TF-IDF
    # as a proxy for 'importance' in text.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        X_text = tfidf.fit_transform(cleaned_texts)
        clf = LogisticRegression(random_state=SEED, solver="liblinear")
        clf.fit(X_text, y)

        # Get coefficients
        coefs = clf.coef_[0]
        top_indices = np.argsort(coefs)[::-1][
            :10
        ]  # Top 10 positive coefficients (Insult)
        feature_names = np.array(tfidf.get_feature_names_out())

        print("  Top words associated with 'Insult':")
        print(f"  {', '.join(feature_names[top_indices])}")

    except Exception as e:
        print(f"  Could not extract top words: {e}")


def main():
    set_seed(SEED)

    # Load
    df = load_data()

    # Target Analysis
    analyze_target(df)

    # Modality Specific Analysis (Text)
    char_lens, word_lens, cleaned_texts = analyze_text_modality(df)

    # Feature Relationships
    analyze_relationships(df, char_lens, word_lens, cleaned_texts)


if __name__ == "__main__":
    main()
