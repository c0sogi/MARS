import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import random

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_DATA_FILE = "simplified-nq-train.jsonl"
TRAIN_META_FILE = "train.parquet"
SAMPLE_SIZE = 5000  # Number of samples for heavy text analysis
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_metadata():
    path = os.path.join(METADATA_DIR, TRAIN_META_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_parquet(path)


def parse_annotations(row):
    """
    Parses the annotations JSON string to determine answer type.
    Returns:
        - has_long_answer (bool)
        - has_short_answer (bool)
        - is_yes_no (bool)
        - answer_type (str): 'None', 'Long', 'Short', 'YesNo' (Hierarchical)
    """
    try:
        anns = json.loads(row["annotations"])
    except:
        return False, False, False, "Error"

    has_long = False
    has_short = False
    is_yes_no = False

    # In NQ, there is usually one annotation object in the list for train
    for ann in anns:
        # Long answer exists if start_token != -1
        la = ann.get("long_answer", {})
        if la.get("start_token", -1) != -1:
            has_long = True

        # Short answer exists if list is not empty
        sa = ann.get("short_answers", [])
        if sa:
            has_short = True

        # Yes/No
        yn = ann.get("yes_no_answer", "NONE")
        if yn != "NONE":
            is_yes_no = True

    if is_yes_no:
        atype = "YesNo"
    elif has_short:
        atype = "Short"
    elif has_long:
        atype = "Long"
    else:
        atype = "NoAnswer"

    return has_long, has_short, is_yes_no, atype


def analyze_targets(df):
    print("TARGET VARIABLE ANALYSIS")

    # Apply parsing
    parsed = df.apply(parse_annotations, axis=1, result_type="expand")
    parsed.columns = ["has_long", "has_short", "is_yes_no", "answer_type"]

    df = pd.concat([df, parsed], axis=1)

    # Distribution of Answer Types
    counts = df["answer_type"].value_counts()
    total = len(df)

    print(f"Total Samples: {total}")
    print("Class Balance (Answer Types Hierarchical):")
    for label, count in counts.items():
        ratio = count / total
        print(f"  {label}: {count} ({ratio:.4f})")

    # Specific boolean distributions
    long_ratio = df["has_long"].mean()
    short_ratio = df["has_short"].mean()
    yesno_ratio = df["is_yes_no"].mean()

    print("Label Frequencies (Independent):")
    print(f"  Has Long Answer: {long_ratio:.4f}")
    print(f"  Has Short Answer: {short_ratio:.4f}")
    print(f"  Has Yes/No Answer: {yesno_ratio:.4f}")

    return df


def analyze_text_data(df):
    print("\nINPUT DATA ANALYSIS (TEXT MODALITY)")

    # Sample data for heavy I/O
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        sample_df = df.copy()

    print(f"Analysis performed on a random sample of {len(sample_df)} records.")

    jsonl_path = os.path.join(INPUT_DIR, TRAIN_DATA_FILE)

    stats = []
    vocab = Counter()

    with open(jsonl_path, "rb") as f:
        for idx, row in sample_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract fields
                doc_text = data.get("document_text", "")
                q_text = data.get("question_text", "")
                candidates = data.get("long_answer_candidates", [])

                # Basic Tokenization (Whitespace)
                doc_tokens = doc_text.split()
                q_tokens = q_text.split()

                # Update Vocab (only top frequent to save memory if needed, but Counter handles it)
                vocab.update(q_tokens)

                stats.append(
                    {
                        "example_id": row["example_id"],
                        "doc_len_chars": len(doc_text),
                        "doc_len_words": len(doc_tokens),
                        "q_len_chars": len(q_text),
                        "q_len_words": len(q_tokens),
                        "num_candidates": len(candidates),
                        "has_long": row["has_long"],  # Target for correlation
                        "answer_type": row["answer_type"],
                    }
                )

            except json.JSONDecodeError:
                continue

    stats_df = pd.DataFrame(stats)

    # 1. Length Analysis
    print("Sequence Lengths (Word Counts):")
    for col in ["doc_len_words", "q_len_words"]:
        desc = stats_df[col].describe()
        print(
            f"  {col}: Mean={desc['mean']:.4f}, Std={desc['std']:.4f}, Min={desc['min']:.4f}, Max={desc['max']:.4f}"
        )

        # Outliers (IQR)
        Q1 = desc["25%"]
        Q3 = desc["75%"]
        IQR = Q3 - Q1
        outliers = stats_df[
            (stats_df[col] < (Q1 - 1.5 * IQR)) | (stats_df[col] > (Q3 + 1.5 * IQR))
        ]
        print(
            f"  {col} Outliers (IQR method): {len(outliers)} ({len(outliers)/len(stats_df):.4f})"
        )

    # 2. Vocabulary
    print("Vocabulary Analysis (Question Text Sample):")
    print(f"  Unique Tokens found: {len(vocab)}")
    # OOV potential: rough proxy is percentage of words appearing only once
    singletons = sum(1 for count in vocab.values() if count == 1)
    print(
        f"  Singleton Tokens (Potential OOV): {singletons} ({singletons/len(vocab):.4f})"
    )

    return stats_df


def analyze_relationships(stats_df):
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # 1. Correlation
    # Correlate numerical features with binary target 'has_long'
    print("Correlations with Target (Has Long Answer):")
    features = ["doc_len_words", "q_len_words", "num_candidates"]

    corr_matrix = stats_df[features + ["has_long"]].corr(method="pearson")
    target_corr = corr_matrix["has_long"].drop("has_long")

    for feat, corr in target_corr.items():
        print(f"  {feat}: {corr:.4f}")

    # Redundancy check
    print("Feature Redundancy (Correlation > 0.90):")
    redundant = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.90:
                redundant.append((features[i], features[j], c))

    if redundant:
        for f1, f2, c in redundant:
            print(f"  {f1} vs {f2}: {c:.4f}")
    else:
        print("  No highly collinear pairs found.")

    # 2. Feature Importance (Random Forest)
    print("Feature Importance (Random Forest - Predicting 'Has Long Answer'):")
    X = stats_df[features]
    y = stats_df["has_long"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(len(features)):
        print(f"  {features[indices[f]]}: {importances[indices[f]]:.4f}")

    # 3. Unstructured/Meta Relationships
    print("Meta-Feature Relationship:")
    doc_cand_corr = stats_df["doc_len_words"].corr(stats_df["num_candidates"])
    print(
        f"  Correlation between Document Length and Candidate Count: {doc_cand_corr:.4f}"
    )


def main():
    set_seed(SEED)

    try:
        # Load
        df = load_metadata()

        # Target Analysis
        df_annotated = analyze_targets(df)

        # Text Analysis (Sampled)
        stats_df = analyze_text_data(df_annotated)

        # Relationship Analysis
        analyze_relationships(stats_df)

    except Exception as e:
        print(f"An error occurred during EDA: {e}")
        exit(1)


if __name__ == "__main__":
    main()
