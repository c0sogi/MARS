import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import get_score, seed_everything


def get_features(train_text, val_text, test_text, load_cached_data=True):
    """
    Generates or loads TF-IDF features for train, val, and test sets.
    Uses Word (1-2) and Char (2-6) n-grams.
    Caches results to disk using scipy.sparse.save_npz to avoid re-computation.
    """
    cache_dir = Config.TFIDF_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_feat_path = os.path.join(cache_dir, "tfidf_train.npz")
    val_feat_path = os.path.join(cache_dir, "tfidf_val.npz")
    test_feat_path = os.path.join(cache_dir, "tfidf_test.npz")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_feat_path)
        and os.path.exists(val_feat_path)
        and os.path.exists(test_feat_path)
    ):

        print(f"Loading cached TF-IDF features from {cache_dir}...")
        train_features = sparse.load_npz(train_feat_path)
        val_features = sparse.load_npz(val_feat_path)
        test_features = sparse.load_npz(test_feat_path)
        return train_features, val_features, test_features

    print("Computing TF-IDF features (this may take a while)...")

    # 1. Word Vectorizer
    print("Fitting Word Vectorizer...")
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
        ngram_range=Config.LINEAR_WORD_NGRAMS,
        max_features=Config.LINEAR_MAX_FEATURES,
    )
    # Fit on train, transform all
    train_word = word_vectorizer.fit_transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    # 2. Char Vectorizer
    print("Fitting Char Vectorizer...")
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="char",
        stop_words="english",
        ngram_range=Config.LINEAR_CHAR_NGRAMS,
        max_features=Config.LINEAR_MAX_FEATURES,
    )
    # Fit on train, transform all
    train_char = char_vectorizer.fit_transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # 3. Stack Features
    print("Stacking features...")
    train_features = sparse.hstack([train_word, train_char])
    val_features = sparse.hstack([val_word, val_char])
    test_features = sparse.hstack([test_word, test_char])

    # 4. Cache Features
    print(f"Saving features to {cache_dir}...")
    sparse.save_npz(train_feat_path, train_features)
    sparse.save_npz(val_feat_path, val_features)
    sparse.save_npz(test_feat_path, test_features)

    return train_features, val_features, test_features


def train_linear_pipeline(load_cached_data=True):
    """
    Main pipeline for the Linear Baseline model.
    1. Loads data.
    2. Extracts/Loads TF-IDF features.
    3. Trains Logistic Regression for each class.
    4. Evaluates and returns predictions.
    """
    seed_everything(Config.SEED)

    print("=== Starting Linear Model Pipeline ===")

    # 1. Load Data
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Handle missing values in text
    train_text = train_df["comment_text"].fillna("").values
    val_text = val_df["comment_text"].fillna("").values
    test_text = test_df["comment_text"].fillna("").values

    # 2. Feature Extraction
    train_features, val_features, test_features = get_features(
        train_text, val_text, test_text, load_cached_data=load_cached_data
    )

    print(f"Feature Shape: {train_features.shape}")

    # 3. Training
    print("Training Logistic Regression models...")

    # Initialize prediction arrays
    val_preds = np.zeros((len(val_df), len(Config.LABEL_COLS)))
    test_preds = np.zeros((len(test_df), len(Config.LABEL_COLS)))

    scores = []

    for i, label in enumerate(Config.LABEL_COLS):
        print(f"Training for label: {label}")
        y_train = train_df[label].values
        y_val = val_df[label].values

        # Logistic Regression
        # using 'sag' solver which is faster for large datasets
        model = LogisticRegression(
            solver="sag", n_jobs=-1, C=1.0, random_state=Config.SEED, max_iter=200
        )

        model.fit(train_features, y_train)

        # Predict
        # proba returns [prob_0, prob_1], we want prob_1
        p_val = model.predict_proba(val_features)[:, 1]
        p_test = model.predict_proba(test_features)[:, 1]

        val_preds[:, i] = p_val
        test_preds[:, i] = p_test

        # Evaluate individual column
        try:
            auc = roc_auc_score(y_val, p_val)
            print(f"  AUC ({label}): {auc}")
            scores.append(auc)
        except ValueError:
            print(f"  AUC ({label}): Skipped (only one class present)")

    # 4. Overall Evaluation
    mean_auc = np.mean(scores) if scores else 0.0
    print(f"\nLinear Model Mean Column-wise ROC AUC: {mean_auc}")

    # Verify using the utility function to ensure consistency
    val_labels = val_df[Config.LABEL_COLS].values
    utility_score = get_score(val_labels, val_preds)
    print(f"Utility Function Score: {utility_score}")

    return val_preds, test_preds
