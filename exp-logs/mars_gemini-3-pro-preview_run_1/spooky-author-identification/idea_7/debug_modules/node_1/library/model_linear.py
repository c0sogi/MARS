import os
import gc
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.dataset import create_folds, get_test_dataset


def run_expert_b(load_cached_data=True, debug=False):
    """
    Main entry point for Expert B (Surface Stylometric - TF-IDF + Logistic Regression).

    Strategy:
    1. Checks cache for OOF and Test predictions.
    2. If not found, runs 5-Fold CV.
    3. Features: Concatenation of Word N-grams and Character N-grams (Sparse).
    4. Model: Logistic Regression.
    5. Generates and saves OOF and Test predictions (bagged).

    Args:
        load_cached_data (bool): Whether to load predictions from cache if available.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        tuple: (oof_preds, test_preds)
    """
    seed_everything(Config.SEED)

    # Define cache paths (adjust for debug)
    oof_cache = Config.CACHE_EXPERT_B_OOF
    test_cache = Config.CACHE_EXPERT_B_TEST

    if debug:
        oof_cache = oof_cache.replace(".npy", "_debug.npy")
        test_cache = test_cache.replace(".npy", "_debug.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(oof_cache) and os.path.exists(test_cache):
        print("[Expert B] Loading cached predictions...")
        oof_preds = np.load(oof_cache)
        test_preds = np.load(test_cache)
        return oof_preds, test_preds

    print("[Expert B] Cache not found or reload requested. Starting training...")

    # 2. Prepare Data
    # Load folds (returns DataFrame with 'text', 'author', 'fold')
    df_folds = create_folds(load_cached_data=True, debug=debug)

    # Load test data
    # get_test_dataset returns (dataset, ids). dataset.texts contains the raw strings.
    test_dataset, _ = get_test_dataset(debug=debug)
    test_texts = test_dataset.texts

    # Initialize containers
    # Shape: (N_samples, 3) for 3 classes
    oof_preds = np.zeros((len(df_folds), 3))
    test_preds_accum = []

    # Map string labels to integers for internal consistency if needed,
    # though we usually use the target column directly if mapped or map it here.
    # Config.LABEL2ID provides the mapping.
    y_all = df_folds["author"].map(Config.LABEL2ID).values

    # 3. K-Fold Loop
    for fold in range(Config.N_FOLDS):
        print(f"\n[Expert B] Training Fold {fold}...")

        # Split Data
        train_mask = df_folds["fold"] != fold
        val_mask = df_folds["fold"] == fold

        X_train_text = df_folds.loc[train_mask, "text"].values.astype(str)
        y_train = y_all[train_mask]

        X_val_text = df_folds.loc[val_mask, "text"].values.astype(str)
        val_indices = df_folds.index[val_mask].values

        # ----------------------------------------------------------------------
        # Feature Engineering (TF-IDF)
        # ----------------------------------------------------------------------
        # Word N-grams
        word_vectorizer = TfidfVectorizer(
            ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
            min_df=Config.TFIDF_MIN_DF,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )

        # Character N-grams
        char_vectorizer = TfidfVectorizer(
            ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
            min_df=Config.TFIDF_MIN_DF,
            strip_accents="unicode",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            analyzer="char",
        )

        # Fit on Train
        print(f"  Fitting vectorizers on {len(X_train_text)} documents...")
        X_train_word = word_vectorizer.fit_transform(X_train_text)
        X_train_char = char_vectorizer.fit_transform(X_train_text)

        # Transform Val and Test
        X_val_word = word_vectorizer.transform(X_val_text)
        X_val_char = char_vectorizer.transform(X_val_text)

        X_test_word = word_vectorizer.transform(test_texts)
        X_test_char = char_vectorizer.transform(test_texts)

        # Stack Features (Sparse)
        X_train_feats = hstack([X_train_word, X_train_char])
        X_val_feats = hstack([X_val_word, X_val_char])
        X_test_feats = hstack([X_test_word, X_test_char])

        print(f"  Feature shape: {X_train_feats.shape}")

        # ----------------------------------------------------------------------
        # Modeling (Logistic Regression)
        # ----------------------------------------------------------------------
        clf = LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            C=1.0,
            random_state=Config.SEED,
            n_jobs=-1,
            max_iter=1000,  # Ensure convergence
        )

        clf.fit(X_train_feats, y_train)

        # Predict Validation
        val_probs = clf.predict_proba(X_val_feats)
        oof_preds[val_indices] = val_probs

        # Predict Test
        test_probs = clf.predict_proba(X_test_feats)
        test_preds_accum.append(test_probs)

        # Evaluate Fold
        # Get true labels for this fold
        y_val_true = y_all[val_mask]
        fold_score = compute_log_loss(y_val_true, val_probs)
        print(f"  Fold {fold} Log Loss: {fold_score}")

        # Cleanup to save memory
        del word_vectorizer, char_vectorizer, clf
        del X_train_word, X_train_char, X_train_feats
        del X_val_word, X_val_char, X_val_feats
        del X_test_word, X_test_char, X_test_feats
        gc.collect()

    # 4. Aggregate Results
    # Average Test Predictions (Bagging)
    test_preds_avg = np.mean(test_preds_accum, axis=0)

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(oof_cache), exist_ok=True)

    np.save(oof_cache, oof_preds)
    np.save(test_cache, test_preds_avg)

    # Final Score
    total_score = compute_log_loss(df_folds["author"].values, oof_preds)
    print(f"\n[Expert B] Overall OOF Log Loss: {total_score}")
    print(f"[Expert B] Predictions saved to {oof_cache} and {test_cache}")

    return oof_preds, test_preds_avg
