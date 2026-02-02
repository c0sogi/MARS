import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.data import get_folds


def preprocess_text(text):
    """
    Basic text preprocessing: whitespace normalization.
    """
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def get_tfidf_features(df_train, df_val, df_test, load_cached_data=True):
    """
    Generates or loads TF-IDF features for train, val, and test sets.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "tfidf_features.joblib")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached TF-IDF features from {cache_path}")
        try:
            data = joblib.load(cache_path)
            return data["train_tfidf"], data["val_tfidf"], data["test_tfidf"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Fitting TfidfVectorizer...")
    # Initialize Vectorizer with Config parameters
    vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

    # Fit on Train only
    train_text = df_train["full_text"].apply(preprocess_text)
    vectorizer.fit(train_text)

    # Transform all splits
    print("Transforming text data...")
    train_tfidf = vectorizer.transform(train_text)
    val_tfidf = vectorizer.transform(df_val["full_text"].apply(preprocess_text))
    test_tfidf = vectorizer.transform(df_test["full_text"].apply(preprocess_text))

    # Cache results
    if load_cached_data:
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        try:
            joblib.dump(
                {
                    "train_tfidf": train_tfidf,
                    "val_tfidf": val_tfidf,
                    "test_tfidf": test_tfidf,
                },
                cache_path,
            )
            print(f"Saved TF-IDF features to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache. {e}")

    return train_tfidf, val_tfidf, test_tfidf


def run_tfidf_ridge(load_cached_data=True, debug=False):
    """
    Runs the Lexical Branch pipeline: TF-IDF + Ridge Regression.

    Args:
        load_cached_data (bool): Whether to use cached features.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        tuple: (df_train_with_oof, df_val_with_pred, df_test_with_pred)
    """
    seed_everything(Config.SEED)

    # --- 1. Load Data ---
    print("Loading data...")
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    if debug:
        print(f"Debug mode: truncating data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)

    # --- 2. Feature Extraction ---
    X_train, X_val, X_test = get_tfidf_features(
        df_train, df_val, df_test, load_cached_data
    )
    y_train = df_train["score"].values

    # --- 3. Cross-Validation (OOF Generation) ---
    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    # Assign folds
    df_train = get_folds(df_train, n_folds=Config.N_FOLDS, seed=Config.SEED)

    oof_preds = np.zeros(len(df_train))

    for fold in range(Config.N_FOLDS):
        train_idx = df_train["fold"] != fold
        valid_idx = df_train["fold"] == fold

        X_tr_fold = X_train[train_idx]
        y_tr_fold = y_train[train_idx]
        X_va_fold = X_train[valid_idx]
        y_va_fold = y_train[valid_idx]

        # Train Ridge
        # Using lsqr solver for sparse data efficiency
        model = Ridge(solver="lsqr", fit_intercept=True, random_state=Config.SEED)
        model.fit(X_tr_fold, y_tr_fold)

        # Predict
        preds = model.predict(X_va_fold)

        # Clip predictions to valid range [1, 6] for safety, though regression can go outside
        preds = np.clip(preds, 1, 6)

        oof_preds[valid_idx] = preds

        # Fold Metric
        # Round to integers for QWK calculation
        fold_score = compute_qwk(y_va_fold, np.round(preds).astype(int))
        print(f"Fold {fold} QWK: {fold_score}")

    # Overall OOF Metric
    oof_qwk = compute_qwk(y_train, np.round(oof_preds).astype(int))
    oof_mse = mean_squared_error(y_train, oof_preds)
    print(f"Overall OOF QWK: {oof_qwk}")
    print(f"Overall OOF MSE: {oof_mse}")

    df_train["ridge_pred"] = oof_preds

    # --- 4. Final Training & Prediction ---
    print("Training final model on full train set...")
    final_model = Ridge(solver="lsqr", fit_intercept=True, random_state=Config.SEED)
    final_model.fit(X_train, y_train)

    print("Predicting on Validation and Test sets...")
    val_preds = final_model.predict(X_val)
    val_preds = np.clip(val_preds, 1, 6)

    test_preds = final_model.predict(X_test)
    test_preds = np.clip(test_preds, 1, 6)

    # Validation Metric
    if "score" in df_val.columns:
        val_qwk = compute_qwk(df_val["score"].values, np.round(val_preds).astype(int))
        print(f"Validation Set QWK: {val_qwk}")

    df_val["ridge_pred"] = val_preds
    df_test["ridge_pred"] = test_preds

    # Return dataframes with predictions
    return df_train, df_val, df_test
