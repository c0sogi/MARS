import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library import config, utils, features_text, features_meta


def assemble_rf_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Assembles the specific feature set for the Random Forest model:
    - TF-IDF (Title + Body)
    - Sentiment (Title + Body)
    - Global Consistency Scalars
    - Top-K Community Indicators
    - Full-Spectrum Metadata (Raw + Ratios)

    Args:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame): Validation dataframe (can be None).
        test_df (pd.DataFrame): Test dataframe.
        load_cached_data (bool): Whether to use cached assembled features.

    Returns:
        tuple: (X_train, X_val, X_test) as numpy arrays.
    """
    cache_file = os.path.join(config.CACHE_DIR, "rf_features.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading assembled RF features from cache: {cache_file}")
        try:
            loaded = np.load(cache_file)
            X_train = loaded["X_train"]
            X_test = loaded["X_test"]
            X_val = loaded["X_val"] if "X_val" in loaded else None

            # Validate dimensions
            if len(X_train) != len(train_df):
                print(
                    f"Cache mismatch: Found {len(X_train)} samples, expected {len(train_df)}. Reassembling."
                )
            else:
                # Handle case where None was saved as a 0-d array or similar if using np.savez
                if val_df is None:
                    X_val = None
                elif X_val is not None and X_val.ndim == 0:
                    X_val = None

                return X_train, X_val, X_test
        except Exception as e:
            print(f"Failed to load RF feature cache: {e}. Reassembling...")

    print("Assembling RF features from scratch...")

    # 2. Generate Base Features
    # We rely on the library modules which handle their own caching
    text_feats = features_text.generate_text_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )
    meta_feats = features_meta.generate_meta_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Helper to Concatenate Features for a Split
    def concat_features(split_prefix):
        # List of feature arrays to concatenate
        arrays = []

        # A. TF-IDF (High Dimensionality)
        arrays.append(text_feats[f"{split_prefix}_tfidf"])

        # B. Sentiment (Title + Body) -> (N, 4) each
        arrays.append(text_feats[f"{split_prefix}_title_sentiment"])
        arrays.append(text_feats[f"{split_prefix}_body_sentiment"])

        # C. Consistency Scalars -> (N, 1) each
        arrays.append(meta_feats[f"{split_prefix}_cons_title"])
        arrays.append(meta_feats[f"{split_prefix}_cons_body"])

        # D. Top-K Indicators -> (N, K)
        arrays.append(meta_feats[f"{split_prefix}_topk"])

        # E. Metadata (RF version - imputed raw/ratios) -> (N, M)
        arrays.append(meta_feats[f"{split_prefix}_meta_rf"])

        # Horizontal Stack
        return np.hstack(arrays)

    # 4. Assemble Matrices
    X_train = concat_features("train")
    X_test = concat_features("test")

    X_val = None
    if val_df is not None:
        X_val = concat_features("val")

    # 5. Save to Cache
    print(f"Saving assembled RF features to {cache_file}...")
    save_dict = {"X_train": X_train, "X_test": X_test}
    if X_val is not None:
        save_dict["X_val"] = X_val

    np.savez(cache_file, **save_dict)

    return X_train, X_val, X_test


def train_rf(X_train, y_train, X_val=None, y_val=None):
    """
    Trains the Consistency-Augmented Top-K Random Forest.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray, optional): Validation features.
        y_val (np.ndarray, optional): Validation labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    print("Initializing Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        class_weight=config.RF_CLASS_WEIGHT,
        n_jobs=config.RF_N_JOBS,
        random_state=config.RF_RANDOM_STATE,
        verbose=0,
    )

    print(
        f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    rf.fit(X_train, y_train)

    # Evaluation
    train_probs = rf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"RF Train AUC: {train_auc}")

    if X_val is not None and y_val is not None:
        val_probs = rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        print(f"RF Val AUC: {val_auc}")

    return rf


def predict_rf(model, X):
    """
    Generates probability predictions using the trained Random Forest.

    Args:
        model (RandomForestClassifier): Trained model.
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Probability of positive class.
    """
    return model.predict_proba(X)[:, 1]
