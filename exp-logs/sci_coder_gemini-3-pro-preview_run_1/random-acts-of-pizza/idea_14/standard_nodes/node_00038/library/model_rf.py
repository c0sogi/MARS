import os
import numpy as np
import pandas as pd
import scipy.sparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

from library import config
from library import data_loader
from library.feature_engineering import FeatureEngineer


def assemble_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates, imputes, and assembles features for the Random Forest.
    Handles caching of the final sparse matrices.
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "rf_X_train_sparse_only.npz"),
        "y_train": os.path.join(cache_dir, "rf_y_train.npy"),
        "X_val": os.path.join(cache_dir, "rf_X_val_sparse_only.npz"),
        "y_val": os.path.join(cache_dir, "rf_y_val.npy"),
        "X_test": os.path.join(cache_dir, "rf_X_test_sparse_only.npz"),
        "feature_names": os.path.join(cache_dir, "rf_feature_names_sparse_only.joblib"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(p) for p in paths.values()):
        print("Loading assembled RF features from cache...")
        X_train = scipy.sparse.load_npz(paths["X_train"])

        # Validate cache against current data (Cite debug_lesson_1)
        if X_train.shape[0] != len(train_df):
            print(
                f"RF Feature cache mismatch (Cache: {X_train.shape[0]}, Current: {len(train_df)}). Re-assembling..."
            )
        else:
            y_train = np.load(paths["y_train"])
            X_val = scipy.sparse.load_npz(paths["X_val"])
            y_val = np.load(paths["y_val"])
            X_test = scipy.sparse.load_npz(paths["X_test"])
            feature_names = joblib.load(paths["feature_names"])
            return X_train, y_train, X_val, y_val, X_test, feature_names

    print("Assembling features from scratch...")

    # Initialize Feature Engineer
    fe = FeatureEngineer()

    # 1. Generate Metadata (Dense)
    print("Generating Metadata...")
    meta_train = fe.generate_metadata_features(train_df)
    meta_val = fe.generate_metadata_features(val_df)
    meta_test = fe.generate_metadata_features(test_df)

    # 2. Generate Action Profiles (Dense)
    # Skipped for RF to avoid noise from dense embeddings (Cite solution_lesson_node_00006)
    # print("Generating Zero-Shot Action Profiles...")
    # prof_train = fe.generate_zero_shot_profiles(train_df, "train", load_cached_data)
    # prof_val = fe.generate_zero_shot_profiles(val_df, "val", load_cached_data)
    # prof_test = fe.generate_zero_shot_profiles(test_df, "test", load_cached_data)

    # 3. Generate TF-IDF (Sparse)
    print("Generating Dual-Lexical TF-IDF...")
    tfidf_train, tfidf_val, tfidf_test = fe.generate_tfidf_features(
        train_df, val_df, test_df, load_cached_data
    )

    # 4. Impute Dense Features
    # Using only Metadata for RF (Cite solution_lesson_node_00006)
    dense_train = meta_train
    dense_val = meta_val
    dense_test = meta_test

    dense_feature_names = dense_train.columns.tolist()

    print("Imputing dense features (Median)...")
    imputer = SimpleImputer(strategy="median")
    dense_train_imp = imputer.fit_transform(dense_train)
    dense_val_imp = imputer.transform(dense_val)
    dense_test_imp = imputer.transform(dense_test)

    # 5. Assemble Final Matrices (Stack Dense + Sparse)
    print("Stacking dense and sparse features...")
    X_train = scipy.sparse.hstack([dense_train_imp, tfidf_train]).tocsr()
    X_val = scipy.sparse.hstack([dense_val_imp, tfidf_val]).tocsr()
    X_test = scipy.sparse.hstack([dense_test_imp, tfidf_test]).tocsr()

    # Extract Targets
    target_col = "requester_received_pizza"
    y_train = train_df[target_col].astype(int).values
    y_val = val_df[target_col].astype(int).values

    # Feature Names (Dense + TFIDF placeholder)
    # Note: TF-IDF feature names are huge, we might not save them all for simplicity,
    # but keeping dense names is useful for importance analysis.
    feature_names = dense_feature_names + [
        f"tfidf_{i}" for i in range(tfidf_train.shape[1])
    ]

    # Save to Cache
    print("Saving assembled features to cache...")
    scipy.sparse.save_npz(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    scipy.sparse.save_npz(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    scipy.sparse.save_npz(paths["X_test"], X_test)
    joblib.dump(feature_names, paths["feature_names"])

    return X_train, y_train, X_val, y_val, X_test, feature_names


def train_rf_model(load_cached_data=True, debug=False):
    """
    Main function to train the Action-Profiled Random Forest.

    Args:
        load_cached_data (bool): Whether to use cached intermediate files.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        dict: Contains 'model', 'val_probs', 'test_probs', 'auc'.
    """
    print("=" * 40)
    print("Stream A: Action-Profiled Random Forest")
    print("=" * 40)

    # 1. Load Data
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Assemble Features
    X_train, y_train, X_val, y_val, X_test, feature_names = assemble_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Initialize Model
    rf_params = config.RF_PARAMS.copy()
    print(f"Initializing Random Forest with params: {rf_params}")
    clf = RandomForestClassifier(**rf_params)

    # 4. Train
    print("Training Random Forest...")
    clf.fit(X_train, y_train)

    # 5. Evaluate on Validation
    print("Evaluating on Validation Set...")
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Validation AUC: {val_auc}")

    # 6. Predict on Test
    print("Generating Test Predictions...")
    test_probs = clf.predict_proba(X_test)[:, 1]

    # Optional: Feature Importance
    if hasattr(clf, "feature_importances_"):
        # Print top 10 dense features (indices 0 to len(dense_names))
        # We need to know how many dense features there are.
        # Based on assemble_features, dense cols come first.
        # We can infer from the feature_names list.
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop 10 Features:")
        for i in range(10):
            idx = indices[i]
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {importances[idx]}")
            else:
                print(f"  Feature index {idx}: {importances[idx]}")

    return {
        "model": clf,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "auc": val_auc,
        "y_val": y_val,
    }
