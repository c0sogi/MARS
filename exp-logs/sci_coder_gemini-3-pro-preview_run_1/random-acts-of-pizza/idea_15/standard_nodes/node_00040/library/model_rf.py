import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from library import config
from library import data_loader
from library import features


def get_assembled_features(load_cached_data=True):
    """
    Orchestrates the generation, loading, and assembly of all feature components
    for the Random Forest model. Implements caching for the final assembled matrices.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    cache_file = os.path.join(config.WORKING_DIR, "rf_features_assembled.npz")

    if load_cached_data and os.path.exists(cache_file):
        print("Loading assembled RF features from cache...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["test_ids"],
        )

    print("Assembling RF features from scratch...")

    # 1. Load Dataframes
    train_df, val_df, test_df = data_loader.load_datasets(
        load_cached_data=load_cached_data
    )

    # Extract Targets and IDs
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values
    test_ids = test_df["request_id"].values

    # 2. Initialize Feature Engineer
    fe = features.FeatureEngineer()

    # 3. Generate Component Features

    # A. Metadata (Numerical + Ratios + Arcsinh)
    print("Generating Metadata...")
    meta_train = fe.generate_metadata_features(train_df, "train", load_cached_data)
    meta_val = fe.generate_metadata_features(val_df, "val", load_cached_data)
    meta_test = fe.generate_metadata_features(test_df, "test", load_cached_data)

    # Select only numeric columns from metadata
    # We exclude object columns if any slipped in, though generate_metadata_features mostly returns numerics
    numeric_cols = meta_train.select_dtypes(include=[np.number]).columns.tolist()

    # Convert to numpy and Impute NaNs (Median)
    imputer = SimpleImputer(strategy="median")
    X_meta_train = imputer.fit_transform(meta_train[numeric_cols])
    X_meta_val = imputer.transform(meta_val[numeric_cols])
    X_meta_test = imputer.transform(meta_test[numeric_cols])

    # B. Dual-Lexical TF-IDF
    print("Generating TF-IDF Features...")
    (
        tfidf_title_train,
        tfidf_body_train,
        tfidf_title_val,
        tfidf_body_val,
        tfidf_title_test,
        tfidf_body_test,
    ) = fe.get_tfidf_features(train_df, val_df, test_df, load_cached_data)

    # 4. Concatenate All Features
    # Removed Semantic Anchors (Cite Lesson 39)
    print("Concatenating features...")
    X_train = np.hstack([X_meta_train, tfidf_title_train, tfidf_body_train])
    X_val = np.hstack([X_meta_val, tfidf_title_val, tfidf_body_val])
    X_test = np.hstack([X_meta_test, tfidf_title_test, tfidf_body_test])

    # 5. Cache Result
    print(f"Saving assembled features to {cache_file}...")
    np.savez(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        test_ids=test_ids,
    )

    return X_train, y_train, X_val, y_val, X_test, test_ids


def train_rf_stream(load_cached_data=True):
    """
    Main function to train the Semantic Anchor Random Forest stream.

    Args:
        load_cached_data (bool): Whether to use cached data/features.

    Returns:
        tuple: (trained_model, val_probs, test_probs)
    """
    # 1. Get Data
    X_train, y_train, X_val, y_val, X_test, test_ids = get_assembled_features(
        load_cached_data
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # 2. Initialize Model
    rf = RandomForestClassifier(**config.RF_PARAMS)

    # 3. Train
    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating on Validation Set...")
    val_probs = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"Random Forest Validation AUC: {val_auc}")

    # 5. Predict on Test
    print("Generating Test Predictions...")
    test_probs = rf.predict_proba(X_test)[:, 1]

    return rf, val_probs, test_probs
