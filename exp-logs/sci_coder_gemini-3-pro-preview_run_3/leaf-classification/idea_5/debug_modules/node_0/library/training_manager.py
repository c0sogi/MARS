import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything, save_pickle, load_pickle, clip_probabilities
from library.feature_extractor import extract_features
from library.data_loader import load_tabular_data
from library.ensemble_model import PipelineMember


def train_bagging_ensemble(load_cached_data=True, debug=False):
    """
    Orchestrates the training of the Bagging Ensemble using Stratified K-Fold.
    Combines Train and Validation metadata to maximize training data for the ensemble.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.
        debug (bool): If True, runs on a subset of data with reduced splits.
    """
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Load and Merge Data
    # ==========================================
    print("Loading training features...")
    train_shape, train_texture, train_labels = extract_features(
        "train", load_cached_data, debug
    )
    _, train_tabular, _ = load_tabular_data(Config.TRAIN_METADATA_PATH, debug)

    print("Loading validation features...")
    val_shape, val_texture, val_labels = extract_features(
        "val", load_cached_data, debug
    )
    _, val_tabular, _ = load_tabular_data(Config.VAL_METADATA_PATH, debug)

    # Concatenate Train and Val sets for Cross-Validation
    X_shape = np.concatenate([train_shape, val_shape], axis=0)
    X_texture = np.concatenate([train_texture, val_texture], axis=0)
    X_tabular = np.concatenate([train_tabular, val_tabular], axis=0)
    y = np.concatenate([train_labels, val_labels], axis=0)

    print(f"Combined Training Data Shape: {X_shape.shape[0]} samples")

    # ==========================================
    # 2. Configure Cross-Validation
    # ==========================================
    if debug:
        # Use KFold with fewer splits for debug to avoid stratification errors on small subsets
        n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        print(f"Debug Mode: Using KFold with {n_splits} splits.")
    else:
        n_splits = Config.N_FOLDS
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        print(f"Standard Mode: Using StratifiedKFold with {n_splits} splits.")

    fold_scores = []

    # ==========================================
    # 3. Training Loop
    # ==========================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_shape, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split Data
        X_shape_tr, X_shape_val = X_shape[train_idx], X_shape[val_idx]
        X_texture_tr, X_texture_val = X_texture[train_idx], X_texture[val_idx]
        X_tabular_tr, X_tabular_val = X_tabular[train_idx], X_tabular[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Initialize Pipeline
        model = PipelineMember()

        # Fit Pipeline
        model.fit(X_shape_tr, X_texture_tr, X_tabular_tr, y_tr)

        # Predict on Validation Split
        val_probs = model.predict_proba(X_shape_val, X_texture_val, X_tabular_val)

        # Clip probabilities for metric stability
        val_probs_clipped = clip_probabilities(val_probs)

        # Calculate Log Loss
        # We explicitly pass labels to ensure correct column mapping for string targets
        fold_loss = log_loss(y_val, val_probs_clipped, labels=model.lda.classes_)
        fold_scores.append(fold_loss)

        print(f"Fold {fold + 1} Log Loss: {fold_loss}")

        # Save Model Artifact
        model_path = Config.PIPELINE_FILENAME_TEMPLATE.format(fold)
        save_pickle(model, model_path)
        print(f"Model saved to {model_path}")

    # ==========================================
    # 4. Summary
    # ==========================================
    avg_loss = np.mean(fold_scores)
    print("\n========================================")
    print(f"Training Complete.")
    print(f"Average CV Log Loss: {avg_loss}")
    print("========================================")


def predict_ensemble(load_cached_data=True, debug=False):
    """
    Generates predictions for the test set using the trained bagging ensemble.
    Aggregates predictions via arithmetic mean and saves submission CSV.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.
        debug (bool): If True, runs on a subset of data.
    """
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Load Test Data
    # ==========================================
    print("Loading test data...")
    test_shape, test_texture, test_ids_feats = extract_features(
        "test", load_cached_data, debug
    )
    test_ids_tabular, test_tabular, _ = load_tabular_data(
        Config.TEST_METADATA_PATH, debug
    )

    # Verify ID alignment between features and tabular data
    if not np.array_equal(test_ids_feats, test_ids_tabular):
        raise ValueError(
            "Critical Error: Mismatch between feature IDs and tabular IDs in test set."
        )

    # ==========================================
    # 2. Ensemble Inference
    # ==========================================
    avg_probs = None
    classes = None
    models_loaded = 0

    print("Starting Ensemble Inference...")

    # Iterate over all potential folds
    for fold in range(Config.N_FOLDS):
        model_path = Config.PIPELINE_FILENAME_TEMPLATE.format(fold)

        # Load model
        model = load_pickle(model_path)

        if model is None:
            # Skip missing models (e.g., if debug training used fewer folds)
            continue

        models_loaded += 1

        # Predict
        probs = model.predict_proba(test_shape, test_texture, test_tabular)

        # Accumulate
        if avg_probs is None:
            avg_probs = probs
            classes = model.lda.classes_
        else:
            avg_probs += probs

            # Verify class alignment across models
            if not np.array_equal(classes, model.lda.classes_):
                raise RuntimeError(f"Class mismatch in model fold {fold}.")

    if models_loaded == 0:
        raise RuntimeError("No trained models found. Please run training first.")

    print(f"Aggregated predictions from {models_loaded} models.")

    # Compute Arithmetic Mean
    avg_probs /= models_loaded

    # Clip Probabilities
    avg_probs = clip_probabilities(avg_probs)

    # ==========================================
    # 3. Save Submission
    # ==========================================
    # Create DataFrame with ID and Class columns
    df_sub = pd.DataFrame(avg_probs, columns=classes)
    df_sub.insert(0, "id", test_ids_tabular)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
