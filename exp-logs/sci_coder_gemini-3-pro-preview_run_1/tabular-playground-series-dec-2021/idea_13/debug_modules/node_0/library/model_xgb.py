import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from library.utils import seed_everything, get_device
from library.data_processing import DataProcessor


def run_xgb_cv(load_cached_data=True, n_splits=5, seed=42):
    """
    Executes the Stratified 5-Fold Cross-Validation for the XGBoost backbone.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        n_splits (int): Number of CV folds.
        seed (int): Random seed.

    Returns:
        oof_preds (np.ndarray): Out-of-fold probability predictions.
        test_preds (np.ndarray): Averaged test set probability predictions.
        le (LabelEncoder): The label encoder used for targets.
        y_full (np.ndarray): The full target array (aligned with oof_preds).
    """
    seed_everything(seed)

    # --- Data Loading ---
    print("Initializing DataProcessor for XGBoost...")
    processor = DataProcessor()
    X_train_part, y_train_part, X_val_part, y_val_part, X_test, le = (
        processor.get_xgb_data(load_cached_data=load_cached_data)
    )

    # Combine provided train and val splits to form the full training set for CV
    # The metadata split (80/20) is ignored in favor of a full 5-fold CV strategy on the entire labeled dataset
    print("Combining initial Train/Val splits for full Stratified 5-Fold CV...")
    X_full = pd.concat([X_train_part, X_val_part], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    print(f"Full Training Shape: {X_full.shape}")
    print(f"Test Shape: {X_test.shape}")

    # --- Setup ---
    num_classes = len(le.classes_)
    oof_preds = np.zeros((len(X_full), num_classes), dtype=np.float32)
    test_preds_sum = np.zeros((len(X_test), num_classes), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # XGBoost Configuration
    device = get_device()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    # Note: 'multi:softprob' allows us to get probabilities directly.
    # We use 'mlogloss' for early stopping monitoring.
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device_type,
        "max_depth": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "n_jobs": 12,
        "random_state": seed,
        "verbosity": 0,
    }

    print(f"Starting XGBoost Training on {device_type}...")

    fold_scores = []

    # --- CV Loop ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split Data
        X_tr, y_tr = X_full.iloc[train_idx], y_full[train_idx]
        X_va, y_va = X_full.iloc[val_idx], y_full[val_idx]

        # Initialize Classifier
        # n_estimators is high, controlled by early_stopping_rounds
        clf = xgb.XGBClassifier(
            n_estimators=5000, early_stopping_rounds=50, **xgb_params
        )

        # Train
        # verbose=False suppresses per-round logs, we print final metrics manually
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # Validation Inference
        val_probs = clf.predict_proba(X_va)
        oof_preds[val_idx] = val_probs

        # Metrics
        val_pred_labels = np.argmax(val_probs, axis=1)
        acc = accuracy_score(y_va, val_pred_labels)
        ll = log_loss(y_va, val_probs)

        print(f"Fold {fold+1} Best Iteration: {clf.best_iteration}")
        print(f"Fold {fold+1} Log Loss: {ll}")
        print(f"Fold {fold+1} Accuracy: {acc}")
        fold_scores.append(acc)

        # Test Inference
        # Accumulate probabilities
        test_probs = clf.predict_proba(X_test)
        test_preds_sum += test_probs

        # Clean up to save memory on GPU if necessary
        del clf, X_tr, y_tr, X_va, y_va, val_probs, test_probs

    # --- Aggregation ---
    avg_test_preds = test_preds_sum / n_splits
    mean_acc = np.mean(fold_scores)

    print(f"\nXGBoost CV Complete.")
    print(f"Average Accuracy: {mean_acc}")

    return oof_preds, avg_test_preds, le, y_full
