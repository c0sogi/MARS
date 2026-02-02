import pandas as pd
import numpy as np
import gc
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.feature_engineering import FeatureProcessor
from library.models import LGBMWrapper, XGBWrapper


def optimize_threshold(y_true, y_probs):
    """
    Finds the best threshold to maximize MCC.
    """
    best_threshold = 0.5
    best_score = -1.0

    # Search space: 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def train_ensemble(load_cached_data=True):
    """
    Orchestrates the training of the heterogeneous ensemble.
    """
    print("Initializing Feature Processor...")
    processor = FeatureProcessor()

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    # Load Training Data
    print("Loading Training Data...")
    df_train = processor.process_split("train", load_cached_data=load_cached_data)

    # Load Validation Data
    print("Loading Validation Data...")
    df_val = processor.process_split("val", load_cached_data=load_cached_data)

    # ---------------------------------------------------------
    # 2. Prepare Features and Targets
    # ---------------------------------------------------------
    # Define columns to exclude from features
    # We keep 'is_ground' as a feature, but drop IDs and targets
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "p1_id",
        "p2_id",
    ]

    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    target_col = "contact"

    print(f"Training with {len(feature_cols)} features.")

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # Clean up large dataframes to free memory for training
    del df_train, df_val
    gc.collect()

    # ---------------------------------------------------------
    # 3. Train LightGBM
    # ---------------------------------------------------------
    print("\n" + "=" * 30)
    print("Training LightGBM...")
    print("=" * 30)

    lgbm_model = LGBMWrapper()
    lgbm_model.fit(X_train, y_train, X_val, y_val)
    lgbm_model.save("lgbm_model.joblib")

    # Generate Probabilities
    lgbm_probs = lgbm_model.predict_proba(X_val)

    # ---------------------------------------------------------
    # 4. Train XGBoost
    # ---------------------------------------------------------
    print("\n" + "=" * 30)
    print("Training XGBoost...")
    print("=" * 30)

    xgb_model = XGBWrapper()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_model.save("xgb_model.joblib")

    # Generate Probabilities
    xgb_probs = xgb_model.predict_proba(X_val)

    # ---------------------------------------------------------
    # 5. Ensemble and Evaluation
    # ---------------------------------------------------------
    print("\n" + "=" * 30)
    print("Evaluating Ensemble...")
    print("=" * 30)

    # Simple Average Ensemble
    ensemble_probs = (lgbm_probs + xgb_probs) / 2.0

    # Optimize Threshold
    best_thresh, best_mcc = optimize_threshold(y_val, ensemble_probs)

    print(f"Best Threshold: {best_thresh}")
    print(f"Validation MCC: {best_mcc}")

    # Individual Model Performance for Reference
    _, lgbm_mcc = optimize_threshold(y_val, lgbm_probs)
    _, xgb_mcc = optimize_threshold(y_val, xgb_probs)

    print(f"LightGBM MCC: {lgbm_mcc}")
    print(f"XGBoost MCC: {xgb_mcc}")

    return best_thresh
