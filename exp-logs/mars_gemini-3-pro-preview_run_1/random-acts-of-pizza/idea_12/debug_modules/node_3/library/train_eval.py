import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.feature_engineering import run_feature_engineering
from library.models_rf import StreamARF
from library.models_mlp import StreamBMLP


def run_training_pipeline(load_cached_data=True, debug=Config.DEBUG):
    """
    Executes the full training and evaluation pipeline for Idea 12.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from cache.
        debug (bool): Whether to run in debug mode with a subset of data.

    Returns:
        dict: Dictionary containing validation metrics.
    """
    print("Starting Training Pipeline...")

    # 1. Feature Engineering / Loading
    # run_feature_engineering handles caching and processing internally
    train_data, val_data, test_data = run_feature_engineering(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Stream A: Dual-Lexical Augmented Random Forest
    print("\n" + "=" * 40)
    print("Stream A: Dual-Lexical Augmented Random Forest")
    print("=" * 40)

    rf_model = StreamARF()
    rf_model.fit(train_data)
    rf_auc = rf_model.evaluate(val_data)

    # 3. Stream B: Hierarchical Decoupled-Attention MLP
    print("\n" + "=" * 40)
    print("Stream B: Hierarchical Decoupled-Attention MLP")
    print("=" * 40)

    mlp_model = StreamBMLP()
    mlp_model.fit(train_data, val_data)
    mlp_auc = mlp_model.evaluate(val_data)

    # 4. Ensemble Evaluation (Validation)
    print("\n" + "=" * 40)
    print("Ensemble Evaluation")
    print("=" * 40)

    rf_val_preds = rf_model.predict_proba(val_data)
    mlp_val_preds = mlp_model.predict_proba(val_data)

    # Weighted Average
    ensemble_val_preds = (
        Config.ENSEMBLE_WEIGHT_RF * rf_val_preds
        + Config.ENSEMBLE_WEIGHT_MLP * mlp_val_preds
    )

    ensemble_auc = roc_auc_score(val_data["y"], ensemble_val_preds)
    print(f"Ensemble Validation ROC AUC: {ensemble_auc}")

    # 5. Test Inference and Submission
    print("\n" + "=" * 40)
    print("Generating Submission")
    print("=" * 40)

    rf_test_preds = rf_model.predict_proba(test_data)
    mlp_test_preds = mlp_model.predict_proba(test_data)

    ensemble_test_preds = (
        Config.ENSEMBLE_WEIGHT_RF * rf_test_preds
        + Config.ENSEMBLE_WEIGHT_MLP * mlp_test_preds
    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "request_id": test_data["ids"],
            "requester_received_pizza": ensemble_test_preds,
        }
    )

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return {"rf_auc": rf_auc, "mlp_auc": mlp_auc, "ensemble_auc": ensemble_auc}
