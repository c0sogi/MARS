import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, log_loss
from scipy.stats import pearsonr
import warnings

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    # 1. Setup and Fast Baseline Configuration
    print("Initializing DEIB-AME Fast Baseline...")
    seed_everything(Config.SEED)

    # Override Config for speed (Fast Baseline requirements)
    # Reducing estimators to ensure completion within 2 hours
    Config.LGBM_PARAMS["n_estimators"] = 100
    Config.XGB_PARAMS["n_estimators"] = 100
    Config.HGB_PARAMS["max_iter"] = 100

    # Initialize Trainer
    trainer = Trainer()

    # 2. Data Loading & Feature Engineering
    # Limit training samples for speed
    TRAIN_SAMPLE_SIZE = 100000

    print(f"Loading training data (Sample Size: {TRAIN_SAMPLE_SIZE})...")
    df_train = trainer.feature_engine.process_train(
        load_cached_data=True, sample_size=TRAIN_SAMPLE_SIZE
    )

    print("Loading validation data (Full set)...")
    # We use the full validation set for accurate metric calculation
    df_val = trainer.feature_engine.process_val(load_cached_data=True)

    # 3. Training Pipeline

    # Phase 1: Train Scouts
    print("\n--- Phase 1: Training Scouts ---")
    scouts = trainer.train_scouts(df_train)

    # Phase 2: Mine Hard Negatives
    print("\n--- Phase 2: Mining Hard Negatives ---")
    hard_neg_indices = trainer.mine_hard_negatives(
        df_train, scouts, load_cached_data=True
    )

    # Phase 3: Train Experts
    print("\n--- Phase 3: Training Experts ---")
    experts = trainer.train_experts(df_train, hard_neg_indices)

    # 4. Validation & Threshold Tuning
    print("\n--- Phase 4: Validation & Analysis ---")

    # Prepare Validation Features
    X_val = df_val.drop(
        columns=[
            "contact",
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ],
        errors="ignore",
    )
    y_val = df_val["contact"].values

    # Ensemble Prediction
    print("Generating validation predictions...")
    probs_sum = np.zeros(len(X_val))
    for name, model in experts.items():
        # Ensure model is in eval mode implicitly by nature of sklearn/gbm predict
        p = model.predict_proba(X_val)
        probs_sum += p

    y_prob = probs_sum / len(experts)

    # Threshold Optimization
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_mcc = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred_temp = (y_prob >= thresh).astype(int)
        mcc = matthews_corrcoef(y_val, y_pred_temp)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {best_mcc}")
    print(f"Best Threshold: {best_thresh}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (absolute difference)
    errors = np.abs(y_val - y_prob)

    print("Correlation between Error Magnitude and Features:")
    correlations = []
    feature_cols = X_val.select_dtypes(include=[np.number]).columns

    for col in feature_cols:
        # Handle NaNs just in case
        if X_val[col].std() == 0:
            continue

        # Pearson correlation
        corr, _ = pearsonr(X_val[col].fillna(0), errors)
        correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for feat, corr in correlations[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 5. Submission
    TARGET_METRIC = 0.6865

    if best_mcc > TARGET_METRIC:
        print(
            f"\nValidation Metric ({best_mcc:.4f}) > Target ({TARGET_METRIC}). Generating Submission..."
        )

        # Save the best threshold for the trainer/inference method to use if needed,
        # though we will pass it directly.
        np.save(
            os.path.join(trainer.models_dir, "best_threshold.npy"),
            np.array([best_thresh]),
        )

        print("Loading and processing test data...")
        df_test = trainer.feature_engine.process_test(load_cached_data=True)

        print("Predicting test set...")
        trainer.predict_test(df_test, experts, best_thresh)

    else:
        print(
            f"\nValidation Metric ({best_mcc:.4f}) did not meet target ({TARGET_METRIC}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
