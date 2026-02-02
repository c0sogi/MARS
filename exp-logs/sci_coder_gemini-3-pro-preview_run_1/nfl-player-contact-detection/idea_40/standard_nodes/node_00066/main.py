import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.training import Trainer
from library.data_manager import DataManager
from library.models import Ensemble


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    setup_logging()

    # 2. Configure for Fast Baseline
    # Override Config for speed and resource management within the time limit
    print("Configuring for fast baseline execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = (
        150000  # Sufficient data for signal, small enough for speed
    )

    # Reduce estimators for faster convergence while maintaining ensemble capacity
    Config.LGBM_PARAMS["n_estimators"] = 800
    Config.XGB_PARAMS["n_estimators"] = 800
    Config.EARLY_STOPPING_ROUNDS = 50

    # 3. Initialize Trainer and DataManager
    trainer = Trainer(Config)
    data_manager = trainer.data_manager

    # 4. Load Data
    # load_cached_data=True allows using pre-computed features if available in working dir
    print("Loading Data...")
    df_train = data_manager.get_train_data(load_cached_data=True)
    df_val = data_manager.get_val_data(load_cached_data=True)

    # 5. Training Pipeline (Step-by-Step execution for granular control)

    # Phase 1: Train Scouts
    # Scouts are trained on a balanced dataset to identify ambiguous samples
    scout_models = trainer.train_scouts(df_train)

    # Phase 2: Mine Hard Negatives
    # Use scouts to find negatives that look like positives (Hard Negatives)
    hard_neg_indices = trainer.mine_hard_negatives(
        df_train, scout_models, load_cached=True
    )

    # Phase 3: Train Experts
    # Experts are trained on Positives + Hard Negatives + Random Anchors
    expert_models = trainer.train_experts(df_train, hard_neg_indices, df_val)

    # Phase 4: Optimize Threshold
    # Find the decision threshold that maximizes MCC on the validation set
    best_threshold = trainer.optimize_threshold(expert_models, df_val)

    # 6. Validation & Failure Analysis
    print("\n--- Performing Validation Assessment & Failure Analysis ---")

    # Prepare Validation Features
    X_val, y_val = data_manager.get_X_y(df_val)

    # Create Ensemble for Validation Inference
    ensemble = Ensemble(expert_models)

    # Predict
    print("Running validation inference...")
    y_pred_prob = ensemble.predict(X_val)
    y_pred_binary = (y_pred_prob >= best_threshold).astype(int)

    # Compute Metric
    mcc = matthews_corrcoef(y_val, y_pred_binary)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mcc}")

    # Failure Analysis
    print("\nAnalyzing Failures...")
    # Calculate error magnitude (L1 distance)
    errors = np.abs(y_val - y_pred_prob)

    # Compute correlations between feature values and error magnitude
    # This identifies which features are most associated with model mistakes
    print("Calculating feature correlations with error...")
    correlations = X_val.corrwith(errors).abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Prediction Error:")
    print(correlations.head(10))

    # 7. Conditional Submission
    TARGET_SCORE = 0.6865

    if mcc > TARGET_SCORE:
        print(f"\nMetric ({mcc}) > {TARGET_SCORE}. Generating submission...")
        trainer.generate_submission(
            expert_models, best_threshold, load_cached_data=True
        )
    else:
        print(f"\nMetric ({mcc}) <= {TARGET_SCORE}. Submission skipped.")


if __name__ == "__main__":
    main()
