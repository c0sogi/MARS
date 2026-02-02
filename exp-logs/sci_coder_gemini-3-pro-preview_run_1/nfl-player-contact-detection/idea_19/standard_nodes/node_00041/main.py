import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

# Import library modules
from library.config import Config
from library.utils import seed_everything, setup_logging
from library.trainer import Trainer
from library.model_factory import EnsemblePredictor


def main():
    # ---------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # ---------------------------------------------------------
    # Speed up training by reducing the number of trees
    Config.LGBM_PARAMS["n_estimators"] = 100
    Config.XGB_PARAMS["n_estimators"] = 100

    # Ensure reproducibility
    seed_everything(Config.SEED)
    setup_logging()

    print("Initializing Pipeline...")
    trainer = Trainer()

    # ---------------------------------------------------------
    # 2. Phase 1: Train Scouts
    # ---------------------------------------------------------
    # Train lightweight models to identify difficult examples
    scouts = trainer.train_scouts()

    # ---------------------------------------------------------
    # 3. Phase 2: Mine Hard Negatives
    # ---------------------------------------------------------
    # Use scouts to find negatives that look like positives
    hard_indices = trainer.mine_hard_negatives(scouts)

    # ---------------------------------------------------------
    # 4. Phase 3: Train Experts
    # ---------------------------------------------------------
    # Train the final ensemble on the augmented dataset
    experts = trainer.train_experts(hard_indices)
    ensemble = EnsemblePredictor(experts)

    # ---------------------------------------------------------
    # 5. Phase 4: Threshold Optimization
    # ---------------------------------------------------------
    # Find the best decision threshold on the validation set
    best_threshold = trainer.optimize_threshold(ensemble)

    # ---------------------------------------------------------
    # 6. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Validation & Failure Analysis ---")

    # Load validation data
    df_val = trainer.data_manager.load_val_features(load_cached_data=True)
    X_val = df_val.drop(columns=["contact"])
    y_val = df_val["contact"].values

    # Generate predictions
    y_probs = ensemble.predict_proba(X_val)
    y_preds = (y_probs >= best_threshold).astype(int)

    # Calculate and print Final Validation Metric
    val_mcc = matthews_corrcoef(y_val, y_preds)
    print(f"Final Validation Metric: {val_mcc}")

    # Failure Analysis: Correlation of Error with Features
    # Error is absolute difference between truth (0/1) and probability
    errors = np.abs(y_val - y_probs)

    print("\nFailure Analysis (Feature correlation with Error):")
    feature_corrs = {}
    for col in X_val.columns:
        # Calculate correlation if column is numeric
        if pd.api.types.is_numeric_dtype(X_val[col]):
            # Handle potential NaNs or constant columns safely
            if X_val[col].std() > 0:
                corr = np.corrcoef(X_val[col].values, errors)[0, 1]
                feature_corrs[col] = corr
            else:
                feature_corrs[col] = 0.0

    # Sort and print top correlations
    sorted_corrs = sorted(feature_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:10]:
        print(f"  {feat}: {corr:.4f}")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    # Generate submission only if metric condition is met
    TARGET_METRIC = 0.6865

    if val_mcc > TARGET_METRIC:
        print(
            f"\nValidation Metric ({val_mcc:.4f}) > {TARGET_METRIC}. Generating submission..."
        )
        trainer.generate_submission(ensemble, best_threshold)
    else:
        print(
            f"\nValidation Metric ({val_mcc:.4f}) <= {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
