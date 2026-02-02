import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

# Import provided libraries
from library.config import Config
from library.utils import seed_everything, setup_logging, load_from_npy
from library.training_manager import TrainingManager
from library.data_manager import DataManager
from library.model_factory import TriModelEnsemble


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    logger = setup_logging()

    # Monkey-patch Config for Fast Baseline Execution
    # Reducing estimators to ensure training completes quickly within the 2-hour limit
    # while maintaining sufficient capacity for the ensemble.
    Config.LGBM_PARAMS["n_estimators"] = 500
    Config.XGB_PARAMS["n_estimators"] = 500
    Config.SKLEARN_HGB_PARAMS["max_iter"] = 500

    # Reduce early stopping rounds for faster convergence checks
    Config.EARLY_STOPPING_ROUNDS = 20

    logger.info("Configuration patched for fast baseline execution.")

    # 2. Pipeline Execution
    # Initialize TrainingManager with debug=False to ensure we use the full dataset
    # for accurate validation metrics, relying on the reduced estimators for speed.
    tm = TrainingManager(debug=False)

    # Phase 1: Train Scouts (Balanced Data)
    tm.train_scouts()

    # Phase 2: Mine Hard Negatives (Full Training Data)
    tm.mine_hard_negatives()

    # Phase 3: Train Expert (Anchored Dataset with Soft Labels)
    tm.train_expert()

    # Optimize Threshold on Validation Set
    best_thresh = tm.optimize_threshold()

    # 3. Validation & Analysis
    logger.info("--- Performing Final Validation & Failure Analysis ---")

    # Load Validation Data explicitly to ensure we have the full hold-out set
    dm = DataManager(debug=False)
    # get_scout_data returns (train, val) processed dataframes
    _, val_df = dm.get_scout_data(load_cached_data=True)

    # Prepare Features and Targets
    X_val, y_val = dm.prepare_X_y(val_df)

    # Load the Trained Expert Ensemble
    ensemble = TriModelEnsemble()
    ensemble.load(tm.expert_dir)

    # Inference on Validation Set
    # Predict probabilities using the unweighted average of the ensemble
    probs = ensemble.predict_proba(X_val)

    # Apply optimized threshold
    preds = (probs >= best_thresh).astype(int)

    # Compute Metric
    mcc = matthews_corrcoef(y_val, preds)
    print(f"Final Validation Metric: {mcc}")

    # 4. Failure Analysis
    # Calculate Error Magnitude (Absolute difference between probability and binary label)
    # High error indicates confident incorrect predictions or high uncertainty on clear cases.
    errors = np.abs(y_val - probs)

    # Compute Correlation between features and error magnitude
    logger.info("Computing feature correlations with error magnitude...")
    correlations = X_val.corrwith(pd.Series(errors, index=X_val.index))
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("\nFailure Analysis - Top Feature Correlations with Error Magnitude:")
    print(sorted_corr.head(10))

    # 5. Submission
    # Generate submission only if metric exceeds the specified threshold
    SUBMISSION_THRESHOLD = 0.6865
    if mcc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation MCC ({mcc}) > {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        tm.generate_submission()
    else:
        logger.warning(
            f"Validation MCC ({mcc}) <= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
