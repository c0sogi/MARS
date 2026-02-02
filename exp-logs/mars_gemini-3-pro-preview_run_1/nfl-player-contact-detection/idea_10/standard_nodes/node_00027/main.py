import sys
import os
import numpy as np
import pandas as pd
import warnings

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.pipeline import Pipeline
from library.utils import seed_everything, calc_mcc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Modify Config for Fast Baseline Execution
    # We reduce the number of estimators and complexity to ensure the run completes quickly.
    # We also silence verbose outputs.
    Config.LGBM_PARAMS["n_estimators"] = 100
    Config.LGBM_PARAMS["num_leaves"] = 31
    Config.LGBM_PARAMS["learning_rate"] = 0.1
    Config.LGBM_PARAMS["verbose"] = -1

    Config.XGB_PARAMS["n_estimators"] = 100
    Config.XGB_PARAMS["max_depth"] = 6
    Config.XGB_PARAMS["learning_rate"] = 0.1
    Config.XGB_PARAMS["verbose_eval"] = False

    Config.VERBOSE_EVAL = 0  # Disable verbose logging in wrappers

    # Initialize the Pipeline
    pipeline = Pipeline()

    # 2. Pipeline Execution (Manual Steps)

    # Phase 1: Scout Training
    # Trains a lightweight model on a balanced subset to learn the rough boundary.
    scout_model = pipeline.run_scout_phase(load_cached_data=True)

    # Phase 2: Hard Negative Mining
    # Uses the Scout model to find difficult negative examples in the full dataset.
    hard_neg_indices = pipeline.run_mining_phase(scout_model, load_cached_data=True)

    # Phase 3: Expert Training
    # Trains the final ensemble on Positives + Hard Negatives + Buffer.
    pipeline.run_expert_phase(hard_neg_indices, load_cached_data=True)

    # Phase 4: Threshold Optimization & Validation
    # We retrieve the validation set to perform optimization and required analysis.
    X_val, y_val, meta_val = pipeline.dm.get_val_dataset(load_cached_data=True)

    # Optimize the ensemble threshold for MCC
    best_thresh, best_mcc = pipeline.ensemble.optimize_threshold(X_val, y_val)

    # Calculate final metric explicitly for reporting
    probs_val = pipeline.ensemble.predict_proba(X_val)
    preds_val = (probs_val > best_thresh).astype(int)
    final_mcc = calc_mcc(y_val, preds_val)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_mcc}")

    # 3. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (Absolute Error)
    # y_val is binary (0/1), probs_val is float [0, 1]
    errors = np.abs(y_val - probs_val)

    # Correlate errors with input features to identify weak points
    # Select only numeric features for correlation
    numeric_features = X_val.select_dtypes(include=[np.number])

    # Compute correlation between each feature and the error vector
    correlations = numeric_features.apply(
        lambda x: x.corr(pd.Series(errors, index=x.index))
    )

    # Sort by magnitude of correlation
    abs_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 10 features correlated with error magnitude:")
    print(abs_corrs.head(10))

    # 4. Inference & Submission
    # Conditional submission based on validation performance
    THRESHOLD_SCORE = 0.6782

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation MCC ({final_mcc}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        pipeline.run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation MCC ({final_mcc}) does not exceed threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
