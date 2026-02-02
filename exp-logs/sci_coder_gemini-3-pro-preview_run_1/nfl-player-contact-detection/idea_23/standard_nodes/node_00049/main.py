import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Import from the provided library files
from library.config import Config
from library.training_core import TrainingCore
from library.evaluation import Evaluator
from library.data_manager import DataManager


def set_seeds(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seeds(Config.SEED)
    print("Initializing VDAM-E Pipeline...")

    # 2. Training Pipeline
    # The TrainingCore handles the Scout -> Mine -> Expert curriculum
    trainer = TrainingCore()
    trainer.run(load_cached_data=True)

    # 3. Evaluation & Metric Calculation
    print("\n--- Validation & Analysis ---")
    evaluator = Evaluator()
    dm = DataManager()

    # Load validation data explicitly for analysis
    # Note: load_cached_data=True utilizes the cache created during the training/prep phase
    df_val = dm.get_val_features(load_cached_data=True)
    X_val = df_val[Config.FEATURES]
    y_val = df_val["contact"].values  # Use raw binary target for metric

    # Optimize Threshold
    # This finds the threshold that maximizes MCC on the validation set
    best_threshold = evaluator.optimize_threshold(load_cached_data=True)

    # Generate Ensemble Predictions on Validation Set
    evaluator.load_expert_models()
    y_probs = evaluator.predict_ensemble(X_val)
    y_preds = (y_probs >= best_threshold).astype(int)

    # Calculate Final Metric
    final_mcc = matthews_corrcoef(y_val, y_preds)
    print(f"Final Validation Metric: {final_mcc:.16f}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error (residuals)
    errors = np.abs(y_val - y_probs)

    # Create a DataFrame to analyze correlations
    df_analysis = X_val.copy()
    df_analysis["error"] = errors

    # Compute correlation between features and error magnitude
    # We drop the 'error' column itself from the correlation index
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Correlation between Prediction Error and Input Features:")
    print(correlations)

    # 5. Submission Generation
    # Condition: Only submit if validation metric meets the requirement
    if final_mcc > 0.6865:
        print(
            f"\nValidation metric ({final_mcc:.4f}) meets threshold (0.6865). Generating submission..."
        )
        evaluator.generate_submission(best_threshold, load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_mcc:.4f}) is below threshold (0.6865). Skipping submission."
        )


if __name__ == "__main__":
    main()
