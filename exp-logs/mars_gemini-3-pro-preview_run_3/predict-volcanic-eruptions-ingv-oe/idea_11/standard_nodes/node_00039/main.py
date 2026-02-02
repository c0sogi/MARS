import os
import sys
import warnings
import numpy as np
import pandas as pd

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from library.config import Config
from library.utils import seed_everything, calculate_mae
from library.data_loader import generate_dataset
from library.training_pipeline import (
    run_training,
    generate_submission_file,
)


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)

    # Optimize hyperparameters
    Config.LGBM_PARAMS["n_estimators"] = 3000
    Config.XGB_PARAMS["n_estimators"] = 3000

    print("Initializing Spatial Concatenation Ensemble Pipeline...")

    # 2. Train Model
    models_dict, features = run_training(load_cached_data=True, debug=False)

    # 3. Validation Inference
    print("Running validation inference...")
    val_df = generate_dataset(Config.VAL_META, "val_features", load_cached_data=True)
    X_val = val_df[features]

    # Average predictions
    final_val_preds = np.zeros(len(X_val))
    model_count = 0

    for name, models in models_dict.items():
        for m in models:
            final_val_preds += m.predict(X_val)
            model_count += 1

    final_val_preds /= model_count

    # 4. Metric Calculation
    y_true = val_df["time_to_eruption"].values
    mae = calculate_mae(y_true, final_val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mae}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - final_val_preds)
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate Pearson correlation between features and error magnitude
    correlations = analysis_df.corr()["error_magnitude"].sort_values(ascending=False)

    print(
        "Top 5 Features positively correlated with Error Magnitude (Systematic Failures):"
    )
    # Skip the first one as it is error_magnitude itself
    print(correlations.head(6).iloc[1:])

    print("\nTop 5 Features negatively correlated with Error Magnitude:")
    print(correlations.tail(5))

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 2739761.2592384242

    if mae < THRESHOLD:
        print(f"\nValidation metric {mae} meets threshold {THRESHOLD}.")
        # generate_submission_file handles test set loading, full inference, and CSV saving
        generate_submission_file(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {mae} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
