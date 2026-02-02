import os
import sys
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from library import config, data_manager, model_trainer


# ==========================================
# Configuration & Setup
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: LightGBM seed is handled in config.LGBM_PARAMS


def main():
    # 1. Setup
    set_seed(config.SEED)

    # 2. Model Training
    # We use debug=False to run on the full dataset for the best possible score.
    # The feature extraction is cached, so subsequent runs are fast.
    print("Initializing Model Trainer...")
    trainer = model_trainer.ModelTrainer()

    print("Starting Cross-Validation Training...")
    trainer.train_cross_validation(debug=False)

    # 3. Validation Metric
    # Retrieve the overall OOF MAE score
    final_metric = trainer.best_score
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    try:
        # Load validation data (features + target)
        val_df = data_manager.generate_feature_matrix(
            "val", debug=False, load_cached_data=True
        )

        # Identify feature columns (exclude metadata and target)
        exclude_cols = ["segment_id", "time_to_eruption"]
        feature_cols = [c for c in val_df.columns if c not in exclude_cols]

        # Prepare data
        X_val = val_df[feature_cols]
        y_val = val_df["time_to_eruption"]

        # Generate predictions using the ensemble
        # Averaging predictions from all fold models
        avg_preds = np.zeros(len(X_val))
        if not trainer.models:
            print("No models found for failure analysis.")
        else:
            for model in trainer.models:
                # Predict
                preds = model.predict(X_val, num_iteration=model.best_iteration)
                avg_preds += preds

            avg_preds /= len(trainer.models)

            # Calculate Absolute Error
            errors = np.abs(y_val - avg_preds)

            # Calculate Correlation between Features and Error Magnitude
            # We use numeric features only
            X_val_numeric = X_val.select_dtypes(include=[np.number])

            # Compute correlations
            correlations = X_val_numeric.corrwith(pd.Series(errors, index=X_val.index))
            abs_correlations = correlations.abs().sort_values(ascending=False)

            print("Top 10 Features Correlated with Error Magnitude:")
            print(abs_correlations.head(10))

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # 5. Submission
    THRESHOLD = 2617304.0647319085

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} < Threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict(debug=False)
    else:
        print(f"\nMetric {final_metric} >= Threshold {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
