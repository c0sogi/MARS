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
    reshape_for_siamese,
    pivot_predictions_to_wide,
    add_stage2_features,
)


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)

    # Optimize hyperparameters for a fast baseline execution
    # Reducing n_estimators ensures the pipeline completes well within the 2-hour limit
    # while retaining sufficient capacity for this dataset size.
    Config.LGBM_PARAMS["n_estimators"] = 1500
    Config.XGB_PARAMS["n_estimators"] = 1500
    Config.CAT_PARAMS["iterations"] = 1500

    print("Initializing Hierarchical Siamese-Stacking Pipeline...")

    # 2. Train Model
    # run_training handles data loading (cached), stage 1-3 training, and model saving.
    # We use debug=False to utilize the full dataset for maximum performance.
    s1_models, s2_models, s3_model, s2_features = run_training(
        load_cached_data=True, debug=False
    )

    # 3. Validation Inference
    # We manually run the inference pipeline on the validation set to ensure
    # we have the exact predictions for metric calculation and failure analysis.
    print("Running validation inference...")
    val_df = generate_dataset(Config.VAL_META, "val_features", load_cached_data=True)

    # --- Stage 1 Inference (Siamese) ---
    # Reshape validation data to long format (one row per sensor)
    X_val_long, _, groups_val = reshape_for_siamese(val_df, is_train=False)

    # Average predictions across all Stage 1 models (Ensemble of LightGBMs)
    s1_preds = np.zeros(len(X_val_long))
    for model in s1_models:
        s1_preds += model.predict(X_val_long) / len(s1_models)

    # Pivot predictions back to wide format (one row per segment) and merge
    pred_df = pivot_predictions_to_wide(groups_val, s1_preds)
    val_aug = val_df.merge(pred_df, on="segment_id", how="left")

    # --- Stage 2 Inference (Stacking) ---
    # Generate aggregate features from Stage 1 predictions
    val_aug = add_stage2_features(val_aug)
    X_val_s2 = val_aug[s2_features]

    # Generate meta-features using Stage 2 models (LGBM, XGB, CatBoost)
    val_meta = pd.DataFrame()
    for name, models in s2_models.items():
        pred = np.zeros(len(X_val_s2))
        for m in models:
            pred += m.predict(X_val_s2) / len(models)
        val_meta[name] = pred

    # --- Stage 3 Inference (Meta-Learner) ---
    final_val_preds = s3_model.predict(val_meta)

    # 4. Metric Calculation
    y_true = val_df["time_to_eruption"].values
    mae = calculate_mae(y_true, final_val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mae}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - final_val_preds)
    analysis_df = X_val_s2.copy()
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
