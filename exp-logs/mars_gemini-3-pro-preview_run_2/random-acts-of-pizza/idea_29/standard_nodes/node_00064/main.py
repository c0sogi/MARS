import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.utils import set_seed, load_object, WORKING_DIR
from library.data_loader import get_data_splits, NUMERICAL_FEATURES
from library.feature_extractor import EmbeddingGenerator, extract_metadata_features
from library.pipeline_builder import combine_features
from library.trainer import Trainer


def main():
    # 1. Setup
    # Ensure reproducibility
    set_seed(42)
    print("Starting runfile execution...")

    # 2. Train and Generate Submission (via Trainer)
    # The Trainer handles data loading, feature generation (train/test),
    # cross-validation, model saving, and initial submission generation.
    # It uses a merged dataset (Train + Val) for 5-fold CV to maximize data usage.
    trainer = Trainer(n_folds=5, random_state=42)
    trainer.run_cross_validation()

    # 3. Validation Assessment
    # We must evaluate on the specific hold-out validation set defined in metadata/val.csv.
    print("\n--- Validation Assessment ---")

    # Load splits (using cache where possible)
    # Note: get_data_splits returns the specific rows for train, val, test based on metadata
    _, val_df, _ = get_data_splits(load_cached_data=True)

    # Generate features for validation set
    # Note: Trainer generated features for 'full_train' (train+val) and 'test'.
    # We specifically generate features for the 'val' split here to ensure alignment.
    emb_gen = EmbeddingGenerator()
    high_res_val, low_res_val = emb_gen.process_split(
        val_df, "val", load_cached_data=True
    )
    meta_val = extract_metadata_features(val_df)

    # Combine features into the format expected by the pipeline
    X_val = combine_features(high_res_val, low_res_val, meta_val)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # Load models and predict
    print("Loading models for validation inference...")
    models_dir = os.path.join(WORKING_DIR, "models")
    preds_sum = np.zeros(len(X_val))
    n_folds = 5
    successful_folds = 0

    for fold in range(n_folds):
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
        if os.path.exists(model_path):
            model = load_object(model_path)
            # Predict probability of class 1 (Success)
            preds_sum += model.predict_proba(X_val)[:, 1]
            successful_folds += 1

    if successful_folds == 0:
        raise RuntimeError("No models found for validation.")

    # Average predictions from the ensemble
    y_pred_val = preds_sum / successful_folds

    # Calculate Metric
    val_auc = roc_auc_score(y_val, y_pred_val)
    # Print exactly as requested
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - y_pred_val)

    # Create a dataframe for analysis
    analysis_df = val_df[NUMERICAL_FEATURES].copy()
    analysis_df["prediction_error"] = errors

    # Calculate correlation between numerical features and prediction error
    correlations = analysis_df.corr()["prediction_error"].drop("prediction_error")
    correlations = correlations.sort_values(key=abs, ascending=False)

    print("Correlation between Model Error and Input Features:")
    print(correlations.head(10))

    # 5. Submission Logic
    # The Trainer has already generated ./submission/submission.csv
    # We must verify the threshold condition.
    threshold = 0.7160806860575912
    submission_path = "./submission/submission.csv"

    if val_auc > threshold:
        print(f"\nValidation metric ({val_auc}) exceeds threshold ({threshold}).")
        print(f"Submission file retained at {submission_path}")
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({threshold})."
        )
        if os.path.exists(submission_path):
            print("Removing submission file...")
            os.remove(submission_path)
        else:
            print("No submission file found to remove.")


if __name__ == "__main__":
    main()
