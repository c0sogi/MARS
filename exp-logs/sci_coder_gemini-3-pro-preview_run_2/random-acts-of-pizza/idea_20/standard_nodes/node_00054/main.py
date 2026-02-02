import os
import joblib
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.data_loader import load_dataset
from library.text_processing import generate_embeddings
from library.feature_engineering import assemble_feature_matrix
from library.training_pipeline import run_stratified_cv
from library.inference_pipeline import generate_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    print("Initializing execution...")

    # 2. Training Phase
    # run_stratified_cv performs 5-fold CV on the training set, saves models,
    # and generates an initial submission (which we will manage later).
    print("Starting Training Pipeline...")
    run_stratified_cv(debug=False)

    # 3. Validation Phase (Hold-out Set)
    print("\nStarting Validation on Hold-out Set...")

    # Load Validation Data
    df_val = load_dataset("val", load_cached_data=True)

    # Load/Generate Validation Embeddings (View 1)
    embeddings_val = generate_embeddings("val", load_cached_data=True)

    # Extract Features for View 2 & 3
    subreddits_val = df_val[Config.SUBREDDIT_COL].tolist()
    metadata_val = df_val[Config.NUMERICAL_COLS].values
    y_val = df_val[Config.TARGET_COL].values

    # Load Models and Predict
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    fold_preds = []

    print("Performing inference on validation set using trained folds...")
    for fold in range(Config.N_FOLDS):
        # Load artifacts
        pls_path = os.path.join(models_dir, f"pls_fold_{fold}.joblib")
        scaler_path = os.path.join(models_dir, f"scaler_fold_{fold}.joblib")
        clf_path = os.path.join(models_dir, f"clf_fold_{fold}.joblib")

        # Ensure artifacts exist
        if not (
            os.path.exists(pls_path)
            and os.path.exists(scaler_path)
            and os.path.exists(clf_path)
        ):
            raise FileNotFoundError(f"Missing artifacts for fold {fold}")

        pls = joblib.load(pls_path)
        scaler = joblib.load(scaler_path)
        clf = joblib.load(clf_path)

        # Transform features
        pls_feat = pls.transform(subreddits_val)
        meta_feat = scaler.transform(metadata_val)

        # Assemble
        X_val = assemble_feature_matrix(embeddings_val, pls_feat, meta_feat)

        # Predict
        preds = clf.predict_proba(X_val)[:, 1]
        fold_preds.append(preds)

    # Ensemble Averaging
    avg_preds_val = np.mean(fold_preds, axis=0)

    # 4. Metric Calculation
    val_auc = roc_auc_score(y_val, avg_preds_val)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nFailure Analysis on Validation Set:")
    df_analysis = df_val.copy()
    df_analysis["prediction"] = avg_preds_val
    # Calculate absolute error
    df_analysis["error"] = np.abs(
        df_analysis[Config.TARGET_COL] - df_analysis["prediction"]
    )

    # Calculate correlations
    correlations = {}

    # Numerical features
    for col in Config.NUMERICAL_COLS:
        if col in df_analysis.columns:
            # Handle potential constant columns or NaNs gracefully
            if df_analysis[col].std() > 0:
                corr = df_analysis["error"].corr(df_analysis[col])
                correlations[col] = corr
            else:
                correlations[col] = 0.0

    # Text Length features (derived)
    if "request_text_edit_aware" in df_analysis.columns:
        df_analysis["text_len_char"] = df_analysis["request_text_edit_aware"].str.len()
        correlations["text_len_char"] = df_analysis["error"].corr(
            df_analysis["text_len_char"]
        )

    # Print Top Correlations
    print("Correlation of Error Magnitude with Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 6. Conditional Submission
    THRESHOLD = 0.7141749705260098

    if val_auc > THRESHOLD:
        print(f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}).")
        print("Generating final submission...")
        # Regenerate submission to ensure it uses the correct inference pipeline logic
        generate_submission(models_dir, debug=False)
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")
        # Ensure no submission file exists (run_stratified_cv might have created one)
        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"Removing existing submission file at {Config.SUBMISSION_PATH}")
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
