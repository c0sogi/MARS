import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_datasets
from library.feature_engineering import FusionTransformer
from library.model import tune_ensemble_hyperparameters


def main():
    # 1. Setup and Initialization
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    logger = setup_logger("RunFile", os.path.join(Config.WORKING_DIR, "runfile.log"))

    logger.info("Initializing Tri-Backbone Asymmetric Early Fusion Pipeline...")

    # 2. Load Data
    # We load cached data to speed up execution if embeddings were pre-computed
    data = load_datasets(load_cached_data=True)

    y = data["y_train"]
    df_test = data["df_test"]

    # Define metadata columns for failure analysis later
    meta_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    # Prepare Raw Test Dictionary (static structure)
    raw_test_dict = {
        "anchor": data["test_embeddings"]["anchor"],
        "aux1": data["test_embeddings"]["aux1"],
        "aux2": data["test_embeddings"]["aux2"],
        "meta": data["meta_test"],
    }

    # Access Raw Train Data for slicing
    raw_train_embeddings = data["train_embeddings"]
    raw_train_meta = data["meta_train"]

    # 3. Stratified Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros(len(y))
    test_preds_accum = np.zeros(len(df_test))

    logger.info(f"Starting {Config.N_FOLDS}-Fold Stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        logger.info(f"--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # A. Slice Data for this Fold
        X_train_dict = {
            "anchor": raw_train_embeddings["anchor"][train_idx],
            "aux1": raw_train_embeddings["aux1"][train_idx],
            "aux2": raw_train_embeddings["aux2"][train_idx],
            "meta": raw_train_meta[train_idx],
        }

        X_val_dict = {
            "anchor": raw_train_embeddings["anchor"][val_idx],
            "aux1": raw_train_embeddings["aux1"][val_idx],
            "aux2": raw_train_embeddings["aux2"][val_idx],
            "meta": raw_train_meta[val_idx],
        }

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # B. Feature Engineering (Fit on Train, Transform All)
        # This handles PCA and Normalization inside the fold to prevent leakage
        transformer = FusionTransformer()
        transformer.fit(X_train_dict)

        X_train_fused = transformer.transform(X_train_dict)
        X_val_fused = transformer.transform(X_val_dict)
        X_test_fused = transformer.transform(raw_test_dict)

        # C. Hyperparameter Tuning & Training
        # Tunes the Bagged Ensemble (C, class_weight) on the current fold
        best_model, best_params, best_score = tune_ensemble_hyperparameters(
            X_train=X_train_fused,
            y_train=y_train_fold,
            X_val=X_val_fused,
            y_val=y_val_fold,
            param_grid=Config.GRID_PARAMS,
            n_estimators=Config.N_BAGGING_ESTIMATORS,
            random_state=Config.SEED,
        )

        logger.info(f"Fold {fold+1} Best Validation AUC: {best_score}")

        # D. Inference
        # OOF Predictions
        val_probs = best_model.predict_proba(X_val_fused)[:, 1]
        oof_preds[val_idx] = val_probs

        # Test Predictions (Accumulate for averaging later)
        test_probs = best_model.predict_proba(X_test_fused)[:, 1]
        test_preds_accum += test_probs

    # 4. Final Evaluation
    overall_auc = roc_auc_score(y, oof_preds)
    # Required output format
    print(f"Final Validation Metric: {overall_auc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y - oof_preds)

    # Construct DataFrame with raw metadata and error
    df_analysis = pd.DataFrame(raw_train_meta, columns=meta_cols)
    df_analysis["error"] = errors

    # Calculate correlation
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("\nFailure Analysis (Correlation of Features with Error Magnitude):")
    print(correlations)

    # 6. Submission Logic
    threshold = 0.7190361601447052

    if overall_auc > threshold:
        logger.info(
            f"Validation metric {overall_auc} exceeds threshold {threshold}. Generating submission."
        )

        # Average predictions across folds (CV-Bagging)
        avg_test_preds = test_preds_accum / Config.N_FOLDS

        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        logger.warning(
            f"Validation metric {overall_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
