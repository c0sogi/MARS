import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import importlib
import library.config
import library.ensemble_model

# Reload modules to ensure updates are picked up in persistent environment (Cite debug_lesson_5)
importlib.reload(library.config)
importlib.reload(library.ensemble_model)

from library.config import Config
from library.utils import seed_everything, setup_logger, compute_rmse
from library.feature_extractor import FeatureExtractor
from library.ensemble_model import Level1Predictors, MetaLearner
from library.data_loader import META_FEATURES


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = setup_logger(name="runfile")

    logger.info("Starting execution of Stacked Hybrid-Feature Ensemble pipeline...")

    # 2. Feature Extraction
    # This handles caching automatically.
    extractor = FeatureExtractor()
    data = extractor.extract_and_cache(load_cached_data=True)

    # 3. Prepare Data for Cross-Validation
    # We concatenate Train and Val sets to perform K-Fold CV on the available labeled data.
    # This allows us to generate OOF predictions for the entire dataset, including the validation portion.
    train_feats = data["train"]["features"]
    train_targets = data["train"]["targets"]
    val_feats = data["val"]["features"]
    val_targets = data["val"]["targets"]

    n_train = train_feats.shape[0]
    n_val = val_feats.shape[0]

    X_full = np.concatenate([train_feats, val_feats], axis=0)
    y_full = np.concatenate([train_targets, val_targets], axis=0)

    logger.info(f"Combined Data Shape: {X_full.shape}")

    # 4. K-Fold Cross-Validation (Level 1)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Placeholder for Out-of-Fold predictions from Level 1 models
    # Shape: (N_samples, 3_models)
    oof_preds_l1 = np.zeros((X_full.shape[0], 3))

    logger.info(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        X_train_fold, y_train_fold = X_full[train_idx], y_full[train_idx]
        X_val_fold, y_val_fold = X_full[val_idx], y_full[val_idx]

        # Train Level 1 Predictors
        l1_model = Level1Predictors()
        l1_model.fit(X_train_fold, y_train_fold)

        # Predict on Validation Fold
        val_fold_preds = l1_model.predict(X_val_fold)
        oof_preds_l1[val_idx] = val_fold_preds

    # 5. Train Level 2 Meta-Learner
    logger.info("Training Level 2 Meta-Learner on TRAIN OOF predictions only...")

    # Split OOF predictions and targets back into Train and Val components
    train_oof_preds = oof_preds_l1[:n_train]
    train_targets_subset = y_full[:n_train]

    val_oof_preds = oof_preds_l1[n_train:]
    val_targets_subset = y_full[n_train:]

    # Fit Meta-Learner ONLY on the training subset to avoid leakage
    meta_learner = MetaLearner()
    meta_learner.fit(train_oof_preds, train_targets_subset)

    # 6. Validation & Metrics
    # Predict on the held-out validation subset
    val_preds_subset = meta_learner.predict(val_oof_preds)

    # Compute RMSE on the hold-out validation set
    val_rmse = compute_rmse(val_targets_subset, val_preds_subset)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_rmse}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate absolute errors
    errors = np.abs(val_targets_subset - val_preds_subset)

    # Extract metadata features for the validation subset
    # Metadata features are the last N columns of the feature matrix
    num_meta = len(META_FEATURES)
    val_meta_features = X_full[n_train : n_train + n_val, -num_meta:]

    print("\nCorrelation between Error Magnitude and Metadata Features:")
    for i, feature_name in enumerate(META_FEATURES):
        feat_values = val_meta_features[:, i]
        # Calculate Pearson correlation
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, feat_values)
        print(f"{feature_name}: {corr:.4f}")

    # 8. Conditional Submission
    THRESHOLD = 18.09007350517167

    if val_rmse < THRESHOLD:
        logger.info(
            f"Validation RMSE ({val_rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Retrain Level 1 models on the FULL dataset (Train + Val)
        logger.info("Retraining Level 1 models on full dataset...")
        final_l1_model = Level1Predictors()
        final_l1_model.fit(X_full, y_full)

        # Prepare Test Data
        X_test = data["test"]["features"]
        ids_test = data["test"]["ids"]

        # Generate Test Predictions
        logger.info("Predicting on Test Set...")
        test_l1_preds = final_l1_model.predict(X_test)
        final_test_preds = meta_learner.predict(test_l1_preds)

        # Clip to valid range
        final_test_preds = np.clip(final_test_preds, 1.0, 100.0)

        # Save Submission
        submission_df = pd.DataFrame({"Id": ids_test, "Pawpularity": final_test_preds})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation RMSE ({val_rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
