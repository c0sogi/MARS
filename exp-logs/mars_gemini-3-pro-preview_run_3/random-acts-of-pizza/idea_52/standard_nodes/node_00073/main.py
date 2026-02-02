import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

from library.config import SEED, TARGET_COL, ID_COL, SUBMISSION_DIR, NUMERICAL_COLS
from library.utils import set_seed, setup_logging, log_metric
from library.data_loader import load_dataset, get_stratified_folds
from library.feature_builder import FeaturePipeline
from library.hybrid_engine import HybridTrainer, HybridPredictor


def main():
    # 1. Setup
    setup_logging()
    set_seed(SEED)

    print("Initializing Hept-View Stacking Ensemble...")

    # 2. Load Data
    print("Loading datasets...")
    df_train = load_dataset("train")
    df_val = load_dataset("val")
    df_test = load_dataset("test")

    # 3. Feature Engineering
    print("Building features...")
    pipeline = FeaturePipeline()

    # Fit pipeline on training data to establish vocabulary and scalers
    pipeline.fit(df_train)

    # Transform all splits
    # We use load_cached_data=True to optimize runtime on re-runs
    print("Transforming Training Data...")
    X_train = pipeline.transform(df_train, "train", load_cached_data=True)
    print("Transforming Validation Data...")
    X_val = pipeline.transform(df_val, "val", load_cached_data=True)
    print("Transforming Test Data...")
    X_test = pipeline.transform(df_test, "test", load_cached_data=True)

    y_train = df_train[TARGET_COL]
    y_val = df_val[TARGET_COL]

    # 4. Training Loop
    trainer = HybridTrainer()
    folds = list(get_stratified_folds(df_train))

    # Define Model Architecture
    # Tuples of (ModelName, LearnerType)
    volatile_models = [
        ("SemanticBooster", "SemanticBooster"),
        ("SemanticGradient", "SemanticGradient"),
        ("TemporalBooster", "TemporalBooster"),
    ]

    stable_models = [
        ("LexicalBagger", "LexicalBagger"),
        ("CommunityBagger", "CommunityBagger"),
        ("SemanticBagger", "SemanticBagger"),
        ("MetadataAnchor", "MetadataAnchor"),
    ]

    # Dictionary to store OOF predictions for Meta-Learner
    oof_preds_dict = {}

    # Train Volatile Learners (Bagging Protocol)
    # Trains K models per learner, uses Early Stopping
    for model_name, learner_name in volatile_models:
        oof = trainer.train_volatile(model_name, learner_name, X_train, y_train, folds)
        oof_preds_dict[model_name] = oof

    # Train Stable Learners (Retrain Protocol)
    # Generates OOF via CV, then retrains single model on Training set
    # We do NOT retrain on Val set here to ensure the Final Validation Metric is valid (no leakage)
    for model_name, learner_name in stable_models:
        oof = trainer.train_stable(
            model_name, learner_name, X_train, y_train, folds, X_retrain=None
        )
        oof_preds_dict[model_name] = oof

    # Train Meta-Learner
    # Stack predictions in a deterministic order
    all_model_names = [m[0] for m in volatile_models] + [m[0] for m in stable_models]
    X_meta_train = np.column_stack([oof_preds_dict[name] for name in all_model_names])

    trainer.train_meta(X_meta_train, y_train)

    # 5. Validation Inference
    print("\nRunning Validation Inference...")
    predictor = HybridPredictor()

    volatile_names = [m[0] for m in volatile_models]
    stable_names = [m[0] for m in stable_models]

    # Predict on hold-out validation set
    val_preds = predictor.predict(X_val, volatile_names, stable_names)

    # Compute and Print Metric
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Compute absolute error
    errors = np.abs(y_val - val_preds)

    # Correlate error with metadata features to find weak spots
    meta_features = X_val["X_meta"]
    correlations = []

    # X_meta columns correspond to NUMERICAL_COLS in config
    for i, col_name in enumerate(NUMERICAL_COLS):
        if i < meta_features.shape[1]:
            feat_vals = meta_features[:, i]
            # Avoid correlation with constant features
            if np.std(feat_vals) > 1e-9:
                corr, _ = stats.pearsonr(errors, feat_vals)
                correlations.append((col_name, corr))
            else:
                correlations.append((col_name, 0.0))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.7222984867326668
    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predictor.predict(X_test, volatile_names, stable_names)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({ID_COL: df_test[ID_COL], TARGET_COL: test_preds})

        # Save
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"\nValidation AUC ({val_auc}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
