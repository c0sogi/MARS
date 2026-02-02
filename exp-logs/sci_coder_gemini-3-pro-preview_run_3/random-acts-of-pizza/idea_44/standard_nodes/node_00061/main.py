import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library.data_loader import DataLoader
from library.ensemble_trainer import StackingTrainer
from library import model_definitions


def main():
    # 1. Setup and Initialization
    utils.set_seed(config.SEED)
    logger = utils.get_logger("RunFile")
    logger.info("Starting execution of runfile.py")

    # 2. Data Loading
    # We use load_cached_data=True to leverage the pre-computed features in ./working
    loader = DataLoader(load_cached_data=True)
    try:
        data = loader.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # 3. Initialize the Stacking Trainer
    trainer = StackingTrainer()

    # 4. Generate Out-Of-Fold (OOF) Predictions
    # This performs 5-Fold CV on the training set to get unbiased inputs for the meta-learner
    oof_preds = trainer.generate_oof(data)

    # 5. Train Level 2 Meta-Learner
    # Trains Logistic Regression on the OOF predictions
    trainer.train_meta_learner(oof_preds, data["y_train"])

    # 6. Hold-out Validation
    # We must compute the metric on the hold-out validation set (val.parquet).
    # Since the 'final_retrain' method in StackingTrainer merges Train and Val for some models,
    # we perform a dedicated validation pass here using models trained ONLY on the training split.
    logger.info("Performing Hold-out Validation...")

    n_val = len(data["y_val"])
    n_models = len(trainer.model_names)
    l1_val_preds = np.zeros((n_val, n_models))

    for i, name in enumerate(trainer.model_names):
        # Instantiate a fresh model for validation purposes
        model = trainer.model_classes[name]()

        # Retrieve specific feature views for Train and Val
        X_spec_train, X_meta_train = trainer._get_model_input(data, name, "train")
        X_spec_val, X_meta_val = trainer._get_model_input(data, name, "val")
        y_train = data["y_train"]
        y_val = data["y_val"]

        # Determine if we use the validation set for Early Stopping (Boosting models)
        # This is standard practice to prevent overfitting during model selection
        eval_set = None
        if name in ["semantic_booster", "temporal_booster"]:
            eval_set = (X_spec_val, X_meta_val, y_val)

        # Fit on Training Data
        model.fit(X_spec_train, X_meta_train, y_train, eval_set=eval_set)

        # Predict on Validation Data
        # Note: predict_proba returns [prob_class_0, prob_class_1], we take index 1
        probs = model.predict_proba(X_spec_val, X_meta_val)[:, 1]
        l1_val_preds[:, i] = probs

    # Generate Final Predictions using the trained Meta-Learner
    val_final_probs = trainer.meta_learner.predict_proba(l1_val_preds)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(data["y_val"], val_final_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    # Identify which features correlate most with prediction errors
    logger.info("Performing Failure Analysis...")
    errors = np.abs(data["y_val"] - val_final_probs)

    # We correlate errors with the dense metadata features
    meta_cols = config.METADATA_COLS
    X_val_meta_raw = data["X_val_meta"]  # Note: These are scaled values

    correlations = []
    for i, col_name in enumerate(meta_cols):
        feat_vals = X_val_meta_raw[:, i]
        # Avoid correlation calculation on constant features
        if np.std(feat_vals) > 1e-9:
            corr, _ = pearsonr(errors, feat_vals)
            correlations.append((col_name, corr))
        else:
            correlations.append((col_name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Model Error (Validation Set):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission Generation
    # Only proceed if we meet the specified threshold
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Retrain models on the fullest possible dataset (Train + Val for Bagging, Train w/ Val ES for Boosting)
        trainer.final_retrain(data)

        # Generate predictions for Test set and save to CSV
        trainer.generate_submission(data)
        logger.info("Submission generation complete.")
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
