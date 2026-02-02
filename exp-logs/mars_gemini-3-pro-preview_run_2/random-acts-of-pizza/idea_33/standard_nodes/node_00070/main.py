import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib

# Import library modules
from library.utils import set_seed, setup_logger
from library.trainer import ModelTrainer
from library.inference import InferenceManager
from library.data_manager import DataManager

# Import custom transformers to ensure they are available for joblib deserialization
import library.custom_transformers


def main():
    # 1. Setup and Initialization
    # Set seed for reproducibility
    set_seed(42)
    logger = setup_logger("RunFile")

    # Define working directory consistent with the provided library defaults
    work_dir = "./working/idea_33"

    # 2. Model Training
    # Initialize the trainer which handles data loading, embedding generation,
    # and the stratified cross-validation training loop.
    logger.info("Initializing ModelTrainer...")
    trainer = ModelTrainer(work_dir=work_dir)

    logger.info("Starting Training Loop...")
    # This executes the 5-fold CV, trains Bagged Logistic Regressions, and saves models.
    trainer.train_loop(n_folds=5)

    # 3. Validation & Metric Calculation
    # To strictly comply with the requirement to evaluate on the "hold-out validation set",
    # we reconstruct the Out-Of-Fold (OOF) predictions. Since the trainer merges train and val
    # for CV, the OOF predictions for the validation subset represent unbiased estimates.
    logger.info("Reconstructing OOF predictions for validation...")

    # Load the full feature matrix and labels used by the trainer (cached in work_dir)
    X_full = np.load(os.path.join(work_dir, "X_train_full.npy"))
    y_full = np.load(os.path.join(work_dir, "y_train_full.npy"))

    # Re-instantiate the same Cross-Validator to reproduce the splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y_full))

    # Iterate through folds to generate OOF predictions
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        model_path = os.path.join(work_dir, "models", f"model_fold_{fold}.joblib")
        if not os.path.exists(model_path):
            logger.error(f"Model for fold {fold} missing!")
            continue

        # Load the trained pipeline for this fold
        model = joblib.load(model_path)

        # Get validation features for this fold
        X_val_fold = X_full[val_idx]

        # Predict probability of success (class 1)
        # Note: Inference is fast as embeddings are pre-computed in X_full
        preds = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = preds

    # Identify the subset corresponding to the hold-out validation set
    # The DataManager loads train.csv then val.csv, and ModelTrainer stacks them.
    # We load the raw DFs to get the exact lengths.
    dm = DataManager(cache_dir=work_dir)
    train_df, val_df, _ = dm.load_dataset(load_cached_data=True)

    len_train = len(train_df)

    # Extract predictions and labels for the validation set portion
    val_preds = oof_preds[len_train:]
    val_labels = y_full[len_train:]

    # Calculate the Final Validation Metric (ROC AUC)
    final_metric = roc_auc_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("Running Failure Analysis on Validation Set...")

    # Construct an analysis DataFrame
    analysis_df = val_df.copy()
    analysis_df["pred"] = val_preds
    analysis_df["label"] = val_labels
    # Calculate absolute error
    analysis_df["error"] = np.abs(analysis_df["label"] - analysis_df["pred"])

    # Calculate correlation between Error and Metadata Features
    correlations = {}
    for col in dm.metadata_cols:
        if col in analysis_df.columns:
            try:
                # Ensure column is numeric
                series = pd.to_numeric(analysis_df[col], errors="coerce").fillna(0)
                # Compute correlation
                corr = series.corr(analysis_df["error"])
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception as e:
                pass

    # Print the top 5 features most correlated with error
    print("Failure Analysis - Top Error Correlations:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:5]:
        print(f"{feat}: {corr:.4f}")

    # 5. Submission Generation
    # Only generate submission if metric exceeds the specified threshold
    threshold = 0.7190361601447052

    if final_metric > threshold:
        logger.info(
            f"Validation metric {final_metric} > {threshold}. Generating submission..."
        )
        inference = InferenceManager(work_dir=work_dir)
        # Predict on test set using the ensemble of 5 models
        inference.predict(load_cached_data=True, n_folds=5)
    else:
        logger.warning(
            f"Validation metric {final_metric} <= {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
