import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_score, get_logger
from library.pipeline import (
    run_kfold_cv,
    train_meta_learner,
    predict_meta_learner,
    save_submission,
)

# Setup logger
logger = get_logger("runfile")


def main():
    # 1. Configuration Setup
    # Optimize for speed while ensuring full coverage for OOF generation
    # A100 can handle batch_size=16 for DeBERTa-Large (max_len=256)
    Config.epochs = 1
    Config.batch_size = 16
    Config.trn_folds = [0, 1, 2, 3, 4]  # Run all folds to get full OOF predictions
    Config.debug = False

    # Set seed for reproducibility
    seed_everything(Config.seed)

    logger.info("Starting execution with optimized config:")
    logger.info(f"Epochs: {Config.epochs}")
    logger.info(f"Batch Size: {Config.batch_size}")
    logger.info(f"Folds: {Config.trn_folds}")

    # 2. Run Base Learners
    # Dictionary to store results
    oof_preds_dict = {}
    test_preds_dict = {}
    y_labels_all = None  # Ground truth for the concatenated dataset (Train + Val)

    # Model A: DeBERTa-v3-Large
    logger.info(f"Training Model A: {Config.model_deberta}")
    oof_deberta, test_deberta, y_labels = run_kfold_cv(Config.model_deberta)
    oof_preds_dict[Config.model_deberta] = oof_deberta
    test_preds_dict[Config.model_deberta] = test_deberta
    y_labels_all = y_labels

    # Model B: RoBERTa-Large
    logger.info(f"Training Model B: {Config.model_roberta}")
    oof_roberta, test_roberta, _ = run_kfold_cv(Config.model_roberta)
    oof_preds_dict[Config.model_roberta] = oof_roberta
    test_preds_dict[Config.model_roberta] = test_roberta

    # 3. Train Meta-Learner (Stacking)
    logger.info("Training Meta-Learner...")
    meta_model = train_meta_learner(oof_preds_dict, y_labels_all)

    # Generate Stacked OOF Predictions for evaluation
    # We stack the OOF predictions from base models to feed into the meta-learner
    model_names = sorted(oof_preds_dict.keys())
    X_meta = np.column_stack([oof_preds_dict[name] for name in model_names])
    stacked_oof_preds = meta_model.predict(X_meta)
    stacked_oof_preds = np.clip(stacked_oof_preds, 0.0, 1.0)

    # 4. Evaluation on Hold-out Validation Set
    # The pipeline concatenates train.csv and validation.csv.
    # We need to isolate the predictions corresponding to the original validation.csv.
    df_train_part = pd.read_csv(Config.TRAIN_PATH)
    val_start_idx = len(df_train_part)

    # Slice the arrays to get the hold-out set
    y_val_holdout = y_labels_all[val_start_idx:]
    pred_val_holdout = stacked_oof_preds[val_start_idx:]

    # Calculate Metric
    final_auc = get_score(y_val_holdout, pred_val_holdout)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")
    df_val = pd.read_csv(Config.VAL_PATH)

    # Calculate Error Magnitude
    errors = np.abs(y_val_holdout - pred_val_holdout)

    # Feature Engineering for Analysis
    # Fill NaNs to avoid errors during length calculation
    comments = df_val["Comment"].fillna("").astype(str)

    df_val["char_count"] = comments.apply(len)
    df_val["word_count"] = comments.apply(lambda x: len(x.split()))
    df_val["caps_ratio"] = comments.apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    # Calculate Correlations
    corr_char = np.corrcoef(errors, df_val["char_count"])[0, 1]
    corr_word = np.corrcoef(errors, df_val["word_count"])[0, 1]
    corr_caps = np.corrcoef(errors, df_val["caps_ratio"])[0, 1]

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"  Char Count: {corr_char:.16f}")
    print(f"  Word Count: {corr_word:.16f}")
    print(f"  Caps Ratio: {corr_caps:.16f}")

    # 6. Submission Logic
    THRESHOLD = 0.9603817733990148
    if final_auc > THRESHOLD:
        logger.info(
            f"Validation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        # Predict on Test Set using Meta-Learner
        final_test_preds = predict_meta_learner(meta_model, test_preds_dict)

        # Save Submission
        save_submission(final_test_preds)
    else:
        logger.info(
            f"Validation metric {final_auc} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
