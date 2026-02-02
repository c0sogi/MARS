import os
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from scipy.stats import spearmanr

from library.config import GlobalConfig, MPNET_CONFIG, DEBERTA_CONFIG
from library.utils import seed_everything, compute_spearman_metric
from library.modeling import (
    train_backbone,
    extract_features,
    train_ridge_head,
    predict_stream,
)


def perform_failure_analysis(val_preds, val_df, target_cols):
    """
    Analyzes the correlation between prediction error and input text length.
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    val_targets = val_df[target_cols].values

    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples,)
    sample_mae = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Extract metadata features
    # Fill NaNs with empty string for length calculation
    q_body_len = val_df["question_body"].fillna("").str.len()
    ans_len = val_df["answer"].fillna("").str.len()

    # Compute Spearman correlations
    # Handle cases where length might be constant or empty
    try:
        corr_q, _ = spearmanr(sample_mae, q_body_len)
    except:
        corr_q = 0.0

    try:
        corr_a, _ = spearmanr(sample_mae, ans_len)
    except:
        corr_a = 0.0

    print(f"Correlation between Error and Question Body Length: {corr_q:.4f}")
    print(f"Correlation between Error and Answer Length:        {corr_a:.4f}")

    # Identify worst performing targets
    col_mae = np.mean(np.abs(val_targets - val_preds), axis=0)
    worst_idx = np.argsort(col_mae)[-3:][::-1]
    print("\nTop 3 Targets with highest MAE:")
    for idx in worst_idx:
        print(f"  {target_cols[idx]}: {col_mae[idx]:.4f}")


def main():
    # 1. Setup
    seed_everything(GlobalConfig.SEED)
    print("Starting Late-Fusion Ensemble Pipeline...")

    # Load Targets for Training
    train_df = pd.read_csv(GlobalConfig.TRAIN_METADATA_PATH)
    train_targets = train_df[GlobalConfig.TARGET_COLS].values.astype(np.float32)

    # Load Validation Data for Evaluation
    val_df = pd.read_csv(GlobalConfig.VAL_METADATA_PATH)
    val_targets = val_df[GlobalConfig.TARGET_COLS].values.astype(np.float32)

    # ==========================================================================
    # STREAM 1: MPNet
    # ==========================================================================
    print("\n" + "-" * 40)
    print(" Processing Stream 1: MPNet")
    print("-" * 40)

    tokenizer_mpnet = AutoTokenizer.from_pretrained(MPNET_CONFIG.model_name)

    # Fine-tune Backbone
    if not os.path.exists(MPNET_CONFIG.model_save_path):
        train_backbone(MPNET_CONFIG, tokenizer_mpnet)
    else:
        print(
            f"Found existing model at {MPNET_CONFIG.model_save_path}, skipping training."
        )

    # Extract Features
    # Train
    mpnet_train_feats = extract_features(
        MPNET_CONFIG,
        tokenizer_mpnet,
        GlobalConfig.TRAIN_METADATA_PATH,
        MPNET_CONFIG.train_features_path,
    )
    # Val
    mpnet_val_feats = extract_features(
        MPNET_CONFIG,
        tokenizer_mpnet,
        GlobalConfig.VAL_METADATA_PATH,
        MPNET_CONFIG.val_features_path,
    )
    # Test
    mpnet_test_feats = extract_features(
        MPNET_CONFIG,
        tokenizer_mpnet,
        GlobalConfig.TEST_METADATA_PATH,
        MPNET_CONFIG.test_features_path,
    )

    # Train Ridge Head
    train_ridge_head(MPNET_CONFIG, mpnet_train_feats, train_targets)

    # Predict on Val and Test
    mpnet_val_preds = predict_stream(MPNET_CONFIG, mpnet_val_feats)
    mpnet_test_preds = predict_stream(MPNET_CONFIG, mpnet_test_feats)

    # Cleanup Stream 1
    del tokenizer_mpnet, mpnet_train_feats, mpnet_val_feats, mpnet_test_feats
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================================
    # STREAM 2: RoBERTa
    # ==========================================================================
    print("\n" + "-" * 40)
    print(" Processing Stream 2: RoBERTa")
    print("-" * 40)

    tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_CONFIG.model_name)

    # Fine-tune Backbone
    if not os.path.exists(ROBERTA_CONFIG.model_save_path):
        train_backbone(ROBERTA_CONFIG, tokenizer_roberta)
    else:
        print(
            f"Found existing model at {ROBERTA_CONFIG.model_save_path}, skipping training."
        )

    # Extract Features
    # Train
    roberta_train_feats = extract_features(
        ROBERTA_CONFIG,
        tokenizer_roberta,
        GlobalConfig.TRAIN_METADATA_PATH,
        ROBERTA_CONFIG.train_features_path,
    )
    # Val
    roberta_val_feats = extract_features(
        ROBERTA_CONFIG,
        tokenizer_roberta,
        GlobalConfig.VAL_METADATA_PATH,
        ROBERTA_CONFIG.val_features_path,
    )
    # Test
    roberta_test_feats = extract_features(
        ROBERTA_CONFIG,
        tokenizer_roberta,
        GlobalConfig.TEST_METADATA_PATH,
        ROBERTA_CONFIG.test_features_path,
    )

    # Train Ridge Head
    train_ridge_head(ROBERTA_CONFIG, roberta_train_feats, train_targets)

    # Predict on Val and Test
    roberta_val_preds = predict_stream(ROBERTA_CONFIG, roberta_val_feats)
    roberta_test_preds = predict_stream(ROBERTA_CONFIG, roberta_test_feats)

    # Cleanup Stream 2
    del tokenizer_roberta, roberta_train_feats, roberta_val_feats, roberta_test_feats
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================================
    # ENSEMBLE & VALIDATION
    # ==========================================================================
    print("\n" + "-" * 40)
    print(" Ensemble & Validation")
    print("-" * 40)

    # Late Fusion (Average Probabilities)
    final_val_preds = 0.5 * mpnet_val_preds + 0.5 * roberta_val_preds

    # Compute Metric
    val_score = compute_spearman_metric(val_targets, final_val_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis
    perform_failure_analysis(final_val_preds, val_df, GlobalConfig.TARGET_COLS)

    # ==========================================================================
    # SUBMISSION
    # ==========================================================================
    THRESHOLD = 0.39777746135407066

    if val_score > THRESHOLD:
        print(
            f"\nValidation score ({val_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensemble Test Predictions
        final_test_preds = 0.5 * mpnet_test_preds + 0.5 * deberta_test_preds

        # Load Test Metadata for IDs
        test_df = pd.read_csv(GlobalConfig.TEST_METADATA_PATH)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(final_test_preds, columns=GlobalConfig.TARGET_COLS)
        sub_df.insert(0, "qa_id", test_df["qa_id"])

        # Save
        sub_df.to_csv(GlobalConfig.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {GlobalConfig.SUBMISSION_PATH}")
        print(sub_df.head())
    else:
        print(
            f"\nValidation score ({val_score}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
