import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy.stats import spearmanr

# Import library modules
from library.config import PathConfig, TrainConfig, ModelConfig
from library.dataset import load_data, QuestDataset
from library.engine import run_backbone_training
from library.feature_pipeline import run_feature_extraction
from library.ridge_head import train_ridge_and_predict
from library.utils import seed_everything, compute_spearmanr, load_joblib


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    seed_everything(TrainConfig.seed)

    # -------------------------------------------------------------------------
    # 2. Backbone Training
    # -------------------------------------------------------------------------
    print("Preparing data for backbone training...")
    # Load raw text data
    train_df, val_df, _ = load_data(load_cached_data=True)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.model_name)

    # Create Datasets
    train_dataset = QuestDataset(
        train_df, tokenizer, max_len=ModelConfig.max_len, mode="train"
    )
    val_dataset = QuestDataset(
        val_df, tokenizer, max_len=ModelConfig.max_len, mode="train"
    )

    # Create DataLoaders
    # We use the batch size from config.
    # Pin memory and num_workers for speed.
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.batch_size * 2,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    print("Starting Backbone Training...")
    run_backbone_training(train_loader, val_loader)

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("\nStarting Feature Extraction...")
    # We set load_cached_data=False to force extraction using the newly trained model
    run_feature_extraction(load_cached_data=False)

    # -------------------------------------------------------------------------
    # 4. Ridge Regression & Submission Generation
    # -------------------------------------------------------------------------
    print("\nStarting Ridge Regression Training...")
    # We set load_cached_model=False to force retraining the head
    train_ridge_and_predict(load_cached_model=False)

    # -------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Final Validation and Failure Analysis...")

    # Load validation features and targets (cached by feature_pipeline)
    if not (
        os.path.exists(PathConfig.VAL_FEATURES_CACHE)
        and os.path.exists(PathConfig.VAL_TARGETS_CACHE)
    ):
        raise FileNotFoundError("Validation cache not found.")

    val_features = np.load(PathConfig.VAL_FEATURES_CACHE)
    val_targets = np.load(PathConfig.VAL_TARGETS_CACHE)

    # Load the trained Ridge model
    if not os.path.exists(PathConfig.RIDGE_SAVE_PATH):
        raise FileNotFoundError("Ridge model not found.")

    ridge_model = load_joblib(PathConfig.RIDGE_SAVE_PATH)

    # Predict on validation set
    val_preds = ridge_model.predict(val_features)
    val_preds = np.clip(val_preds, 0.0, 1.0)

    # Compute Final Metric
    final_metric = compute_spearmanr(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # 1. Compute error magnitude per sample (Mean Absolute Error across 30 targets)
    error_magnitude = np.mean(np.abs(val_targets - val_preds), axis=1)

    # 2. Load metadata to get input features (lengths)
    # val_df is already loaded, but let's ensure we have the columns
    val_df["q_len"] = val_df["question_body"].fillna("").str.len()
    val_df["a_len"] = val_df["answer"].fillna("").str.len()
    val_df["q_title_len"] = val_df["question_title"].fillna("").str.len()

    # 3. Calculate correlations
    print("Failure Analysis (Correlation with Error Magnitude):")
    features_to_analyze = ["q_len", "a_len", "q_title_len"]

    for feat in features_to_analyze:
        if feat in val_df.columns:
            corr, _ = spearmanr(error_magnitude, val_df[feat])
            print(f"  Correlation with {feat}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Check
    # -------------------------------------------------------------------------
    threshold = 0.39678116179820055
    submission_path = PathConfig.SUBMISSION_FILE

    if final_metric > threshold:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({threshold}). Submission saved to {submission_path}."
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({threshold}). Removing submission file."
        )
        if os.path.exists(submission_path):
            os.remove(submission_path)


if __name__ == "__main__":
    main()
