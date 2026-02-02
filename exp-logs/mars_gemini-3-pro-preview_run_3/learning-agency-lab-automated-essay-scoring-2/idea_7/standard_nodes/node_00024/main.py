import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_qwk, get_logger
from library.data import load_and_preprocess
from library.trainer import train_backbone
from library.stacking import train_stacking, load_oof_data, inference


def main():
    # 1. Initialize Configuration
    config = Config()

    # --- FAST BASELINE SETTINGS ---
    # We restrict epochs to 1 to ensure the entire 5-fold pipeline completes
    # within the 2-hour limit on the available hardware.
    config.epochs = 1

    # Setup output directories
    config.setup()

    # Initialize Logger
    logger = get_logger(os.path.join(config.output_dir, "runfile.log"))
    logger.info("Starting Runfile Execution")

    # 2. Set Random Seed for Reproducibility
    seed_everything(config.seed)

    # 3. Load and Preprocess Data
    logger.info("Loading and preprocessing data...")
    # load_cached_data=True allows using pre-computed features/folds if they exist
    train_df, test_df = load_and_preprocess(config, load_cached_data=True)

    # 4. Train Backbone (Fine-tune DeBERTa)
    # This processes all 5 folds and saves OOF embeddings to ./working/idea_X/cache
    logger.info("Starting Backbone Training (5 Folds)...")
    train_backbone(train_df, config)

    # 5. Train Stacking Head (LightGBM)
    # Trains LightGBM on the generated OOF embeddings + meta-features
    logger.info("Starting Stacking Training...")
    train_stacking(train_df, config)

    # 6. Global Validation & Failure Analysis
    logger.info("Performing Global Validation and Failure Analysis...")

    # Load the OOF data generated during backbone training
    oof_ids, oof_embeddings, oof_targets = load_oof_data(config)

    # Construct a DataFrame for the OOF data
    emb_cols = [f"emb_{i}" for i in range(oof_embeddings.shape[1])]
    df_oof = pd.DataFrame(oof_embeddings, columns=emb_cols)
    df_oof["essay_id"] = oof_ids
    df_oof["target"] = oof_targets

    # Merge with original train_df to get meta-features and fold assignments
    # We use inner join to ensure alignment
    meta_cols = ["word_count", "char_count", "sentence_count", "unique_word_ratio"]
    df_merged = train_df[["essay_id", "fold"] + meta_cols].merge(
        df_oof, on="essay_id", how="inner"
    )

    # Generate predictions using the trained LightGBM models
    # We iterate through each fold, load the corresponding model, and predict on that fold's data
    final_oof_preds = np.zeros(len(df_merged))
    feature_cols = emb_cols + meta_cols

    for fold in range(config.n_folds):
        model_path = os.path.join(config.model_dir, f"lgbm_fold_{fold}.txt")
        if not os.path.exists(model_path):
            logger.warning(
                f"LGBM model for fold {fold} not found. Skipping prediction for this fold."
            )
            continue

        # Load the LightGBM model
        model = lgb.Booster(model_file=model_path)

        # Select rows belonging to this fold
        val_idx = df_merged["fold"] == fold
        X_val = df_merged.loc[val_idx, feature_cols]

        if len(X_val) > 0:
            preds = model.predict(X_val)
            final_oof_preds[val_idx] = preds

    # Compute Final Metric
    # Predictions must be rounded to nearest integer and clipped to [1, 6] for QWK
    final_preds_rounded = np.round(np.clip(final_oof_preds, 1, 6)).astype(int)
    final_metric = compute_qwk(df_merged["target"].values, final_preds_rounded)

    # --- REQUIRED OUTPUT ---
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error and correlate with meta-features
    df_merged["pred"] = final_oof_preds
    df_merged["error"] = (df_merged["target"] - df_merged["pred"]).abs()

    logger.info("Calculating error correlations...")
    correlations = df_merged[meta_cols + ["error"]].corr()["error"]

    print("Error Correlations with Meta-Features:")
    print(correlations)

    # 7. Submission
    # Generate submission only if metric exceeds the specified threshold
    threshold = 0.8246384329994252

    if final_metric > threshold:
        logger.info(
            f"Validation metric {final_metric} exceeds threshold {threshold}. Generating submission..."
        )
        # Execute inference pipeline using the test dataframe and current config
        inference(test_df, config)
    else:
        logger.info(
            f"Validation metric {final_metric} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
