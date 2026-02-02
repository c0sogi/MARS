import os
import sys
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import set_seed, setup_logger, compute_auc
from library.data_loader import load_dataset
from library.feature_extractors import (
    TextEmbedder,
    BayesianSubredditEncoder,
    RankGaussScaler,
)
from library.models import TunedLogisticRegression
from library.trainer import CrossValidationStacker


def run_diagnostic_validation(df_train, df_val, logger):
    """
    Performs a single-split validation (Train on df_train, Eval on df_val)
    to compute the hold-out metric and perform failure analysis.
    """
    logger.info("Starting Diagnostic Validation (Single Split)...")

    y_train = df_train[Config.TARGET_COL].values
    y_val = df_val[Config.TARGET_COL].values

    # 1. Text Features
    # Automatically detect device for inference optimization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for TextEmbedder: {device}")
    embedder = TextEmbedder(device=device)

    # We use specific cache names for this diagnostic split to avoid conflicts
    # with the full-training caches used by the Stacker
    X_text_train = embedder.transform(df_train, cache_name="train_diagnostic")
    X_text_val = embedder.transform(df_val, cache_name="val_diagnostic")

    # Stage 1: Text Expert
    logger.info("Training Diagnostic Text Expert...")
    text_model = TunedLogisticRegression(
        param_grid=Config.TEXT_EXPERT_GRID, random_state=Config.RANDOM_SEED
    )
    text_model.fit(X_text_train, y_train)

    # Get probabilities (feature for meta-learner)
    prob_text_train = text_model.predict_proba(X_text_train)[:, 1]
    prob_text_val = text_model.predict_proba(X_text_val)[:, 1]

    # 2. History Features
    logger.info("Training Diagnostic History Expert...")
    history_encoder = BayesianSubredditEncoder()
    # Pass y as Series as required by the encoder
    history_encoder.fit(df_train, pd.Series(y_train))

    score_hist_train = history_encoder.transform(df_train)
    score_hist_val = history_encoder.transform(df_val)

    # 3. Metadata Features
    logger.info("Scaling Diagnostic Metadata...")
    scaler = RankGaussScaler()
    scaler.fit(df_train)

    meta_train = scaler.transform(df_train)
    meta_val = scaler.transform(df_val)

    # 4. Fusion (Passthrough Stacking)
    # Combine: [Text_Prob, History_Score, Metadata_Scaled]
    X_meta_train = np.hstack(
        [prob_text_train.reshape(-1, 1), score_hist_train, meta_train]
    )
    X_meta_val = np.hstack([prob_text_val.reshape(-1, 1), score_hist_val, meta_val])

    # 5. Stage 2: Meta-Learner
    logger.info("Training Diagnostic Meta-Learner...")
    meta_model = TunedLogisticRegression(
        param_grid=Config.META_LEARNER_GRID, random_state=Config.RANDOM_SEED
    )
    meta_model.fit(X_meta_train, y_train)

    # Inference on Hold-out Set
    val_preds = meta_model.predict_proba(X_meta_val)[:, 1]

    # Compute Metric
    val_auc = compute_auc(y_val, val_preds)

    # --- Failure Analysis ---
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_val - val_preds)

    # Correlate errors with numerical features in df_val
    numeric_cols = Config.NUMERIC_COLS
    correlations = {}
    for col in numeric_cols:
        if col in df_val.columns:
            # Handle potential NaNs for correlation calculation
            series = df_val[col].fillna(df_val[col].median())
            corr = series.corr(pd.Series(errors, index=df_val.index))
            correlations[col] = corr

    # Sort correlations by magnitude
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # --- Reporting ---
    print("\n" + "=" * 30)
    print(f"Final Validation Metric: {val_auc}")
    print("=" * 30)

    print("\nFailure Analysis - Top Feature Correlations with Error:")
    for feat, corr in sorted_corr[:5]:
        print(f"{feat}: {corr:.4f}")
    print("=" * 30 + "\n")

    return val_auc


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.RANDOM_SEED)
    logger = setup_logger("runfile")

    # 2. Load Data
    # Using cached data if available for speed
    df_train, df_val, df_test = load_dataset(load_cached_data=True)

    # 3. Run Diagnostic Validation
    # This computes the metric on the hold-out set (df_val) and performs failure analysis
    val_auc = run_diagnostic_validation(df_train, df_val, logger)

    # 4. Submission Logic
    THRESHOLD = 0.7141749705260098

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) > Threshold ({THRESHOLD}). Proceeding to submission generation..."
        )

        # Instantiate the CrossValidationStacker from the library
        # This class handles the 5-Fold CV on merged Train+Val and generates the submission file
        stacker = CrossValidationStacker()
        stacker.run_cv(df_train, df_val, df_test)

    else:
        logger.warning(
            f"Validation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )
        # Ensure no stale submission file exists
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
