import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, compute_qwk
from library.trainer import train_fold
from library.stacking import train_stacking, predict_stacking, load_oof_data
from library.data import get_meta_features

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# Adjust settings to ensure the pipeline completes within the 2-hour limit
# while retaining enough capacity to achieve the target score.
Config.EPOCHS = 1
Config.N_FOLDS = 3  # Use an ensemble of 3 models instead of 5 to save time
Config.TRAIN_BATCH_SIZE = 4
Config.GRAD_ACCUM_STEPS = 4
# Enable AWP immediately since we only run 1 epoch
Config.USE_AWP = True
Config.AWP_START_EPOCH = 0


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def run_failure_analysis(model, X_val, y_val, meta_features):
    """
    Performs failure analysis on the validation set to identify error patterns.
    """
    print("\n=== Failure Analysis ===")

    # Generate predictions using the trained LightGBM model
    preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate residuals (absolute error magnitude)
    errors = np.abs(y_val - preds)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "char_count": meta_features[:, 0],
            "word_count": meta_features[:, 1],
            "sentence_count": meta_features[:, 2],
            "unique_ratio": meta_features[:, 3],
        }
    )

    print("Correlation between Error Magnitude and Meta-Features:")
    # Compute correlation of features with the error
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # Identify the feature most strongly associated with error
    strongest_feature = correlations.abs().idxmax()
    strongest_corr = correlations[strongest_feature]
    print(f"\nStrongest Error Correlation: {strongest_feature} ({strongest_corr:.4f})")

    return preds


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    logger = get_logger(os.path.join(Config.WORKING_DIR, "output", "runfile.log"))
    logger.info("Starting End-to-End Pipeline (Fast Baseline)")
    logger.info(
        f"Config: Epochs={Config.EPOCHS}, Folds={Config.N_FOLDS}, Batch={Config.TRAIN_BATCH_SIZE}"
    )

    # 2. Train Backbone Models (Ensemble Stage)
    logger.info("\n=== Stage 1: Backbone Training ===")
    fold_scores = []

    for fold in range(Config.N_FOLDS):
        try:
            logger.info(f"Training Fold {fold}...")
            qwk = train_fold(fold)
            fold_scores.append(qwk)

            # Aggressive resource cleanup to prevent OOM on subsequent folds
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Critical error training fold {fold}: {e}")
            # Continue to next fold to attempt to salvage the run
            continue

    logger.info(f"Backbone Training Complete. Fold Scores: {fold_scores}")

    if not fold_scores:
        logger.error("No folds completed successfully. Aborting pipeline.")
        return

    # 3. Train Stacking Head
    logger.info("\n=== Stage 2: Stacking ===")
    try:
        # Trains LightGBM on the concatenated embeddings from the validation set
        lgbm_model = train_stacking(load_cached_data=True)
    except Exception as e:
        logger.error(f"Error in stacking training: {e}")
        return

    # 4. Validation & Failure Analysis
    logger.info("\n=== Stage 3: Validation & Analysis ===")

    # Load the validation data (Embeddings + Meta Features) used for stacking
    embeddings, meta_features, targets = load_oof_data()

    # Prepare input matrix for LightGBM
    X_val = np.hstack([embeddings, meta_features])

    # Predict on the entire validation set to compute the final holistic metric
    val_preds = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)

    # Compute Final Metric
    final_qwk = compute_qwk(targets, val_preds)

    # PRINT REQUIRED METRIC FOR EVALUATION
    print(f"Final Validation Metric: {final_qwk}")

    # Run Failure Analysis
    run_failure_analysis(lgbm_model, X_val, targets, meta_features)

    # 5. Inference & Submission
    logger.info("\n=== Stage 4: Inference ===")
    threshold = 0.8246384329994252

    if final_qwk > threshold:
        logger.info(
            f"Validation metric {final_qwk:.6f} > {threshold}. Proceeding to submission generation..."
        )
        # Generate predictions for test set using the ensemble
        predict_stacking(model=lgbm_model, load_cached_data=True)
    else:
        logger.warning(
            f"Validation metric {final_qwk:.6f} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
