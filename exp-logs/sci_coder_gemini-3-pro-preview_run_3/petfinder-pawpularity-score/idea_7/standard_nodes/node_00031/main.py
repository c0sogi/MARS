import sys
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_rmse
from library.feature_extractor import FeatureEngine
from library.dimensionality_reduction import DimensionalityReducer
from library.ensemble_model import EnsembleTrainer


def main():
    # 1. Setup
    logger = get_logger("runfile")
    seed_everything(Config.SEED)

    logger.info("Initializing Pipeline...")

    # 2. Feature Extraction
    # Extracts SPP features from 4 backbones (Swin, EffNet, DINOv2, CLIP)
    # Uses caching to avoid re-computation
    fe = FeatureEngine()
    fe.run(load_cached_data=True)

    # 3. Dimensionality Reduction & Fusion
    # Applies PCA to image features and concatenates with scaled metadata
    dr = DimensionalityReducer()
    X_train, y_train, X_val, y_val, X_test, ids_test = dr.run(load_cached_data=True)

    # 4. Validation Phase
    # We explicitly train on Train and evaluate on Val to get the hold-out metric
    logger.info("Starting Validation Phase...")
    trainer = EnsembleTrainer()

    # 4.1. CV on Training Set to get OOF preds for Meta-Learner
    # This determines optimal hyperparameters (e.g., LGBM estimators)
    oof_preds_train, avg_best_iter = trainer.train_level1_cv(X_train, y_train)

    # 4.2. Train Meta-Learner on Training OOFs
    meta_model = trainer.train_meta_learner(oof_preds_train, y_train)

    # 4.3. Retrain Level 1 models on full Training set
    models_l1 = trainer.retrain_level1_full(X_train, y_train, avg_best_iter)

    # 4.4. Predict on Hold-out Validation Set
    val_preds = trainer.predict(models_l1, meta_model, X_val)

    # 4.5. Compute and Print Metric
    val_rmse = calculate_rmse(y_val, val_preds)
    print(f"Final Validation Metric: {val_rmse}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")
    errors = np.abs(y_val - val_preds)

    # Metadata features are the last N columns in the feature matrix
    meta_cols = Config.METADATA_COLS
    num_meta = len(meta_cols)
    meta_data_val = X_val[:, -num_meta:]

    correlations = {}
    for i, col_name in enumerate(meta_cols):
        # Calculate correlation between the binary feature and the error magnitude
        feat_vals = meta_data_val[:, i]
        # Handle constant features (std=0) to avoid division by zero
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        correlations[col_name] = corr

    logger.info("Correlation between Error Magnitude and Metadata Features:")
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        logger.info(f"  {name}: {corr:.4f}")

    # 6. Submission Phase
    # Threshold defined in task requirements
    THRESHOLD = 17.429365583625966

    if val_rmse < THRESHOLD:
        logger.info(
            f"Validation RMSE ({val_rmse:.4f}) is below threshold ({THRESHOLD:.4f}). Generating submission..."
        )

        # For the final submission, we retrain the ensemble on the combined Train + Val set
        # to maximize the data used for the test set predictions.
        # trainer.run handles the full pipeline: CV on full data -> Meta Train -> Full Retrain -> Predict Test
        trainer.run(
            X_train, y_train, X_val, y_val, X_test, ids_test, load_cached_data=True
        )

    else:
        logger.warning(
            f"Validation RMSE ({val_rmse:.4f}) did not meet threshold ({THRESHOLD:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
