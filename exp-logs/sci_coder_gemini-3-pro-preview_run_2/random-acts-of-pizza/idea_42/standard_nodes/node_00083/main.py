import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import embedding_manager
from library import trainer
from library import inference

# Import custom transformers to ensure they are available for joblib deserialization
from library.custom_transformers import ArraySelector, WhitenedPCANormalizer


def main():
    # 1. Setup Environment
    # Ensure reproducibility across all operations
    utils.set_seed(config.SEED)

    # Setup logging
    logger = utils.setup_logger(
        "runfile", os.path.join(config.WORKING_DIR, "runfile.log")
    )
    logger.info("Orchestration script started.")

    # 2. Training Phase
    # Execute the provided training module which runs Stratified CV.
    # We use the full dataset (debug_mode=False) to ensure the baseline performance meets the threshold.
    # The trainer handles feature construction, model training, and saving artifacts.
    logger.info("Initiating training phase...")
    try:
        trainer.run_cv_training(load_cached_data=True, debug_mode=False)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # 3. Validation Phase
    # Evaluate the ensemble on the hold-out validation set defined in metadata/val.csv.
    logger.info("Initiating validation phase...")

    try:
        # Load datasets and embeddings
        # Reloading ensures we have the exact dataframes corresponding to the metadata splits
        train_df, val_df, test_df = data_loader.load_dataset(
            load_cached_data=True, debug_mode=False
        )
        embeddings = embedding_manager.get_embeddings(
            train_df, val_df, test_df, load_cached_data=True
        )

        # Construct Feature Matrix for Validation Set
        # This aligns the validation data with the pipeline input structure
        X_val = trainer.build_feature_matrix(val_df, embeddings, "val")
        y_val = val_df[config.TARGET_COL].values.astype(int)

        # Load Trained Models and Generate Predictions (Ensemble)
        models_dir = os.path.join(config.WORKING_DIR, "models")
        val_preds_accum = np.zeros(len(y_val))
        successful_folds = 0

        # Aggregate predictions from all 5 folds (CV-Bagging)
        for fold in range(config.N_FOLDS):
            model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")
            if not os.path.exists(model_path):
                logger.warning(f"Model for fold {fold} not found at {model_path}")
                continue

            # Load model (Pipeline includes Preprocessing + Classifier)
            model = joblib.load(model_path)

            # Predict probability of success (Class 1)
            # Inference is optimized by the pipeline (no gradient computation needed)
            preds = model.predict_proba(X_val)[:, 1]
            val_preds_accum += preds
            successful_folds += 1

        if successful_folds == 0:
            raise RuntimeError("No models available for validation.")

        # Average predictions to reduce variance
        avg_val_preds = val_preds_accum / successful_folds

        # Calculate ROC AUC Metric
        val_auc = roc_auc_score(y_val, avg_val_preds)

        # REQUIRED OUTPUT: Print the final validation metric
        print(f"Final Validation Metric: {val_auc}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # 4. Failure Analysis
    # Identify systematic errors by correlating prediction error with input features
    logger.info("Performing failure analysis...")
    try:
        # Calculate absolute prediction error
        errors = np.abs(y_val - avg_val_preds)

        # Analyze correlations with numeric metadata features
        numeric_cols = config.NUMERIC_FEATURES
        correlations = {}

        for col in numeric_cols:
            if col in val_df.columns:
                # Extract feature values, handling potential missing values
                feat_vals = val_df[col].fillna(0).values.astype(float)

                # Calculate correlation if the feature is not constant
                if np.std(feat_vals) > 1e-9:
                    corr = np.corrcoef(errors, feat_vals)[0, 1]
                    # Handle potential NaNs from numerical instability
                    if np.isnan(corr):
                        corr = 0.0
                    correlations[col] = corr
                else:
                    correlations[col] = 0.0

        # Sort features by correlation strength (magnitude)
        sorted_corrs = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("Top Feature Correlations with Prediction Error:")
        for feat, corr in sorted_corrs[:5]:
            print(f"{feat}: {corr:.4f}")

    except Exception as e:
        logger.warning(f"Failure analysis encountered an issue: {e}")

    # 5. Submission Phase
    # Generate submission only if the validation metric exceeds the specified threshold
    threshold = 0.7201989696216022

    if val_auc > threshold:
        logger.info(
            f"Validation metric ({val_auc}) meets threshold ({threshold}). Generating submission..."
        )
        try:
            # Invoke the provided inference module to generate test predictions
            inference.generate_submission(load_cached_data=True, debug_mode=False)
            logger.info("Submission generation complete.")
        except Exception as e:
            logger.error(f"Submission generation failed: {e}")
            sys.exit(1)
    else:
        logger.warning(
            f"Validation metric ({val_auc}) is below threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
