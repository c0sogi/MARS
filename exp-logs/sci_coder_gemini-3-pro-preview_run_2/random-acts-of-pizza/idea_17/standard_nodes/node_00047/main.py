import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.utils import setup_logger, set_seed
from library.trainer import Trainer
from library.predictor import Predictor
from library.feature_engineering import ViewTransformer  # Necessary for joblib loading


def main():
    # 1. Setup
    logger = setup_logger("RunFile")
    set_seed(Config.SEED)

    # 2. Training Phase
    # The Trainer handles 5-fold CV on the training split (metadata/train.csv)
    # and saves the models to disk.
    logger.info("=== Starting Training Phase ===")
    trainer = Trainer()
    trainer.run_training()

    # 3. Hold-out Validation Phase
    # We evaluate the ensemble on the separate validation split (metadata/val.csv).
    logger.info("\n=== Starting Hold-out Validation Phase ===")

    # Load validation data
    X_req_val, X_meta_val, y_val, ids_val = trainer.load_all_data("val")

    # Initialize array for ensemble predictions
    val_preds_sum = np.zeros(len(y_val))
    models_dir = os.path.join(Config.WORKING_DIR, "models")

    # Iterate through saved fold models
    models_found = 0
    for fold_idx in range(Config.N_FOLDS):
        model_path = os.path.join(models_dir, f"model_fold_{fold_idx}.joblib")
        transformer_path = os.path.join(
            models_dir, f"transformer_fold_{fold_idx}.joblib"
        )

        if not os.path.exists(model_path):
            logger.warning(f"Model for fold {fold_idx} not found. Skipping.")
            continue

        # Load model and transformer
        try:
            model = joblib.load(model_path)
            vt = joblib.load(transformer_path)
        except Exception as e:
            logger.error(f"Error loading artifacts for fold {fold_idx}: {e}")
            continue

        # Transform validation data using the fold's transformer
        # Note: The transformer was fitted on the fold's training data
        X_val_fused = vt.transform(X_req_val, X_meta_val)

        # Predict probabilities
        probs = model.predict_proba(X_val_fused)[:, 1]
        val_preds_sum += probs
        models_found += 1

    if models_found == 0:
        logger.error("No models found. Cannot perform validation.")
        return

    # Average predictions (Bagging)
    avg_val_preds = val_preds_sum / models_found

    # Compute Final Metric
    final_metric = roc_auc_score(y_val, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("\n=== Performing Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - avg_val_preds)

    # Create DataFrame with numerical features and error
    # We use the numerical features defined in DataLoader
    feature_names = trainer.data_loader.numerical_features
    df_analysis = pd.DataFrame(X_meta_val, columns=feature_names)
    df_analysis["Error_Magnitude"] = errors

    # Compute correlation between features and error
    correlations = df_analysis.corr()["Error_Magnitude"].drop("Error_Magnitude")

    # Sort by absolute correlation
    correlations_sorted = correlations.abs().sort_values(ascending=False)

    print("Correlation between Input Features and Error Magnitude:")
    print(correlations.loc[correlations_sorted.index].head(10))

    # 5. Submission
    threshold = 0.7141749705260098
    if final_metric > threshold:
        logger.info(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        predictor = Predictor()
        predictor.generate_submission()
    else:
        logger.warning(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
