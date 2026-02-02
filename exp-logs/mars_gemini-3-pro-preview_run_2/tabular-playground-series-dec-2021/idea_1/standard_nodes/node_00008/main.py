import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.config import Config
from library.utils import setup_logger, save_submission
from library.data_manager import load_dataset, LabelMapper
from library.model_trainer import GradientBoostingTrainer


def main():
    # Initialize logger
    logger = setup_logger("runfile")
    logger.info("Initializing pipeline...")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    # Load datasets using the data manager.
    # load_cached_data=True allows using pre-processed artifacts if available.
    logger.info("Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_dataset(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    # Initialize the trainer wrapper
    trainer = GradientBoostingTrainer()

    # Train the model
    # The trainer handles the LightGBM training loop, early stopping, and model saving.
    # Config.MODEL_PARAMS ensures GPU usage and specific hyperparameters.
    logger.info("Starting model training...")
    trainer.train(X_train, y_train, X_val, y_val)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing validation inference...")

    # Predict on validation set
    val_pred_indices = trainer.predict(X_val)

    # Calculate Accuracy
    val_accuracy = accuracy_score(y_val, val_pred_indices)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_accuracy}")

    # Failure Analysis
    logger.info("Performing failure analysis...")

    # Define Error: 1 if prediction is wrong, 0 if correct
    errors = (val_pred_indices != y_val).astype(int)

    # Calculate correlation between features and the error signal.
    # We create a temporary DataFrame to leverage pandas' efficient correlation computation.
    analysis_df = X_val.copy()
    analysis_df["error_flag"] = errors

    # Compute correlation of all features with 'error_flag'
    # Drop 'error_flag' from the result to keep only feature correlations
    correlations = analysis_df.corrwith(analysis_df["error_flag"]).drop("error_flag")

    # Identify top 10 features with highest absolute correlation with error
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print(
        "\n--- Failure Analysis: Top 10 Features Correlated with Prediction Error ---"
    )
    for feature_name in top_correlations.index:
        # Print the original signed correlation
        corr_val = correlations[feature_name]
        print(f"{feature_name}: {corr_val:.6f}")
    print(
        "--------------------------------------------------------------------------\n"
    )

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    # Regression Guarding via Conditional Artifact Generation (Cite solution_lesson_node_00002)
    BASELINE_METRIC = 0.9604513888888889

    if val_accuracy > BASELINE_METRIC:
        logger.info(
            f"Validation accuracy ({val_accuracy:.6f}) improved over baseline ({BASELINE_METRIC:.6f}). Generating submission..."
        )

        # Predict on test set
        test_pred_indices = trainer.predict(X_test)

        # Decode predictions: Map 0-indexed integers back to original Cover_Type labels
        logger.info("Mapping predictions to original class labels...")
        final_predictions = LabelMapper.decode(test_pred_indices)

        # Save Submission
        logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
        save_submission(
            ids=test_ids,
            predictions=final_predictions,
            output_path=Config.SUBMISSION_PATH,
            id_col=Config.ID_COL,
            target_col=Config.TARGET_COL,
        )
    else:
        logger.warning(
            f"Validation accuracy ({val_accuracy:.6f}) did not improve over baseline ({BASELINE_METRIC:.6f}). Skipping submission."
        )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
