import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config
from library.utils import setup_logger, set_seed
from library.dataset_builder import DatasetBuilder
from library.pipeline_factory import PipelineFactory


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.RANDOM_SEED)

    logger.info("Starting execution...")

    # 2. Load Data
    # DatasetBuilder handles loading metadata, generating/loading embeddings,
    # and constructing the feature matrices.
    logger.info("Loading datasets...")
    builder = DatasetBuilder()
    X_train, y_train, X_val, y_val, X_test, test_ids = builder.build_datasets(
        load_cached_data=True
    )

    logger.info(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # 3. Training
    # We train on X_train and tune hyperparameters using internal CV on X_train.
    # This ensures X_val remains a strict hold-out set for the final metric.
    logger.info("Initializing pipeline and grid search...")

    # Create base pipeline
    pipeline = PipelineFactory.create_pipeline({})

    # Map Config parameter grid to pipeline parameter names
    # The pipeline structure is: preprocessor -> classifier (Bagging) -> estimator (LogReg)
    # Note: BaggingClassifier parameters are prefixed with 'classifier__',
    # and the base estimator parameters inside Bagging are prefixed with 'classifier__estimator__'
    pipeline_grid = {}
    for key, values in Config.PARAM_GRID.items():
        pipeline_grid[f"classifier__estimator__{key}"] = values

    # Configure Grid Search
    # We use 3-fold CV within the training set for hyperparameter tuning
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Starting training (GridSearchCV)...")
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    logger.info(f"Best parameters found: {gs.best_params_}")

    # 4. Validation
    logger.info("Performing validation on hold-out set...")
    # Predict probabilities for the positive class
    y_val_pred = best_model.predict_proba(X_val)[:, 1]

    # Calculate Metric
    val_auc = roc_auc_score(y_val, y_val_pred)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate absolute error (0 to 1)
    errors = np.abs(y_val - y_val_pred)

    # Calculate correlation between features and error
    # X_val is a DataFrame, so we can use corrwith
    # We align the errors series to the dataframe index
    error_series = pd.Series(errors, index=X_val.index, name="error")

    # Compute correlations
    # This might take a moment given the dimensionality (~400 dims)
    correlations = X_val.corrwith(error_series)

    # Sort by magnitude of correlation (absolute value)
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("\n--- Failure Analysis: Top 10 Features Correlated with Error ---")
    print(correlations_abs.head(10))
    print("---------------------------------------------------------------\n")

    # 6. Submission
    threshold = 0.7160806860575912

    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        y_test_pred = best_model.predict_proba(X_test)[:, 1]

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": y_test_pred}
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
