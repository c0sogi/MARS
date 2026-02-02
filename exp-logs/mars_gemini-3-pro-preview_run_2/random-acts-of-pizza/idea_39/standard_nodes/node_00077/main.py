import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings

# Import from provided libraries
from library.config import Config
from library.utils import setup_logger, set_seed
from library.embedding_manager import generate_embeddings
from library.pipeline_factory import create_model_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Starting runfile execution...")

    # 2. Load Data and Embeddings
    # Uses cached data if available, or computes it using GPU
    # Returns the fixed splits defined in metadata
    logger.info("Loading data and embeddings...")
    X_train, y_train, X_val, y_val, X_test, schema = generate_embeddings(
        load_cached_data=True
    )

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Val shape: {X_val.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    # 3. Build Pipeline
    # Construct the MF-ADBE pipeline using the factory
    pipeline = create_model_pipeline(schema)

    # 4. Hyperparameter Tuning (Grid Search)
    # Tune on the training set using 3-fold internal CV
    # The pipeline structure is: preprocessor -> classifier (Bagging) -> estimator (LogisticRegression)
    # Config.PARAM_GRID keys are defined as 'estimator__C', etc.
    # We prefix them with 'classifier__' to reach the estimator inside the pipeline step
    param_grid = {f"classifier__{k}": v for k, v in Config.PARAM_GRID.items()}

    logger.info(f"Tuning hyperparameters with grid: {param_grid}")

    # Use n_jobs=1 because the pipeline's BaggingClassifier already uses n_jobs=-1
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        verbose=0,
    )

    logger.info("Fitting Grid Search...")
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    logger.info(f"Best parameters: {gs.best_params_}")

    # 5. Validation
    logger.info("Evaluating on hold-out validation set...")
    # Predict probabilities for the positive class
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate error magnitude (absolute difference between truth and probability)
    errors = np.abs(y_val - val_probs)

    # Correlate errors with the metadata features to find systematic issues
    # Extract metadata columns from X_val using the schema
    meta_start, meta_end = schema["meta"]
    X_val_meta = X_val[:, meta_start:meta_end]

    # Create a DataFrame for correlation analysis
    df_analysis = pd.DataFrame(X_val_meta, columns=Config.NUMERICAL_FEATURES)
    df_analysis["error"] = errors

    # Calculate correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(correlations)

    # 7. Submission
    threshold = 0.7190361601447052

    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = best_model.predict_proba(X_test)[:, 1]

        # Load Test Metadata to ensure correct ID alignment
        if not os.path.exists(Config.METADATA_TEST):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.METADATA_TEST}"
            )

        df_test_meta = pd.read_csv(Config.METADATA_TEST)

        if len(df_test_meta) != len(test_probs):
            raise ValueError(
                f"Mismatch between test metadata rows ({len(df_test_meta)}) and predictions ({len(test_probs)})"
            )

        submission = pd.DataFrame(
            {
                "request_id": df_test_meta["request_id"],
                "requester_received_pizza": test_probs,
            }
        )

        # Save submission file
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
