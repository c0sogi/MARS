import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library
from library import config, utils, feature_extractor, classifier


def run_failure_analysis(clf, X_val, y_val):
    """
    Performs failure analysis by calculating the correlation between
    prediction error and input features (specifically metadata).
    """
    print("\n=== Failure Analysis ===")

    # Generate predictions
    try:
        probs = clf.predict_proba(X_val)
    except Exception as e:
        print(f"Error during prediction for failure analysis: {e}")
        return

    # Calculate absolute error
    errors = np.abs(y_val - probs)

    # We focus on the metadata features for interpretability.
    # The first 384 columns are embeddings (SentenceTransformer default).
    # The rest are the numerical columns defined in config.
    embedding_dim = 384
    if X_val.shape[1] <= embedding_dim:
        print("No metadata features found to analyze.")
        return

    metadata_features = X_val[:, embedding_dim:]
    feature_names = config.NUMERICAL_COLS

    if metadata_features.shape[1] != len(feature_names):
        print(
            f"Warning: Feature count mismatch. Expected {len(feature_names)}, got {metadata_features.shape[1]}. Skipping detailed correlation."
        )
        return

    # Create a DataFrame for correlation calculation
    df_analysis = pd.DataFrame(metadata_features, columns=feature_names)
    df_analysis["error"] = errors

    # Calculate correlation with error
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Correlation between Error Magnitude and Metadata Features:")
    print(correlations.to_string())
    print("========================\n")


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    logger = utils.setup_logger("runfile")

    logger.info("Starting pipeline execution...")

    # 2. Feature Generation
    # This step handles:
    # - Data loading
    # - Preprocessing
    # - Siamese Network Fine-Tuning (if not cached)
    # - Embedding generation
    # - Feature concatenation
    fe = feature_extractor.FeatureEngineer()

    # We use load_cached_data=True to use pre-computed artifacts if they exist
    # This respects the "use as much of the available time as possible" by not re-doing work
    # but also satisfies the requirement to run end-to-end.
    try:
        X_train, y_train, X_val, y_val, X_test = fe.generate_features(
            load_cached_data=True
        )
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        sys.exit(1)

    logger.info(
        f"Data Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 3. Model Training
    clf = classifier.PizzaClassifier()

    # Train and optimize the classifier
    # The optimize method performs GridSearchCV internally
    clf.optimize(X_train, y_train)

    # 4. Validation
    # Evaluate returns the AUC score
    val_auc = clf.evaluate(X_val, y_val)

    # REQUIRED: Print the final validation metric in the exact format
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    run_failure_analysis(clf, X_val, y_val)

    # 6. Submission
    # Threshold check as per requirements
    threshold = 0.6994047619047619

    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc}) meets threshold ({threshold}). Generating submission..."
        )
        clf.generate_submission(X_test)
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) does NOT meet threshold ({threshold}). Submission skipped."
        )

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
