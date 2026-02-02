import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import from provided library files
from library import config, utils, data, pipeline, model


def run():
    # 1. Setup
    utils.set_seed(config.SEED)
    logger = utils.setup_logger()
    logger.info(
        "Starting execution of Sanitized Integral-Geometric OAS Discriminant..."
    )

    # 2. Data Loading
    # The data loader handles geometric feature extraction and fusion automatically.
    # It returns float64 dataframes.
    logger.info("Loading Training Data...")
    X_train, y_train, ids_train = data.load_dataset(
        split="train", load_cached_data=True
    )

    logger.info("Loading Validation Data...")
    X_val, y_val, ids_val = data.load_dataset(split="val", load_cached_data=True)

    # 3. Pipeline Construction & Fitting
    # Strategy: VarianceThreshold -> PowerTransformer -> StandardScaler
    logger.info("Initializing and fitting preprocessing pipeline...")
    preprocessor = pipeline.get_preprocessing_pipeline()

    # Fit pipeline on training data only (Inductive)
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform validation data
    X_val_transformed = preprocessor.transform(X_val)

    # 4. Model Training
    # Strategy: OAS Covariance Estimation -> Linear Discriminant Analysis
    logger.info("Initializing and training OAS Discriminant model...")
    clf = model.OASDiscriminant()
    clf.fit(X_train_transformed, y_train)

    # 5. Validation & Metric Calculation
    logger.info("Performing validation inference...")
    # Predict probabilities (returns float64)
    val_probs = clf.predict_proba(X_val_transformed)

    # Compute Log Loss
    # We pass clf.classes_ to ensure correct column mapping
    val_loss = utils.compute_log_loss(y_val, val_probs, labels=clf.classes_)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate per-sample loss (Cross Entropy)
    # 1. Encode y_val to indices matching clf.classes_
    class_map = {label: i for i, label in enumerate(clf.classes_)}
    y_val_indices = np.array([class_map[label] for label in y_val])

    # 2. Extract probability assigned to the true class
    # Clip to avoid log(0)
    eps = 1e-15
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    # 3. Compute negative log likelihood per sample
    sample_losses = -np.log(true_class_probs)

    # 4. Correlate sample loss with features
    # We use the transformed features to see what the model actually saw
    n_features = X_val_transformed.shape[1]
    correlations = []

    # Convert sparse matrix to dense if necessary (though pipeline output is usually dense here)
    if hasattr(X_val_transformed, "toarray"):
        X_val_analysis = X_val_transformed.toarray()
    else:
        X_val_analysis = X_val_transformed

    # Get feature names from the pipeline if possible, or use indices
    # Since VarianceThreshold might drop features, we need to map back carefully or just use indices
    # For simplicity in this report, we use indices of the transformed space
    for i in range(n_features):
        feat_values = X_val_analysis[:, i]
        # Avoid constant features in correlation (though variance threshold should have removed them)
        if np.std(feat_values) > 1e-9:
            corr, _ = pearsonr(sample_losses, feat_values)
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Top 5 Features correlated with Error ---")
    for i, (feat_idx, corr) in enumerate(correlations[:5]):
        print(f"Feature Index {feat_idx}: Correlation {corr:.4f}")
    print("--------------------------------------------------------------\n")

    # 7. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 3.058881515561734e-14

    if val_loss < THRESHOLD:
        logger.info(
            f"Validation metric ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        logger.info("Loading Test Data...")
        X_test, _, ids_test = data.load_dataset(split="test", load_cached_data=True)

        logger.info("Transforming Test Data...")
        X_test_transformed = preprocessor.transform(X_test)

        logger.info("Predicting Test Data...")
        test_probs = clf.predict_proba(X_test_transformed)

        logger.info("Saving Submission...")
        utils.save_submission(
            ids_test, test_probs, clf.classes_, config.SUBMISSION_FILE
        )
        logger.info(f"Submission saved to {config.SUBMISSION_FILE}")

    else:
        logger.info(
            f"Validation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
