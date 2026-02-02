import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import SEED, SUBMISSION_PATH, SAMPLE_SUBMISSION_PATH
from library.utils import set_seed, get_logger, ensure_dir
from library.data import load_dataset, SanitizedPreprocessor
from library.model import OASDiscriminant


def run():
    # 1. Setup
    set_seed(SEED)
    logger = get_logger("runfile")
    logger.info("Starting runfile execution...")

    # 2. Load Data
    # load_dataset handles feature generation (tabular + geometric) and caching
    logger.info("Loading datasets...")
    X_train, y_train, ids_train = load_dataset("train", load_cached_data=True)
    X_val, y_val, ids_val = load_dataset("val", load_cached_data=True)
    X_test, _, ids_test = load_dataset("test", load_cached_data=True)

    # 3. Preprocessing
    # The SanitizedPreprocessor enforces float64, removes constant features,
    # and applies Yeo-Johnson + StandardScaling.
    logger.info("Preprocessing data...")
    preprocessor = SanitizedPreprocessor()

    # Fit on Train, Transform All
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # 4. Model Training
    # OASDiscriminant uses robust covariance estimation for linear classification
    logger.info("Training OASDiscriminant model...")
    clf = OASDiscriminant()
    clf.fit(X_train_trans, y_train)

    # 5. Validation
    logger.info("Performing validation...")
    # Predict probabilities
    val_probs_df = clf.predict_proba(X_val_trans)
    val_probs = val_probs_df.values

    # Calculate Multi-class Log Loss
    # We pass raw probabilities to log_loss, which handles clipping (eps) internally.
    # Manual clipping after normalization can introduce artifacts (Cite solution_lesson_node_00184).
    score = log_loss(y_val, val_probs, labels=clf.classes_)

    # REQUIRED: Print the final validation metric with full precision
    print(f"Final Validation Metric: {score}")

    # 6. Failure Analysis
    logger.info("Running failure analysis...")

    # Map string labels to integer indices to extract probability of true class
    class_to_idx = {cls: i for i, cls in enumerate(clf.classes_)}
    y_val_indices = np.array([class_to_idx[label] for label in y_val])

    # Get probability assigned to the true class for each sample
    # Use fancy indexing
    prob_true = val_probs[np.arange(len(y_val)), y_val_indices]

    # Calculate error magnitude (Negative Log Likelihood)
    # Clip to avoid log(0) - though log_loss handles this, we do it for analysis
    prob_true_clipped = np.clip(prob_true, 1e-15, 1.0)
    error_magnitude = -np.log(prob_true_clipped)

    # Recover feature names from preprocessor
    # X_train was a DataFrame, so names are preserved in _feature_names_in
    # We need to filter by the variance selector support
    if preprocessor._feature_names_in is not None:
        full_feature_names = np.array(preprocessor._feature_names_in)
        support = preprocessor.variance_selector.get_support()
        kept_feature_names = full_feature_names[support]
    else:
        kept_feature_names = [f"feat_{i}" for i in range(X_val_trans.shape[1])]

    # Calculate correlation between Error Magnitude and each Feature
    correlations = []
    for i in range(X_val_trans.shape[1]):
        feature_values = X_val_trans[:, i]
        # Check for constant features in validation set (unlikely after preprocessing but possible)
        if np.std(feature_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(error_magnitude, feature_values)
        correlations.append((kept_feature_names[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n--- Failure Analysis: Top Correlations with Error Magnitude ---")
    for name, corr in correlations[:5]:
        print(f"Feature: {name}, Correlation: {corr:.4f}")
    print("---------------------------------------------------------------")

    # 7. Submission
    # Strict threshold as per requirements
    SUBMISSION_THRESHOLD = 3.058881515561734e-14

    if score < SUBMISSION_THRESHOLD:
        logger.info("Generating submission file...")
        test_probs_df = clf.predict_proba(X_test_trans)

        # Format submission: id, Class_1, Class_2, ...
        submission = test_probs_df.copy()
        submission.insert(0, "id", ids_test)

        ensure_dir(SUBMISSION_PATH)
        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved successfully to {SUBMISSION_PATH}")
    else:
        logger.warning(f"Validation score {score} is too high. Skipping submission.")


if __name__ == "__main__":
    run()
