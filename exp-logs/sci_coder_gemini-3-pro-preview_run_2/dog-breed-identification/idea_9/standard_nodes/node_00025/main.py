import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library import config
from library import feature_engine
from library import classifier_engine


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(config.SEED)

    # 2. Feature Extraction (Train/Val/Test)
    # We load cached features if available to save time, otherwise compute them.
    # The feature_engine handles the heavy lifting of CNN inference.
    print("Extracting/Loading features...")
    X_train, y_train, train_ids = feature_engine.extract_features(
        "train", load_cached_data=True
    )
    X_val, y_val, val_ids = feature_engine.extract_features(
        "val", load_cached_data=True
    )
    # Test features are needed for submission later
    X_test, _, test_ids = feature_engine.extract_features("test", load_cached_data=True)

    # 3. Model Training
    # The classifier engine trains on train set and evaluates on val set
    # It returns the model and the overall val_loss
    print("Training classifier...")
    model, val_loss = classifier_engine.train_classifier(
        load_cached_data=True, save_model=True
    )

    # 4. Validation Metric Output
    # Required format: Final Validation Metric: <value>
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Predict probabilities on validation set manually to get per-sample errors
    y_val_prob = model.predict_proba(X_val)

    # Calculate Log Loss per sample (Cross Entropy)
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_val_prob = np.clip(y_val_prob, epsilon, 1 - epsilon)

    # Gather probabilities of true classes
    # y_val are integer indices corresponding to the columns of y_val_prob
    # We assume model.classes_ corresponds to 0..N-1 which matches y_val indices
    true_class_probs = y_val_prob[np.arange(len(y_val)), y_val]
    errors = -np.log(true_class_probs)

    # Analysis 1: Correlation with Feature Magnitude (L2 Norm)
    # High magnitude features might indicate stronger signal or outliers.
    feature_magnitudes = np.linalg.norm(X_val, axis=1)
    corr_mag, p_mag = pearsonr(errors, feature_magnitudes)
    print(
        f"Correlation between Error and Feature Magnitude: {corr_mag:.4f} (p={p_mag:.4f})"
    )

    # Analysis 2: Correlation with Class Frequency
    # Check if rare classes have higher errors.
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    val_class_counts = np.array([class_counts.get(y, 0) for y in y_val])
    corr_freq, p_freq = pearsonr(errors, val_class_counts)
    print(
        f"Correlation between Error and Class Frequency: {corr_freq:.4f} (p={p_freq:.4f})"
    )

    # Analysis 3: Correlation with Prediction Confidence (Max Prob)
    # Check if the model is well-calibrated (low confidence when wrong).
    max_probs = np.max(y_val_prob, axis=1)
    corr_conf, p_conf = pearsonr(errors, max_probs)
    print(
        f"Correlation between Error and Prediction Confidence: {corr_conf:.4f} (p={p_conf:.4f})"
    )

    # 6. Submission
    # Only generate submission if validation score is better than the threshold
    THRESHOLD = 0.11640673500383826

    if val_loss < THRESHOLD:
        print(
            f"\nValidation score ({val_loss}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        classifier_engine.predict_submission(model, load_cached_data=True)
    else:
        print(
            f"\nValidation score ({val_loss}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
