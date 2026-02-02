import os
import sys
import numpy as np
import pandas as pd
import random
import warnings

# Ensure the library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_datasets
from library.feature_engineering import extract_features
from library.model_trainer import AuthorClassifier
from library.utils import compute_log_loss, save_submission


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(X_val_text, y_val, val_probs, classes):
    """
    Analyzes validation errors to identify systematic patterns.
    Computes the correlation between error magnitude and input text length.
    """
    print("\n==== Failure Analysis ====")

    # 1. Map string labels to integer indices matching the probability columns
    class_map = {label: idx for idx, label in enumerate(classes)}
    y_indices = y_val.map(class_map).values

    # 2. Extract the predicted probability for the true class
    # val_probs is shape (n_samples, 3), y_indices is shape (n_samples,)
    true_class_probs = val_probs[np.arange(len(y_val)), y_indices]

    # Clip probabilities to avoid log(0)
    eps = 1e-15
    true_class_probs = np.clip(true_class_probs, eps, 1 - eps)

    # 3. Calculate Error Magnitude (Negative Log Likelihood)
    # Higher values indicate worse predictions for the true class
    error_magnitudes = -np.log(true_class_probs)

    # 4. Compute Features for Correlation Analysis
    # We use Text Length (Character Count) and Word Count as proxies for input complexity
    char_counts = X_val_text.astype(str).apply(len).values
    word_counts = X_val_text.astype(str).apply(lambda x: len(x.split())).values

    # 5. Calculate and Print Correlations
    if len(char_counts) > 1:
        char_corr = np.corrcoef(error_magnitudes, char_counts)[0, 1]
        print(
            f"Correlation between Error Magnitude and Text Length (Chars): {char_corr:.6f}"
        )

        word_corr = np.corrcoef(error_magnitudes, word_counts)[0, 1]
        print(f"Correlation between Error Magnitude and Word Count: {word_corr:.6f}")
    else:
        print("Insufficient validation samples for correlation analysis.")


def main():
    # 1. Initialization
    set_seed(Config.SEED)
    warnings.filterwarnings("ignore")
    print(f"Starting execution with SEED={Config.SEED}")

    # 2. Load Data
    # Loads raw text and labels. Uses caching to speed up re-runs.
    print("\n[1/5] Loading Datasets...")
    (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids) = (
        load_datasets(load_cached_data=True)
    )

    # 3. Feature Engineering
    # Converts text to sparse TF-IDF matrices (Word + Char n-grams)
    print("\n[2/5] Extracting Features...")
    X_train_vec, X_val_vec, X_test_vec = extract_features(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Model Training
    # Trains Logistic Regression with Early Stopping
    print("\n[3/5] Training Model...")
    clf = AuthorClassifier()
    clf.train(
        X_train_vec,
        y_train,
        X_val=X_val_vec,
        y_val=y_val,
        patience=5,
        load_cached_model=False,  # Force retrain to ensure baseline validity
    )

    # 5. Validation & Failure Analysis
    print("\n[4/5] Validating and Analyzing Failures...")
    # Inference on validation set
    val_probs = clf.predict_proba(X_val_vec)

    # Compute and print the required metric
    val_loss = compute_log_loss(y_val, val_probs)
    print(f"Final Validation Metric: {val_loss}")

    # Analyze errors
    perform_failure_analysis(X_val, y_val, val_probs, clf.classes_)

    # 6. Test Inference & Submission
    print("\n[5/5] Generating Submission...")
    # Inference on test set
    test_probs = clf.predict_proba(X_test_vec)

    # Save to CSV
    save_submission(test_ids, test_probs)
    print("Runfile execution completed successfully.")


if __name__ == "__main__":
    main()
