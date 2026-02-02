import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import warnings

# Import provided library modules
from library.config import Config
from library.data_loader import load_data
from library.feature_engineering import extract_features
from library.model_definition import get_logistic_regression_model
from library.training_engine import train_and_predict


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    # Note: Sklearn models use their own random_state parameter which is set in Config


def perform_failure_analysis(y_true, y_prob, text_series):
    """
    Analyzes the correlation between model error and input features.

    Args:
        y_true (np.ndarray): True integer labels.
        y_prob (np.ndarray): Predicted probabilities.
        text_series (pd.Series): Original text data corresponding to the validation set.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Cross-Entropy Loss per sample
    # We extract the probability assigned to the correct class
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)

    # Select the probability of the true class for each sample
    # Advanced indexing: y_prob_clipped[row_indices, class_indices]
    p_true = y_prob_clipped[np.arange(len(y_true)), y_true]
    sample_losses = -np.log(p_true)

    # Extract meta-features from text
    # Handle potential NaNs by filling with empty string
    texts = text_series.fillna("").astype(str)
    char_lens = texts.apply(len).values
    word_counts = texts.apply(lambda x: len(x.split())).values

    # Compute correlations
    if len(sample_losses) > 1:
        corr_char, _ = pearsonr(sample_losses, char_lens)
        corr_word, _ = pearsonr(sample_losses, word_counts)

        print(f"Correlation between Error and Character Length: {corr_char:.6f}")
        print(f"Correlation between Error and Word Count: {corr_word:.6f}")

        # Interpretation
        if abs(corr_char) > 0.1 or abs(corr_word) > 0.1:
            print(
                "Observation: Weak to moderate correlation detected between sentence length and error."
            )
        else:
            print(
                "Observation: No significant correlation between sentence length and error."
            )
    else:
        print("Insufficient samples for correlation analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    warnings.filterwarnings("ignore")

    print("Starting Spooky Author Identification Baseline...")

    # 2. Load Data
    # Utilizing load_cached_data=True to speed up execution if cache exists
    print("Loading datasets...")
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # 3. Feature Extraction
    # This handles vectorization and caching of dense numpy arrays
    print("Extracting features...")
    X_train, y_train, X_val, y_val, X_test, classes = extract_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Validation Phase
    print("Training model for validation...")
    # Initialize model with config parameters
    val_model = get_logistic_regression_model()

    # Train on training split
    val_model.fit(X_train, y_train)

    # Inference on validation split
    # Note: Sklearn runs on CPU. 'n_jobs=-1' in Config handles parallelism.
    print("Running validation inference...")
    y_val_pred_proba = val_model.predict_proba(X_val)

    # Compute Metric
    # log_loss handles the multiclass calculation
    val_metric = log_loss(y_val, y_val_pred_proba)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(y_val, y_val_pred_proba, val_df["text"])

    # 6. Submission Phase
    print("\n--- Generating Submission ---")
    print("Retraining on combined Training and Validation data...")

    # Combine datasets for final training to maximize data usage
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # Generate predictions using the training engine
    # This function handles model retraining, prediction, and saving to CSV
    train_and_predict(
        X_train=X_full,
        y_train=y_full,
        X_test=X_test,
        classes=classes,
        test_ids=test_df["id"],
    )

    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
