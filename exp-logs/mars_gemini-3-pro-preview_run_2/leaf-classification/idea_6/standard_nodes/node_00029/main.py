import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from library
from library.config import (
    RANDOM_SEED,
    SUBMISSION_FILE_PATH,
)
from library.utils import calculate_log_loss, save_submission, clip_probabilities
from library.data_loader import load_datasets
from library.models import get_probabilistic_ensemble


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(X_val, y_val, probs, classes):
    """
    Calculates per-sample log loss and correlates it with features to identify
    systematic error patterns.
    """
    print("\nPerforming Failure Analysis...")

    # Map class labels to indices
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_indices = np.array([class_to_idx[label] for label in y_val])

    # Calculate per-sample log loss
    # Clip probabilities for stability
    probs_clipped = clip_probabilities(probs)

    # Extract probability assigned to the true class
    # advanced indexing: [row_indices, col_indices]
    true_class_probs = probs_clipped[np.arange(len(y_val)), y_indices]

    # Log loss = -log(p_true)
    sample_losses = -np.log(true_class_probs)

    # Create DataFrame for correlation analysis
    # We use generic feature names as we are working with numpy arrays
    n_features = X_val.shape[1]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    df_analysis = pd.DataFrame(X_val, columns=feature_names)
    df_analysis["error_magnitude"] = sample_losses

    # Calculate correlations between features and error magnitude
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Identify top features associated with high error (positive correlation)
    top_positive = correlations.sort_values(ascending=False).head(5)

    # Identify top features associated with low error (negative correlation)
    top_negative = correlations.sort_values(ascending=True).head(5)

    print(
        "Top 5 features positively correlated with error (associated with poor performance):"
    )
    print(top_positive)
    print(
        "\nTop 5 features negatively correlated with error (associated with good performance):"
    )
    print(top_negative)


def run_pipeline():
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # STAGE 1: Hyperparameter Tuning & Validation
    # -------------------------------------------------------------------------
    print("=== Stage 1: Tuning & Validation ===")

    # Load Split Data (Train/Val)
    # load_datasets handles scaling internally based on the split
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_datasets(
        load_cached_data=True, combine_train_val=False
    )

    # Initialize Ensemble Components
    lda_model, lr_model = get_probabilistic_ensemble()

    # Train models on the Training Split
    print("Training LDA on split training set...")
    lda_model.fit(X_train, y_train)

    print("Training Logistic Regression CV on split training set...")
    lr_model.fit(X_train, y_train)
    print(f"  Selected C: {lr_model.C_[0]}")

    # Inference on Validation Split
    print("Evaluating on validation set...")
    probs_lda = lda_model.predict_proba(X_val)
    probs_lr = lr_model.predict_proba(X_val)

    # Soft Voting Ensemble: Average the probabilities
    probs_ensemble = (probs_lda + probs_lr) / 2.0

    # Calculate Final Validation Metric
    # calculate_log_loss handles row-normalization and clipping
    val_loss = calculate_log_loss(y_val, probs_ensemble, class_labels=classes)
    print(f"Final Validation Metric: {val_loss}")

    # Perform Failure Analysis on Validation Set
    perform_failure_analysis(X_val, y_val, probs_ensemble, classes)

    # -------------------------------------------------------------------------
    # STAGE 2: Final Training & Submission
    # -------------------------------------------------------------------------
    # Threshold defined in the task description
    THRESHOLD = 0.010054905410813797

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric ({val_loss}) is lower than threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Load Combined Data (Train + Val)
        # This re-loads and re-scales the data on the full set
        print("=== Stage 2: Final Retraining ===")
        X_train_full, y_train_full, _, _, X_test, test_ids, classes = load_datasets(
            load_cached_data=True, combine_train_val=True
        )

        # Re-initialize models to ensure fresh training
        lda_final, lr_final = get_probabilistic_ensemble()

        # Train on Full Dataset
        print("Retraining LDA on full dataset...")
        lda_final.fit(X_train_full, y_train_full)

        print("Retraining Logistic Regression CV on full dataset...")
        lr_final.fit(X_train_full, y_train_full)
        print(f"  Selected C: {lr_final.C_[0]}")

        # Inference on Test Set
        print("Predicting on test set...")
        probs_test_lda = lda_final.predict_proba(X_test)
        probs_test_lr = lr_final.predict_proba(X_test)

        # Soft Voting Ensemble
        probs_test_final = (probs_test_lda + probs_test_lr) / 2.0

        # Save Submission
        save_submission(test_ids, classes, probs_test_final, SUBMISSION_FILE_PATH)

    else:
        print(
            f"\nValidation metric ({val_loss}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
