import numpy as np
import os
from library.utils import set_seed, save_submission


def predict_ensemble(models, X_test):
    """
    Generates predictions using the Soft-Voting Ensemble strategy.
    Aggregates predictions from Linear, Generative, and Quadratic models.

    Args:
        models (dict): Dictionary containing trained models.
                       Expected keys: 'linear', 'generative', 'quadratic'.
        X_test (np.ndarray): Feature matrix for testing (n_samples, n_features).

    Returns:
        np.ndarray: Averaged and clipped probability matrix (n_samples, n_classes).
    """
    # Ensure reproducibility
    set_seed(42)

    required_keys = ["linear", "generative", "quadratic"]
    # Validate that all ensemble components are present
    for key in required_keys:
        if key not in models:
            raise ValueError(f"Model dictionary missing required key: {key}")

    print(f"Generating predictions for {len(X_test)} samples using ensemble...")

    # 1. Linear Component (Logistic Regression)
    # Provides robust, regularized linear decision boundaries
    print("Predicting with Linear Component...")
    probs_linear = models["linear"].predict_proba(X_test)

    # 2. Generative Component (LDA)
    # Provides density-based estimation, effective for small sample sizes
    print("Predicting with Generative Component...")
    probs_gen = models["generative"].predict_proba(X_test)

    # 3. Quadratic Component (PCA -> Poly -> LR)
    # Captures non-linear feature interactions
    print("Predicting with Quadratic Component...")
    probs_quad = models["quadratic"].predict_proba(X_test)

    # Soft Voting (Average)
    # Combines the strengths of all three approaches
    print("Averaging probabilities...")
    avg_probs = (probs_linear + probs_gen + probs_quad) / 3.0

    # Clipping to avoid log loss extremes
    # Metric requirement: max(min(p, 1-10^-15), 10^-15)
    # This prevents infinite penalties for confident but wrong predictions
    epsilon = 1e-15
    clipped_probs = np.clip(avg_probs, epsilon, 1 - epsilon)

    return clipped_probs


def generate_submission(
    models, X_test, test_ids, classes, output_path="./submission/submission.csv"
):
    """
    Orchestrates the prediction generation and saves the submission file.

    Args:
        models (dict): Dictionary of trained models.
        X_test (np.ndarray): Test features.
        test_ids (np.ndarray): Array of test image IDs.
        classes (np.ndarray): Array of class names corresponding to column indices.
        output_path (str): File path to save the submission CSV.
    """
    # Generate the ensemble predictions
    predictions = predict_ensemble(models, X_test)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save using the provided utility function
    print(f"Saving submission to {output_path}...")
    save_submission(test_ids, predictions, classes, output_path)
