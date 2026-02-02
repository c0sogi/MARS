import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import library modules
from library.config import Config
from library.utils import seed_everything, save_submission
from library.preprocessing import FeaturePipeline
from library.ensemble_model import SingleLDA
from library.data_loader import _get_class_mapping


def main():
    # 1. Setup and Configuration
    Config.setup()
    seed_everything(Config.SEED)

    print("Initializing pipeline...")

    # 2. Data Loading & Preprocessing
    # The pipeline handles caching, feature extraction (GPU), and transformation
    pipeline = FeaturePipeline()

    # Load Training Data
    # load_cached_data=True ensures we use pre-computed features if available
    print("Loading training data...")
    X_train, y_train, _ = pipeline.get_processed_data("train", load_cached_data=True)

    # 3. Model Training
    print("Training Single LDA Model...")
    # Cite solution_lesson_node_00034: Avoid feature subsampling on separable features
    model = SingleLDA(
        solver=Config.LDA_SOLVER,
        shrinkage=Config.LDA_SHRINKAGE,
    )
    model.fit(X_train, y_train)

    # 4. Validation & Evaluation
    print("Validating model...")
    X_val, y_val, _ = pipeline.get_processed_data("val", load_cached_data=True)

    # Inference on Validation set
    val_probs = model.predict_proba(X_val)

    # Compute Metric (Multi-class Log Loss)
    # model.classes_ corresponds to the column order of val_probs (indices 0..98)
    # Cite debug_lesson_6: Remove Deprecated eps Parameter from Scikit-Learn log_loss
    # Manually clip probabilities to avoid log extremes as per task spec
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    metric = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Cross-Entropy Loss per sample
    # Gather probability of the true class for each sample
    n_samples = len(y_val)
    rows = np.arange(n_samples)

    # Extract prob of true class, clip to avoid log(0)
    true_probs = val_probs[rows, y_val]
    true_probs = np.clip(true_probs, 1e-15, 1.0)
    sample_errors = -np.log(true_probs)

    # Calculate correlation between Error and Features
    # Vectorized approach for efficiency
    # corr(x, y) = cov(x, y) / (std(x) * std(y))

    # Centering
    error_centered = sample_errors - sample_errors.mean()
    X_centered = X_val - X_val.mean(axis=0)

    # Covariance: (F, N) dot (N,) -> (F,)
    covariance = np.dot(X_centered.T, error_centered) / (n_samples - 1)

    # Standard deviations
    error_std = sample_errors.std(ddof=1)
    X_std = X_val.std(axis=0, ddof=1)

    # Calculate correlation vector, handling potential division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_vec = covariance / (X_std * error_std)

    # Replace NaNs (from constant features) with 0
    corr_vec = np.nan_to_num(corr_vec)

    # Get top 5 features correlated with prediction error (magnitude)
    top_indices = np.argsort(np.abs(corr_vec))[::-1][:5]

    print("Top 5 Features correlated with Prediction Error:")
    for idx in top_indices:
        print(f"  Feature {idx}: Correlation = {corr_vec[idx]:.4f}")

    # 6. Submission
    # Cite debug_lesson_5: Avoid Strict Inequality Gating for Critical Artifact Generation
    # Adjusted threshold to ensure submission is generated
    submission_threshold = 100.0

    if metric < submission_threshold:
        print(f"\nGenerating submission (Metric {metric} < {submission_threshold})...")

        # Load Test Data
        X_test, _, ids_test = pipeline.get_processed_data("test", load_cached_data=True)

        # Predict
        test_probs = model.predict_proba(X_test)

        # Get class names for header
        _, class_names = _get_class_mapping(load_cached_data=True)

        # Save
        save_submission(
            ids_test, test_probs, class_names, filename=Config.SUBMISSION_PATH
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {metric} is too high. Submission skipped.")


if __name__ == "__main__":
    main()
