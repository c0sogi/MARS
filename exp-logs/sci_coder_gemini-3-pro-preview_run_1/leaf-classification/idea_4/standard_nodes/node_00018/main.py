import sys
import os
import numpy as np
import torch
import warnings

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_data
from library.models import GlobalLDA
from library.evaluation import calculate_log_loss, create_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration and Hardware Detection
    # Initialize Config to set seeds and directories
    config = Config()

    # Automatically detect device (Requirement: Detect and utilize available GPU)
    # Note: While we detect the GPU here, the specific GlobalLDA model
    # relies on scikit-learn (CPU-bound). We perform this check to satisfy
    # the environment capability detection requirement.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Load data with caching DISABLED to ensure full dataset is processed
    # Cite solution_lesson_node_00015: Avoid data fragmentation; use global dataset.
    data = load_data(debug=False, load_cached_data=False)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        test_ids,
        species_encoder,
    ) = data

    # 3. Model Training
    # Instantiate the Global LDA model
    # Cite solution_lesson_node_00008: Use standalone global model when ensemble weights skew to 1.0.
    model = GlobalLDA()

    # Fit the model on training data
    model.fit(X_train, y_train)

    # 4. Validation
    # Generate probability predictions for the validation set
    val_probs = model.predict_proba(X_val)

    # Calculate the Multi-class Log Loss metric
    val_loss = calculate_log_loss(y_val, val_probs)

    # Print the validation metric in the required format
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    # We analyze which features correlate with higher error magnitudes.

    # Calculate per-sample error (Negative Log Likelihood of the true class)
    # Clip probabilities to match the metric calculation stability
    eps = Config.PROB_CLIP
    val_probs_clipped = np.clip(val_probs, eps, 1.0 - eps)

    # Extract the probability assigned to the true class for each sample
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val]

    # Error magnitude = -log(p_true)
    sample_errors = -np.log(true_class_probs)

    # Vectorized correlation calculation between Features and Error
    # X_val shape: (N_samples, N_features)
    # sample_errors shape: (N_samples,)

    # Center the data
    X_centered = X_val - X_val.mean(axis=0)
    error_centered = sample_errors - sample_errors.mean()

    # Calculate standard deviations
    X_std = X_val.std(axis=0)
    error_std = sample_errors.std()

    # Avoid division by zero for constant features
    valid_features_mask = X_std > 0
    n_features = X_val.shape[1]
    correlations = np.zeros(n_features)

    if error_std > 0:
        # Covariance = mean(X_centered * error_centered)
        covariance = np.mean(
            X_centered[:, valid_features_mask] * error_centered[:, None], axis=0
        )
        # Correlation = Covariance / (std_X * std_error)
        correlations[valid_features_mask] = covariance / (
            X_std[valid_features_mask] * error_std
        )

    # Identify top correlations
    feature_names = Config.get_feature_columns()
    sorted_indices = np.argsort(correlations)

    # Top 5 Negative Correlations (Low feature value -> High error)
    top_neg_indices = sorted_indices[:5]

    # Top 5 Positive Correlations (High feature value -> High error)
    top_pos_indices = sorted_indices[-5:][::-1]

    print("\nFailure Analysis - Correlation between Feature Value and Error Magnitude:")
    print("Top Positive Correlations (High Feature Value associated with High Error):")
    for idx in top_pos_indices:
        print(f"  {feature_names[idx]}: {correlations[idx]:.4f}")

    print("\nTop Negative Correlations (Low Feature Value associated with High Error):")
    for idx in top_neg_indices:
        print(f"  {feature_names[idx]}: {correlations[idx]:.4f}")

    # 6. Submission Generation
    # Strict threshold check as per instructions
    THRESHOLD = 1.4705447816556679e-08

    if val_loss < THRESHOLD:
        # Generate predictions for the test set
        test_probs = model.predict_proba(X_test)

        # Create and save the submission file
        create_submission(
            test_ids=test_ids,
            predictions=test_probs,
            class_names=species_encoder.classes_,
        )
    else:
        # If metric is not better than the threshold, we do not save the submission.
        pass


if __name__ == "__main__":
    main()
