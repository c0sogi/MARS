import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import warnings

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import (
    SEED,
    SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
    FLOAT_PRECISION,
)
from library.preprocessing import get_preprocessed_data
from library.model import AnalyticalOASLDA
from library.utils import clip_probabilities


# Set random seeds for reproducibility
def set_seed(seed=SEED):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed()
    warnings.filterwarnings("ignore")

    print("=== Starting Leaf Classification Workflow ===")

    # 2. Data Loading & Preprocessing
    # Loads cached data if available, otherwise computes it.
    # The pipeline (Yeo-Johnson + StandardScaler) is fitted on Train and applied to all.
    print("\n[1/5] Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, ids_test = get_preprocessed_data(
        load_cached_data=True
    )

    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # 3. Model Training
    # Initialize the Analytical OAS LDA model (Cite solution_lesson_node_00108)
    print("\n[2/5] Training Analytical OAS LDA...")
    model = AnalyticalOASLDA()

    # Fit the model
    model.fit(X_train, y_train)

    # 4. Validation
    print("\n[3/5] Performing Validation...")

    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val)

    # Clip probabilities to avoid log(0) extremes, consistent with metric definition
    val_probs_clipped = clip_probabilities(val_probs)

    # Calculate Multi-class Log Loss
    # We pass model.classes_ to ensure correct label mapping
    val_metric = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n[4/5] Performing Failure Analysis...")

    # Encode validation labels to indices to identify the probability of the true class
    y_val_indices = model.le_.transform(y_val)

    # Extract the predicted probability for the true class for each sample
    # Array indexing: [row_indices, col_indices]
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val_indices]

    # Calculate per-sample Log Loss (Error Magnitude)
    sample_losses = -np.log(true_class_probs)

    # Calculate correlation between each feature and the error magnitude
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = X_val[:, i]
        # Avoid correlation calculation on constant features
        if np.std(feature_vals) > 1e-12:
            corr, _ = pearsonr(feature_vals, sample_losses)
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for idx, corr in correlations[:5]:
        print(f"  Feature {idx}: Correlation = {corr:.4f}")

    # 6. Submission Generation
    print("\n[5/5] Generating Submission...")

    # Threshold check
    # Updated to the strict threshold required by the task
    THRESHOLD = 1.2136771218566717e-09

    if val_metric < THRESHOLD:
        print(
            f"Validation metric {val_metric} is below threshold {THRESHOLD}. Generating submission file."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)
        test_probs_clipped = clip_probabilities(test_probs)

        # Load Sample Submission to get correct column order
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        # The sample submission columns (excluding 'id') are the target species
        target_cols = [c for c in sample_sub.columns if c != "id"]

        # Create Submission DataFrame
        submission = pd.DataFrame()
        submission["id"] = ids_test

        # Map predicted probabilities to the correct columns
        # model.classes_ contains the sorted class names used during training
        # We assume these match the column names in sample_submission (checked in metadata)
        # We create a dictionary mapping class name to predicted probability column
        pred_dict = {
            cls: test_probs_clipped[:, i] for i, cls in enumerate(model.classes_)
        }

        # Assign columns in the order required by sample_submission
        for col in target_cols:
            if col in pred_dict:
                submission[col] = pred_dict[col]
            else:
                # Fallback for missing classes (should not happen with correct data)
                submission[col] = 0.0

        # Save Submission
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {val_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
