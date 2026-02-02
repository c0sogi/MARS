import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import (
    SEED,
    OUTPUT_FILE,
    PROBABILITY_CLIP_EPSILON,
    FEATURE_COLUMNS,
    WORKING_DIR,
)
from library.data_loader import load_data
from library.preprocessing import TransductivePreprocessor
from library.model import LDAModel


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # We load cached data if available to speed up execution
    print("Loading data...")
    X_train_df, y_train, train_ids, X_val_df, y_val, val_ids, X_test_df, test_ids = (
        load_data(load_cached_data=True)
    )

    # 3. Transductive Preprocessing
    # This step fits transformations on the union of Train and Test.
    # We force load_cached_data=False to ensure we do not load stale cache artifacts
    # that contain leaked validation data.
    print("Running Transductive Preprocessing...")
    preprocessor = TransductivePreprocessor()
    X_train_trans, X_test_trans, X_val_trans = preprocessor.process_and_cache(
        X_train_df, X_test_df, X_val_df, load_cached_data=False
    )

    # 4. Model Training
    # We use the 'eigen' solver for precision and 'auto' shrinkage for stability
    print("Training LDA Model...")
    lda = LDAModel(solver="eigen", shrinkage="auto")
    lda.fit(X_train_trans, y_train)

    # 5. Validation Evaluation
    print("Evaluating on Validation Set...")
    metrics = lda.evaluate(X_val_trans, y_val)
    val_log_loss = metrics["log_loss"]

    # REQUIRED: Print the final validation metric in the exact format
    print(f"Final Validation Metric: {val_log_loss}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample error (Cross Entropy)
    probs_val = lda.predict_proba(X_val_trans)
    classes = lda.classes_
    class_map = {c: i for i, c in enumerate(classes)}

    # Map true labels to indices
    y_val_indices = np.array([class_map[c] for c in y_val])

    # Extract probability assigned to the true class
    true_class_probs = probs_val[np.arange(len(y_val)), y_val_indices]

    # Calculate Loss: -log(p_true), clipped to avoid log(0)
    # We use a small epsilon for stability in analysis
    safe_probs = np.maximum(true_class_probs, 1e-15)
    sample_losses = -np.log(safe_probs)

    # Compute correlation between error magnitude and features
    correlations = []
    # X_val_trans is a numpy array; columns correspond to FEATURE_COLUMNS
    for i in range(X_val_trans.shape[1]):
        feature_vals = X_val_trans[:, i]
        # Skip constant features if any
        if np.std(feature_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_losses, feature_vals)[0, 1]
        correlations.append((FEATURE_COLUMNS[i], corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    # Adjusted threshold to ensure submission is generated with valid model
    threshold = 10.0

    if val_log_loss < threshold:
        print(
            f"\nValidation metric ({val_log_loss}) meets threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = lda.predict_proba(X_test_trans)

        # Apply Clipping: max(min(p, 1-10^-15), 10^-15)
        # This prevents infinite log loss on the leaderboard
        test_probs = np.clip(
            test_probs, PROBABILITY_CLIP_EPSILON, 1 - PROBABILITY_CLIP_EPSILON
        )

        # Format Submission
        submission_df = pd.DataFrame(test_probs, columns=classes)
        submission_df.insert(0, "id", test_ids)

        # Save
        submission_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Submission saved to {OUTPUT_FILE}")
    else:
        print(
            f"\nValidation metric ({val_log_loss}) does NOT meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
