import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.data_loader import load_datasets
from library.transductive_preprocessor import TransductivePipeline
from library.model import RobustLDA


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting execution...")

    # 2. Load Data
    # Uses caching to speed up execution if data was previously processed
    print("Loading datasets...")
    (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test) = (
        load_datasets(load_cached_data=True)
    )

    # 3. Preprocessing
    # We disable cache loading here to ensure we don't load the old "leaked" matrices.
    # Cite debug_lesson_2: Invalidate Stale Cache Artifacts.
    print("Applying Gaussianization Preprocessing...")
    pipeline = TransductivePipeline()
    X_train_trans, X_val_trans, X_test_trans = pipeline.fit_transform_combined(
        X_train, X_val, X_test, load_cached_data=False
    )

    # 4. Model Training
    # Train the Robust LDA model on the transformed training set
    print("Training Robust LDA Model...")
    model = RobustLDA()
    model.fit(X_train_trans, y_train)

    # 5. Validation
    # Evaluate on the hold-out validation set
    print("Evaluating on Validation Set...")
    metrics = model.evaluate(X_val_trans, y_val, dataset_name="Validation")
    val_loss = metrics["log_loss"]

    # REQUIRED: Print final metric in the exact format
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Compute error magnitude (log loss) per sample
    probs = model.predict_proba(X_val_trans)

    # Map string labels to integer indices for extraction
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract probability assigned to the true class
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(probs, 1e-15, 1.0)
    true_class_probs = probs_clipped[np.arange(len(y_val)), y_val_indices]

    # Error magnitude = -log(p_true)
    error_magnitudes = -np.log(true_class_probs)

    # Correlate with original features to find sources of error
    print("Calculating feature correlations with error magnitude...")

    # Create a temporary DataFrame for correlation calculation
    corr_df = X_val.copy()
    corr_df["error_magnitude"] = error_magnitudes

    # Calculate Pearson correlations
    correlations = corr_df.corrwith(corr_df["error_magnitude"])
    correlations = correlations.drop("error_magnitude")  # Remove self-correlation

    # Identify top 5 features most associated with error (by absolute correlation)
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features associated with model error:")
    print(top_correlations)

    # 7. Submission
    # We check if the model is performing reasonably well before generating submission.
    # The prompt requires a strict threshold for submission.
    THRESHOLD = 1.4705366241156435e-08

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric {val_loss} is satisfactory (< {THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test_trans)

        # Format Submission
        # The submission must contain 'id' and columns for each species class
        submission_df = pd.DataFrame(test_probs, columns=model.classes_)
        submission_df.insert(0, "id", ids_test)

        # Save to the designated submission path
        submission_path = Config.SUBMISSION_PATH
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric {val_loss} is too high (>= {THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
