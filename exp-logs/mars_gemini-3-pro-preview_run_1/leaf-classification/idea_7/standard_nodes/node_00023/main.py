import os
import numpy as np
import pandas as pd
from library.config import (
    SEED,
    SUBMISSION_FILE_PATH,
    ID_COL,
    LDA_SOLVER,
    LDA_SHRINKAGE,
)
from library.utils import (
    set_seed,
    calculate_log_loss,
    save_submission,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.data_loader import load_datasets
from library.preprocessor import GlobalPreprocessor


def run():
    # 1. Initialization
    set_seed(SEED)

    # 2. Data Loading
    # Load datasets using the provided data loader
    print("Loading datasets...")
    train_data, val_data, test_data, classes = load_datasets(load_cached_data=True)

    X_train_raw = train_data["X"]
    y_train = train_data["y"]
    X_val_raw = val_data["X"]
    y_val = val_data["y"]
    X_test_raw = test_data["X"]
    test_ids = test_data["ids"]

    # 3. Preprocessing (Validation Phase)
    print("Preprocessing data for validation...")
    preprocessor = GlobalPreprocessor()
    # Fit on Train, Transform Val and Test
    X_train_trans, X_val_trans, X_test_trans = preprocessor.process_and_cache(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training (Validation Phase)
    print("Training Single LDA Model on training set...")
    model = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)
    model.fit(X_train_trans, y_train)

    # 5. Evaluation
    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val_trans)

    # Calculate Log Loss
    val_log_loss = calculate_log_loss(y_val, val_probs, labels=classes)

    # Print Metric with full precision as required
    print(f"Final Validation Metric: {val_log_loss}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Map string labels to indices to extract probability of true class
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_val_indices = np.array([class_to_idx[y] for y in y_val])

    # Extract probability assigned to the true class
    # val_probs is (n_samples, n_classes)
    true_class_probs = val_probs[np.arange(len(y_val_indices)), y_val_indices]

    # Calculate error magnitude (Log Loss per sample)
    # Clip to match the metric definition for stability
    epsilon = 1e-15
    clipped_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)
    sample_errors = -np.log(clipped_probs)

    # Calculate correlation between features and error magnitude
    n_features = X_val_trans.shape[1]
    correlations = []
    for j in range(n_features):
        # Compute correlation between feature column j and error vector
        # Returns matrix [[1, r], [r, 1]], we take [0, 1]
        if np.std(X_val_trans[:, j]) == 0:
            corr = 0
        else:
            corr = np.corrcoef(X_val_trans[:, j], sample_errors)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top 5 features most correlated with error (positive or negative)
    top_corr_indices = np.argsort(np.abs(correlations))[::-1][:5]

    print("Top features correlated with prediction error:")
    for idx in top_corr_indices:
        print(f"Feature Index {idx}: Correlation = {correlations[idx]:.6f}")

    # 7. Submission Generation
    THRESHOLD = 1.4705447816556679e-08

    if val_log_loss < THRESHOLD:
        print(f"\nValidation metric {val_log_loss} meets threshold {THRESHOLD}.")
        print("Proceeding to generate submission with combined data...")

        # Combine Train and Validation data
        X_full_raw = np.vstack((X_train_raw, X_val_raw))
        y_full = np.concatenate((y_train, y_val))

        # Re-initialize preprocessor for full data
        # We disable cache loading to ensure we process the combined dataset fresh
        print("Preprocessing combined dataset...")
        full_preprocessor = GlobalPreprocessor()
        # Pass X_full_raw as both train and val arguments to satisfy signature,
        # though we only care about the fitted transformation on X_full and X_test.
        X_full_trans, _, X_test_final_trans = full_preprocessor.process_and_cache(
            X_full_raw, X_full_raw, X_test_raw, load_cached_data=False
        )

        # Retrain model on full data
        print("Retraining single LDA model on full dataset...")
        final_model = LinearDiscriminantAnalysis(
            solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE
        )
        final_model.fit(X_full_trans, y_full)

        # Generate predictions on test set
        print("Generating test predictions...")
        test_probs = final_model.predict_proba(X_test_final_trans)

        # Save submission
        save_submission(test_ids, test_probs, classes, SUBMISSION_FILE_PATH)

    else:
        print(
            f"\nValidation metric {val_log_loss} does not meet threshold {THRESHOLD}."
        )
        print("Submission skipped.")


if __name__ == "__main__":
    run()
