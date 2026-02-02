import os
import numpy as np
import pandas as pd
from library.utils import set_seed, save_submission
from library.data_loader import load_datasets
from library.feature_engineering import extract_features
from library.model import InsultClassifier


def main():
    # 1. Setup and Reproducibility
    # Set fixed random seed for reproducibility
    set_seed(42)
    print("Initialized environment and seeds.")

    # 2. Data Loading
    # We load a subset of the data (max_samples=500) to ensure the demonstration runs quickly.
    # In a full run, max_samples would be set to None.
    print("Loading datasets...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, test_df) = load_datasets(
        load_cached_data=False,  # Force reload to demonstrate processing
        max_samples=500,
    )

    # Validation: Check data shapes and types
    assert len(X_train_raw) == len(y_train), "Training data and labels length mismatch."
    assert len(X_val_raw) == len(y_val), "Validation data and labels length mismatch."
    assert len(X_test_raw) == len(test_df), "Test data and dataframe length mismatch."
    assert isinstance(X_train_raw[0], str), "Input features should be strings (text)."
    print(
        f"Data loaded successfully. Train size: {len(X_train_raw)}, Val size: {len(X_val_raw)}"
    )

    # 3. Feature Engineering
    # Extract TF-IDF features (Word + Char n-grams)
    print("Extracting features...")
    X_train_feats, X_val_feats, X_test_feats = extract_features(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        load_cached_data=False,  # Force re-computation for demonstration
    )

    # Validation: Check sparse matrix properties
    assert X_train_feats.shape[0] == len(
        y_train
    ), "Feature matrix rows must match label count."
    assert (
        X_train_feats.shape[1] == X_val_feats.shape[1]
    ), "Train and Val feature dimensions mismatch."
    # Check that features are not empty (sparse matrix should have some non-zero entries)
    assert X_train_feats.nnz > 0, "Feature matrix is empty."
    print(f"Features extracted. Dimension: {X_train_feats.shape[1]}")

    # 4. Model Training
    # Initialize and train the Logistic Regression classifier
    print("Training model...")
    classifier = InsultClassifier(C=1.0, class_weight="balanced", random_state=42)

    classifier.fit(X_train_feats, y_train, X_val=X_val_feats, y_val=y_val)

    # 5. Prediction and Evaluation
    print("Generating predictions...")
    # Predict on validation set to sanity check logic
    val_preds = classifier.predict(X_val_feats)

    # Validation: Check prediction range
    assert np.all(
        (val_preds >= 0) & (val_preds <= 1)
    ), "Predictions must be probabilities in [0, 1]."

    # Predict on test set
    test_preds = classifier.predict(X_test_feats)
    assert len(test_preds) == len(
        test_df
    ), "Number of predictions must match test set size."

    # 6. Submission
    print("Saving submission...")
    submission_dir = "./working/submission"
    save_submission(test_preds, test_df, output_dir=submission_dir)

    # Validation: Check if file exists
    submission_path = os.path.join(submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content format
    df_sub = pd.read_csv(submission_path)
    assert "Insult" in df_sub.columns, "Submission missing 'Insult' column."
    assert df_sub.shape[0] == len(test_df), "Submission row count mismatch."

    print("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
