import numpy as np
import pandas as pd
import scipy.sparse
import sys
import os

# Import from provided libraries
from library.utils import set_seed, compute_auc, save_submission
from library.data_loader import load_datasets
from library.feature_engineering import extract_features
from library.model import InsultClassifier


def main():
    # 1. Setup
    set_seed(42)

    # 2. Load Data
    # We load the full dataset. Given the small size (~4k total rows),
    # there is no need to limit samples for this baseline; it will run in seconds.
    print("Loading datasets...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, test_df) = load_datasets(
        load_cached_data=True
    )

    # 3. Feature Engineering
    # Extracts TF-IDF features (Word + Char n-grams) using the provided library
    print("Extracting features...")
    X_train_feats, X_val_feats, X_test_feats = extract_features(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 4. Model Training
    # Logistic Regression with balanced class weights
    print("Training model...")
    model = InsultClassifier(C=1.0, class_weight="balanced", random_state=42)
    model.fit(X_train_feats, y_train, X_val=X_val_feats, y_val=y_val)

    # 5. Validation Evaluation
    print("Evaluating on validation set...")
    # Predict probabilities (class 1)
    val_probs = model.predict(X_val_feats)
    val_auc = compute_auc(y_val, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Error magnitude: |y_true - y_pred|
    # For binary classification, this highlights confident wrong predictions.
    errors = np.abs(y_val - val_probs)

    # A. Correlation with Comment Length
    # Longer comments might be harder to classify or contain more noise.
    lengths = np.array([len(str(t)) for t in X_val_raw])
    if np.std(lengths) > 0:
        corr_len = np.corrcoef(lengths, errors)[0, 1]
        print(f"Correlation between Error and Comment Length: {corr_len:.4f}")

    # B. Correlation with Top Features
    # Identify if specific strong features are associated with errors.
    if hasattr(model.model, "coef_"):
        coefs = model.model.coef_.flatten()
        # Get indices of the top 5 features with the highest absolute weight
        top_indices = np.argsort(np.abs(coefs))[-5:][::-1]

        print("Correlation between Error and Top 5 Features (by weight magnitude):")
        for idx in top_indices:
            # Extract the specific feature column from the sparse validation matrix
            # We convert to dense to compute correlation
            feature_values = X_val_feats[:, idx].toarray().flatten()

            # Only compute if there is variance in the feature
            if np.std(feature_values) > 0:
                corr = np.corrcoef(feature_values, errors)[0, 1]
                weight = coefs[idx]
                print(
                    f"  Feature Idx {idx} (Weight: {weight:.4f}): Correlation {corr:.4f}"
                )
            else:
                print(f"  Feature Idx {idx}: No variance in validation set.")

    # 7. Submission
    # Only generate submission if validation AUC improves upon the baseline
    baseline_auc = 0.8992692939244664
    if val_auc > baseline_auc:
        print(
            f"\nValidation AUC ({val_auc:.6f}) improved over baseline ({baseline_auc:.6f})."
        )
        print("Generating submission...")
        test_probs = model.predict(X_test_feats)
        save_submission(test_probs, test_df)
    else:
        print(
            f"\nValidation AUC ({val_auc:.6f}) did not improve over baseline ({baseline_auc:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
