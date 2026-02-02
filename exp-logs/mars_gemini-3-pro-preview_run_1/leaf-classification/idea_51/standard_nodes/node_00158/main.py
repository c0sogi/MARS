import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import warnings

# Import from the provided library files
from library.config import SEED, SUBMISSION_PATH
from library.preprocessor import get_transformed_data
from library.oas_model import CustomOASDiscriminant


def main():
    # 1. Setup and Configuration
    np.random.seed(SEED)
    warnings.filterwarnings("ignore")

    # 2. Data Loading
    # Load transformed data (float64) using the preprocessor module
    # This handles feature extraction, merging, and pipeline transformations
    X_train, y_train, X_val, y_val, X_test, ids_test = get_transformed_data(
        load_cached_data=True
    )

    # 3. Label Encoding
    # Fit encoder on all known labels to ensure consistency
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val])
    le.fit(all_labels)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    # 4. Model Training
    # Initialize and fit the Custom OAS Discriminant model
    model = CustomOASDiscriminant()
    model.fit(X_train, y_train_enc)

    # 5. Validation
    # Predict probabilities on the validation set
    val_probs = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # We provide the labels argument to ensure the loss function knows the full set of classes
    val_loss = log_loss(y_val_enc, val_probs, labels=np.arange(len(le.classes_)))

    # Print the required metric
    print(f"Final Validation Metric: {val_loss}")

    # 6. Failure Analysis
    # Calculate per-sample error magnitude (negative log likelihood of the true class)
    # Use advanced indexing to get the probability assigned to the true class
    true_class_probs = val_probs[np.arange(len(y_val_enc)), y_val_enc]

    # Clip probabilities to avoid log(0), matching the model's internal clipping
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0)
    sample_errors = -np.log(true_class_probs)

    print("\nFailure Analysis (Correlation with Error Magnitude):")

    # Compute correlation between each feature and the error vector
    n_features = X_val.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = X_val[:, i]
        # Check for constant features to avoid division by zero in correlation
        if np.std(feature_col) < 1e-12:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_col, sample_errors)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Identify top 5 features most positively correlated with error (associated with poor performance)
    top_indices = np.argsort(correlations)[-5:][::-1]

    print("Top 5 features associated with higher error:")
    for idx in top_indices:
        print(f"Feature index {idx}: Correlation = {correlations[idx]:.4f}")

    # 7. Submission Generation
    # Strict threshold check as per requirements
    THRESHOLD = 3.3382359570696616e-14

    if val_loss < THRESHOLD:
        print(
            f"\nValidation metric {val_loss} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Generate predictions for the test set
        test_probs = model.predict_proba(X_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=le.classes_)
        submission_df.insert(0, "id", ids_test.astype(int))

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save submission
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {val_loss} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
