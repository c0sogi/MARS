import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import library.config as config
import library.utils as utils
import library.preprocessing as pp
import library.model as model_lib


def main():
    # ==========================================
    # 1. Setup and Initialization
    # ==========================================
    print("Initializing Leaf Classification Pipeline...")
    utils.set_seed(42)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Data Loading & Preprocessing
    # ==========================================
    # We use debug=True to load a small subset (50 samples) for rapid demonstration.
    # load_cached_data=False forces the pipeline to run feature extraction from scratch,
    # demonstrating the geometric feature extraction and sanitization logic.
    print("\nLoading and Preprocessing Data (Debug Mode)...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = pp.load_data(
        debug=True, load_cached_data=False
    )

    # ==========================================
    # 3. Data Integrity Verification
    # ==========================================
    print("\nVerifying Data Integrity...")
    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    print(f"Test Shape:  {X_test.shape}")
    print(f"Classes:     {len(classes)} unique species found in subset")

    # Verify precision requirements (Float64)
    assert X_train.dtype == config.FLOAT_PRECISION, "Feature matrix must be float64"
    assert X_test.dtype == config.FLOAT_PRECISION, "Test feature matrix must be float64"

    # Verify Sanitization (No NaNs or Infs allowed)
    assert not np.isnan(X_train).any(), "X_train contains NaNs"
    assert not np.isinf(X_train).any(), "X_train contains Infs"

    # ==========================================
    # 4. Model Training (OAS Discriminant)
    # ==========================================
    print("\nTraining OAS Discriminant Model...")
    clf = model_lib.OASDiscriminant()
    clf.fit(X_train, y_train)

    # Verify model learned attributes
    assert hasattr(clf, "coef_"), "Model failed to compute coefficients"
    assert hasattr(clf, "covariance_estimator_"), "Model failed to estimate covariance"

    # ==========================================
    # 5. Evaluation
    # ==========================================
    print("\nEvaluating on Validation Set...")
    val_probs = clf.predict_proba(X_val)

    # Ensure probabilities are valid
    assert np.allclose(np.sum(val_probs, axis=1), 1.0), "Probabilities do not sum to 1"

    # Calculate Log Loss
    # Note: In debug mode, we may not see all classes. We calculate loss based on
    # the classes present in the training subset.
    try:
        score = log_loss(y_val, val_probs, labels=clf.classes_)
        print(f"Validation Multi-class Log Loss: {score:.4f}")
    except ValueError as e:
        print(f"Metric calculation skipped due to debug subset limitations: {e}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\nGenerating Submission for Test Set...")
    test_probs = clf.predict_proba(X_test)

    # Map model outputs (indices) to class names
    # clf.classes_ contains the integer labels seen during training
    # classes array maps these integers to string names
    model_class_names = classes[clf.classes_.astype(int)]

    # Create Submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=model_class_names)
    submission_df.insert(0, "id", test_ids)

    # In a full run, we would ensure all 99 columns exist.
    # For this demo, we save what the model predicted.

    # Save to disk
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {config.SUBMISSION_PATH}")

    # Verify output file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"
    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
