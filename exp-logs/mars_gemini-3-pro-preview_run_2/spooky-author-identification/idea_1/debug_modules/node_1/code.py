import sys
import os
import pandas as pd
import numpy as np
import random
import shutil

# Ensure the current directory is in the path to import the library correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_datasets
from library.feature_engineering import extract_features
from library.model_trainer import AuthorClassifier
from library.utils import compute_log_loss, save_submission


def setup_environment():
    """
    Configures the environment for a fast demonstration run.
    Modifies Config parameters to use a small subset of data and fewer iterations.
    """
    # 1. Set Random Seeds for Reproducibility
    random.seed(42)
    np.random.seed(42)

    # 2. Clean Working Directory to ensure fresh execution
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Override Config for Speed
    print("Configuring environment for speed optimization...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 200  # Restrict data to 200 samples per split

    # Limit Logistic Regression epochs to 5 for demonstration
    Config.MODEL_PARAMS["max_iter"] = 5

    # Simplify TF-IDF to reduce vectorization time
    Config.WORD_TFIDF_PARAMS["ngram_range"] = (1, 1)  # Only unigrams
    Config.CHAR_TFIDF_PARAMS["ngram_range"] = (2, 3)  # Smaller char n-grams

    print("Configuration complete.")


def main():
    # --- Step 1: Data Loading ---
    print("\n[1/5] Loading Datasets...")
    # Force loading from source (metadata) to demonstrate loader logic
    (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids) = (
        load_datasets(load_cached_data=False, debug=Config.DEBUG)
    )

    # Validation
    print(f"  - Train Shape: {X_train.shape}")
    print(f"  - Val Shape:   {X_val.shape}")
    print(f"  - Test Shape:  {X_test.shape}")

    assert (
        len(X_train) == Config.DEBUG_SAMPLES
    ), "Train set size does not match debug limit."
    assert (
        len(X_val) == Config.DEBUG_SAMPLES
    ), "Val set size does not match debug limit."
    assert len(X_train) == len(y_train), "Mismatch between Train features and labels."
    assert isinstance(X_train, pd.Series), "X_train should be a pandas Series."

    # --- Step 2: Feature Engineering ---
    print("\n[2/5] Extracting Features...")
    # Extract features using the DualStreamVectorizer (Word + Char TF-IDF)
    X_train_vec, X_val_vec, X_test_vec = extract_features(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Validation
    print(f"  - Train Features: {X_train_vec.shape}")
    print(f"  - Val Features:   {X_val_vec.shape}")

    assert X_train_vec.shape[0] == len(
        X_train
    ), "Feature matrix rows must match sample count."
    assert (
        X_train_vec.shape[1] == X_val_vec.shape[1]
    ), "Feature dimension mismatch between Train and Val."
    assert X_test_vec.shape[0] == len(X_test), "Test feature matrix rows mismatch."

    # --- Step 3: Model Training ---
    print("\n[3/5] Training Model...")
    clf = AuthorClassifier()

    # Check if Config override worked
    assert clf.max_iter == 5, "Classifier did not pick up the modified max_iter config."

    # Train with early stopping enabled (patience=2)
    clf.train(
        X_train_vec,
        y_train,
        X_val=X_val_vec,
        y_val=y_val,
        patience=2,
        load_cached_model=False,
    )

    assert clf.is_fitted, "Model should be marked as fitted after training."
    assert hasattr(
        clf.model, "coef_"
    ), "Underlying LogisticRegression model should have coefficients."

    # --- Step 4: Evaluation ---
    print("\n[4/5] Evaluating on Validation Set...")
    # Predict probabilities
    val_probs = clf.predict_proba(X_val_vec)

    # Calculate Log Loss
    loss = compute_log_loss(y_val, val_probs)
    print(f"  - Validation Log Loss: {loss:.5f}")

    assert val_probs.shape == (
        len(X_val),
        3,
    ), "Prediction shape mismatch (should be N_samples x 3 classes)."
    assert isinstance(loss, float), "Loss should be a float."
    assert loss > 0, "Loss should be positive."

    # --- Step 5: Submission Generation ---
    print("\n[5/5] Generating Submission...")
    # Predict on Test Set
    test_probs = clf.predict_proba(X_test_vec)

    # Save to CSV
    save_submission(test_ids, test_probs)

    # Verify Output
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"  - Submission saved to: {submission_path}")
    print(f"  - Submission head:\n{df_sub.head(2)}")

    assert list(df_sub.columns) == [
        "id",
        "EAP",
        "HPL",
        "MWS",
    ], "Submission columns are incorrect."
    assert len(df_sub) == len(test_ids), "Submission row count mismatch."

    print("\n=== Success: Pipeline demonstration completed without errors. ===")


if __name__ == "__main__":
    setup_environment()
    main()
