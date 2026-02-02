import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import SEED, OUTPUT_FILE, FEATURE_COLUMNS
from library.data_loader import load_data
from library.preprocessing import TransductivePreprocessor
from library.model import LDAModel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Starting Demonstration Script...")
    set_seed(SEED)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    print("\n[Step 1] Loading Data...")
    # We set load_cached_data=False to force the loader to read from metadata CSVs
    # and reconstruct the datasets, ensuring we test the core logic.
    X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids = load_data(
        load_cached_data=False
    )

    # Validation: Check types and shapes
    print(f"  Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Val shape:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test shape:  {X_test.shape}")

    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length"
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"

    # Verify strict alphanumeric sorting of columns as per config
    assert (
        list(X_train.columns) == FEATURE_COLUMNS
    ), "Feature columns are not sorted correctly"

    # ==========================================
    # 2. Transductive Preprocessing
    # ==========================================
    print("\n[Step 2] Applying Transductive Preprocessing...")
    # This step fits a PowerTransformer and StandardScaler on the concatenation
    # of Train + Val + Test to maximize the normality of the feature manifold.
    preprocessor = TransductivePreprocessor()

    X_train_trans, X_test_trans, X_val_trans = preprocessor.process_and_cache(
        X_train, X_test, X_val, load_cached_data=False
    )

    # Validation: Check output stats
    print(f"  Transformed Train shape: {X_train_trans.shape}")

    # Check for NaNs or Infs
    assert not np.isnan(X_train_trans).any(), "Transformed training data contains NaNs"
    assert not np.isinf(X_train_trans).any(), "Transformed training data contains Infs"

    # Check Standardization (Mean ~ 0, Std ~ 1)
    # Note: Since standardization is applied on the full concatenated set,
    # the train subset stats will be close to but not exactly 0/1.
    train_mean = np.mean(X_train_trans)
    train_std = np.std(X_train_trans)
    print(f"  Train Subset Mean: {train_mean:.4f}, Std: {train_std:.4f}")

    assert abs(train_mean) < 0.5, "Data does not appear centered"
    assert 0.5 < train_std < 1.5, "Data does not appear scaled"

    # ==========================================
    # 3. Model Training (LDA)
    # ==========================================
    print("\n[Step 3] Training LDA Model...")
    # Using 'eigen' solver and 'auto' (Ledoit-Wolf) shrinkage for stability
    # on high-dimensional, small-sample data.
    model = LDAModel(solver="eigen", shrinkage="auto")

    model.fit(X_train_trans, y_train)
    print(f"  Model fitted on {len(model.classes_)} classes.")

    # ==========================================
    # 4. Model Evaluation
    # ==========================================
    print("\n[Step 4] Evaluating on Validation Set...")
    metrics = model.evaluate(X_val_trans, y_val)

    # Validation assertions
    assert "log_loss" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.0, "Accuracy should be greater than 0"
    assert metrics["log_loss"] > 0.0, "Log loss should be positive"

    # ==========================================
    # 5. Generating Submission
    # ==========================================
    print("\n[Step 5] Generating Submission...")

    # Predict probabilities for test set
    test_probs = model.predict_proba(X_test_trans)

    assert test_probs.shape == (
        len(X_test),
        len(model.classes_),
    ), f"Probability shape mismatch. Expected {(len(X_test), len(model.classes_))}, got {test_probs.shape}"

    # Create submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", test_ids)

    # Save to disk
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Submission saved to: {OUTPUT_FILE}")

    # Verify file creation
    assert os.path.exists(OUTPUT_FILE), "Submission file was not created"

    # Verify content format
    saved_df = pd.read_csv(OUTPUT_FILE)
    assert saved_df.shape == (
        len(X_test),
        len(model.classes_) + 1,
    ), "Saved submission has incorrect shape"
    assert "id" in saved_df.columns, "id column missing in submission"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
