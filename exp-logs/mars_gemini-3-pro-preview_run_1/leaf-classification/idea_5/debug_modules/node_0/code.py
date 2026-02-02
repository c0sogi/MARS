import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import library modules
from library.config import Config
from library.data_loader import load_datasets
from library.preprocessor import preprocess_data
from library.model import train_model, FisherGaussianEnsemble
from library.evaluation import calculate_log_loss, create_submission_file

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("=== Starting Fisher-Gaussian Process Pipeline Demo ===\n")

    # 1. Configuration and Setup
    set_seed(Config.RANDOM_SEED)
    print(f"Configuration loaded.")
    print(f"Input Directory: {Config.INPUT_DIR}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    # We load the full dataset. Since N=~712 for training, this is fast enough.
    # Using the full dataset ensures we see all 99 classes, preventing LDA errors.
    print("\n[Step 1] Loading Datasets...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_datasets(
        load_cached_data=False  # Force reload from metadata CSVs for demonstration
    )

    # Verification of loaded data
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    assert X_train.shape[1] == 192, "Incorrect number of features in X_train"
    assert len(y_train) == len(X_train), "Mismatch between X_train and y_train length"
    assert len(np.unique(y_train)) > 1, "Training data must have multiple classes"

    # 3. Preprocessing
    # Applies Yeo-Johnson transformation and Standardization
    print("\n[Step 2] Preprocessing Data...")
    X_train_trans, X_val_trans, X_test_trans = preprocess_data(
        X_train, X_val, X_test, load_cached_data=False  # Force re-computation
    )

    # Verification of preprocessing
    # Check that shapes are preserved
    assert X_train_trans.shape == X_train.shape
    # Check that data has been scaled (mean approx 0, std approx 1)
    # We check a random feature column
    feat_mean = np.mean(X_train_trans[:, 0])
    feat_std = np.std(X_train_trans[:, 0])
    print(
        f"Feature 0 Stats after preprocessing: Mean={feat_mean:.4f}, Std={feat_std:.4f}"
    )
    assert np.abs(feat_mean) < 0.1, "Preprocessing failed to center data"
    assert np.abs(feat_std - 1.0) < 0.1, "Preprocessing failed to scale data"

    # 4. Model Training
    # Trains the FisherGaussianEnsemble (LDA Backbone + GPC Head)
    print("\n[Step 3] Training Fisher-Gaussian Ensemble...")
    model = train_model(X_train_trans, y_train, X_val_trans, y_val)

    # Verification of model
    assert isinstance(model, FisherGaussianEnsemble)
    assert hasattr(model, "classes_"), "Model failed to store classes"
    assert hasattr(model, "lda"), "Model missing LDA component"
    assert hasattr(model, "gpc"), "Model missing GPC component"
    print(f"Model trained on {len(model.classes_)} classes.")

    # 5. Evaluation
    print("\n[Step 4] Validating Model...")
    # Generate probabilities for validation set
    val_probs = model.predict_proba(X_val_trans)

    # Verify probabilities
    assert val_probs.shape == (len(X_val), len(model.classes_))
    # Check if rows sum to 1 (within floating point error)
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Calculate Log Loss explicitly
    loss = calculate_log_loss(y_val, val_probs, model.classes_)
    print(f"Manual Log Loss Check: {loss:.4f}")

    # 6. Submission Generation
    print("\n[Step 5] Generating Submission...")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    create_submission_file(model, X_test_trans, test_ids, output_path=submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created at: {submission_path}")
    print(f"Submission shape: {df_sub.shape}")

    # Check columns: id + 99 classes = 100 columns
    expected_cols = 1 + len(model.classes_)
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, found {df_sub.shape[1]}"
    assert "id" in df_sub.columns, "ID column missing from submission"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
