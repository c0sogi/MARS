import os
import sys
import numpy as np
import pandas as pd
import shutil
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

# Import from the provided library files
from library.utils import set_seed, create_submission_file
from library.data_loader import load_dataset
from library.model_factory import (
    build_linear_branch,
    build_generative_branch,
    build_kernel_branch,
)
from library.engine import train_ensemble, predict_ensemble


def demonstrate_leaf_classification_task():
    print("Starting Leaf Classification Task Demonstration...")

    # 1. Setup and Seeding
    print("\n[Step 1] Setting random seeds...")
    set_seed(42)

    # Clean up previous cache if exists to demonstrate fresh loading
    cache_dir = "./working/idea_15"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache at {cache_dir}")

    # 2. Data Loading
    print("\n[Step 2] Loading Dataset...")
    # We set load_cached_data=False to force reading from ./metadata/ CSVs
    X_train, y_train, X_test, test_ids, label_encoder = load_dataset(
        load_cached_data=False
    )

    # Verification of Data Loading
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "Mismatch in training samples and labels"
    assert (
        X_train.shape[1] == 192
    ), "Expected 192 features (64 margin + 64 shape + 64 texture)"
    assert X_test.shape[1] == 192, "Expected 192 features in test set"
    assert len(label_encoder.classes_) == 99, "Expected 99 plant species classes"

    # 3. Model Factory Demonstration
    print("\n[Step 3] Inspecting Model Architectures...")

    # Instantiate models to verify structure (without training yet)
    linear_pipeline = build_linear_branch(random_state=42)
    generative_pipeline = build_generative_branch()
    kernel_pipeline = build_kernel_branch(random_state=42)

    # Verify they are pipelines
    assert isinstance(linear_pipeline, Pipeline), "Linear branch must be a Pipeline"
    assert isinstance(
        generative_pipeline, Pipeline
    ), "Generative branch must be a Pipeline"
    assert isinstance(kernel_pipeline, Pipeline), "Kernel branch must be a Pipeline"

    print("Model architectures verified successfully.")

    # 4. Training the Ensemble
    print("\n[Step 4] Training Ensemble (This may take a few minutes)...")
    # train_ensemble handles fitting of all three branches
    models = train_ensemble(X_train, y_train, random_state=42)

    # Verify training results
    assert "linear" in models
    assert "generative" in models
    assert "kernel" in models

    # Check if models are fitted (sklearn raises NotFittedError if not, but we check attributes)
    # LogisticRegressionCV has 'classes_' attribute after fit
    assert hasattr(
        models["linear"].named_steps["classifier"], "classes_"
    ), "Linear model not fitted"
    assert hasattr(
        models["generative"].named_steps["classifier"], "classes_"
    ), "Generative model not fitted"
    assert hasattr(
        models["kernel"].named_steps["classifier"], "classes_"
    ), "Kernel model not fitted"

    print("Ensemble training completed and verified.")

    # 5. Inference
    print("\n[Step 5] Generating Predictions...")
    avg_probs = predict_ensemble(models, X_test)

    # Verify predictions
    assert avg_probs.shape == (
        len(X_test),
        99,
    ), f"Prediction shape mismatch. Expected ({len(X_test)}, 99), got {avg_probs.shape}"
    assert np.all(avg_probs >= 0) and np.all(
        avg_probs <= 1
    ), "Probabilities must be in [0, 1]"

    # Check if probabilities sum roughly to 1 (tolerance for float precision)
    row_sums = np.sum(avg_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Predictions generated and verified.")

    # 6. Submission Creation
    print("\n[Step 6] Creating Submission File...")
    submission_path = "./working/demo_submission/submission.csv"
    class_names = list(label_encoder.classes_)

    create_submission_file(test_ids, class_names, avg_probs, submission_path)

    # Verify file creation and content
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    assert df_sub.shape == (
        99,
        100,
    ), "Submission should have 99 rows and 100 columns (id + 99 classes)"
    assert "id" in df_sub.columns, "'id' column missing"
    assert df_sub.iloc[0]["id"] == test_ids[0], "ID mismatch in submission file"

    # Check a random probability column
    random_class = class_names[0]
    assert random_class in df_sub.columns, f"Class column {random_class} missing"

    print("Submission file verified.")
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    demonstrate_leaf_classification_task()
