import os
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config, set_seed
from library.model_definitions import PLSSupervisedProjector
from library.training import train_and_evaluate


def verify_pls_logic():
    """
    Verifies the logic of the PLSSupervisedProjector custom transformer.
    Ensures it correctly identifies embedding columns, projects them,
    and concatenates them with metadata.
    """
    print("Verifying PLSSupervisedProjector logic...")

    # 1. Create Synthetic Data
    n_samples = 20
    n_emb_dim = 50
    n_metadata = 5
    n_components = 2

    # Create random embeddings
    emb_data = np.random.rand(n_samples, n_emb_dim)
    emb_cols = [f"emb_{i}" for i in range(n_emb_dim)]

    # Create random metadata
    meta_data = np.random.rand(n_samples, n_metadata)
    meta_cols = [f"meta_{i}" for i in range(n_metadata)]

    # Create random target
    y = np.random.randint(0, 2, size=n_samples)

    # Combine into DataFrame
    df = pd.DataFrame(np.hstack([emb_data, meta_data]), columns=emb_cols + meta_cols)

    # 2. Instantiate Projector
    projector = PLSSupervisedProjector(
        n_components=n_components, embedding_prefix="emb_"
    )

    # 3. Fit
    projector.fit(df, y)

    # Check if columns were correctly identified
    assert (
        len(projector.embedding_cols_) == n_emb_dim
    ), "Failed to identify all embedding columns."
    assert (
        len(projector.metadata_cols_) == n_metadata
    ), "Failed to identify all metadata columns."

    # 4. Transform
    transformed_data = projector.transform(df)

    # 5. Validate Output Shape
    # Expected shape: (n_samples, n_components + n_metadata)
    expected_width = n_components + n_metadata
    assert transformed_data.shape == (
        n_samples,
        expected_width,
    ), f"Shape mismatch. Expected ({n_samples}, {expected_width}), got {transformed_data.shape}"

    print("PLSSupervisedProjector verification passed successfully.\n")


def run_demo_pipeline():
    """
    Runs the full training and evaluation pipeline in debug mode with reduced grids.
    """
    print("Configuring pipeline for fast execution...")

    # Override Config for speed
    # We use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./working/demo_submission/demo_submission.csv"

    # Update cache paths to point to the new working dir
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    # Reduce Hyperparameter Grids to a single iteration for speed
    Config.LOGREG_GRID = {"C": [1.0]}
    Config.SVM_GRID = {"C": [1.0], "gamma": ["scale"]}
    Config.PLS_GRID = {"n_components": [2]}  # Small component count for debug

    # Reduce CV splits
    Config.N_SPLITS = 2

    # Ensure directories exist
    Config.setup()

    print("Starting pipeline execution (Debug Mode)...")
    # Run the main training loop
    # debug=True loads only 50 samples per split
    # load_cached_data=False forces feature extraction to run
    train_and_evaluate(load_cached_data=False, debug=True)

    print("Pipeline execution completed.\n")


def verify_submission():
    """
    Verifies that the submission file exists and has the correct format.
    """
    print("Verifying submission output...")

    sub_path = Config.SUBMISSION_PATH
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    df_sub = pd.read_csv(sub_path)

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check row count
    # In debug mode, we load 50 samples for test.
    expected_rows = 50
    if len(df_sub) != expected_rows:
        raise ValueError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
        )

    # Check probabilities
    probs = df_sub["requester_received_pizza"]
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError("Probabilities out of range [0, 1].")

    print("Submission verification passed successfully.")
    print(f"Submission Head:\n{df_sub.head()}")


if __name__ == "__main__":
    # 1. Set Seed
    set_seed(42)

    # 2. Verify Custom Logic
    verify_pls_logic()

    # 3. Run Pipeline
    run_demo_pipeline()

    # 4. Verify Output
    verify_submission()
