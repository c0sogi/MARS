import os
import numpy as np
import pandas as pd
import warnings
import sys

# Import the provided library modules
from library import config
from library import data_loader
from library import preprocessing
from library import model


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("=== Leaf Classification Pipeline Demo ===")

    # 1. Setup
    # -------------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    set_seed(config.SEED)

    # 2. Verify Configuration
    # -------------------------------------------------------------------------
    print("\n[1/6] Verifying Configuration...")
    features = config.get_ordered_feature_list()
    print(f"   - Total features defined: {len(features)}")

    # Assertions to ensure schema correctness
    assert (
        len(features) == 192
    ), "Error: Expected 192 features (64 margin + 64 shape + 64 texture)."
    assert features[0] == "margin1", "Error: Feature list is not sorted correctly."
    assert os.path.exists(config.INPUT_DIR), "Error: Input directory not found."
    assert os.path.exists(config.METADATA_DIR), "Error: Metadata directory not found."
    print("   - Configuration verified.")

    # 3. Verify Data Loader
    # -------------------------------------------------------------------------
    print("\n[2/6] Testing Data Loader...")
    # Load a small subset of training data to verify loading logic
    X_train_sub, y_train_sub, ids_train_sub = data_loader.load_dataset(
        "train", load_cached_data=False, max_samples=50
    )

    print(f"   - Loaded train subset shape: {X_train_sub.shape}")

    # Validate dimensions and types
    assert X_train_sub.shape == (50, 192), "Error: Train subset shape mismatch."
    assert len(y_train_sub) == 50, "Error: Train target length mismatch."
    assert isinstance(X_train_sub, pd.DataFrame), "Error: X should be a DataFrame."

    # Load test data (should have no targets)
    X_test_sub, y_test_sub, _ = data_loader.load_dataset(
        "test", load_cached_data=False, max_samples=10
    )
    assert y_test_sub is None, "Error: Test dataset should not have targets."
    print("   - Data Loader verified.")

    # 4. Verify Preprocessing
    # -------------------------------------------------------------------------
    print("\n[3/6] Testing Preprocessing Pipeline...")
    # Instantiate the pipeline manually to test fit/transform
    pipeline = preprocessing.GaussianPipeline()

    # Fit on the small subset
    pipeline.fit(X_train_sub)

    # Transform
    X_trans = pipeline.transform(X_train_sub)

    # Check statistics (StandardScaler should make mean ~0 and std ~1)
    # Note: With N=50, this is an approximation, but ensures logic runs.
    mean_val = np.mean(X_trans)
    std_val = np.std(X_trans)
    print(f"   - Transformed Data Mean: {mean_val:.4f} (Expected ~0)")
    print(f"   - Transformed Data Std:  {std_val:.4f} (Expected ~1)")

    assert X_trans.shape == (50, 192), "Error: Transformed shape mismatch."
    assert isinstance(
        X_trans, np.ndarray
    ), "Error: Transformed output should be numpy array."
    print("   - Preprocessing verified.")

    # 5. Verify Model Logic
    # -------------------------------------------------------------------------
    print("\n[4/6] Testing UniformPriorLDA Model...")
    lda_model = model.UniformPriorLDA()

    # Fit on the transformed subset
    lda_model.fit(X_trans, y_train_sub)

    # Predict probabilities
    probas = lda_model.predict_proba(X_trans)

    # Verify probability properties
    n_classes_sub = len(np.unique(y_train_sub))
    print(f"   - Probability matrix shape: {probas.shape}")

    assert probas.shape == (50, n_classes_sub), "Error: Probability shape mismatch."
    assert np.allclose(probas.sum(axis=1), 1.0), "Error: Probabilities do not sum to 1."

    # Verify Clipping
    clipped_probas = model.clip_probabilities(probas)
    assert clipped_probas.min() >= config.EPSILON, "Error: Clipping lower bound failed."
    assert clipped_probas.max() <= (
        1.0 - config.EPSILON
    ), "Error: Clipping upper bound failed."
    print("   - Model logic verified.")

    # 6. Full Pipeline Execution (Train & Evaluate)
    # -------------------------------------------------------------------------
    print("\n[5/6] Running Full Training and Evaluation...")
    # This function:
    # 1. Fits pipeline on full train set
    # 2. Transforms train and val sets
    # 3. Trains LDA on train set
    # 4. Evaluates Log Loss on val set
    # We use load_cached_data=False to ensure we test the computation path.
    trained_model, trained_pipeline, val_loss = model.train_and_evaluate(
        load_cached_data=False
    )

    print(f"   - Final Validation Log Loss: {val_loss:.6f}")

    # Sanity check on metric
    assert val_loss > 0, "Error: Log loss must be positive."
    assert val_loss < 5.0, "Error: Log loss is unusually high, check model convergence."
    print("   - Training and Evaluation verified.")

    # 7. Generate Submission
    # -------------------------------------------------------------------------
    print("\n[6/6] Generating Submission File...")
    model.generate_submission(trained_model, trained_pipeline, load_cached_data=False)

    # Verify file existence and format
    assert os.path.exists(
        config.SUBMISSION_FILE_PATH
    ), "Error: Submission file was not created."

    df_sub = pd.read_csv(config.SUBMISSION_FILE_PATH)
    print(f"   - Submission file shape: {df_sub.shape}")

    # Check columns: ID + 99 Classes
    expected_cols = config.N_CLASSES + 1
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Error: Expected {expected_cols} columns, got {df_sub.shape[1]}."
    assert config.ID_COL in df_sub.columns, "Error: ID column missing from submission."

    # Check that values are within valid range
    feature_cols = [c for c in df_sub.columns if c != config.ID_COL]
    assert (
        df_sub[feature_cols].min().min() >= config.EPSILON
    ), "Error: Submission contains values below epsilon."
    assert df_sub[feature_cols].max().max() <= (
        1.0 - config.EPSILON
    ), "Error: Submission contains values above 1-epsilon."

    print("   - Submission verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
