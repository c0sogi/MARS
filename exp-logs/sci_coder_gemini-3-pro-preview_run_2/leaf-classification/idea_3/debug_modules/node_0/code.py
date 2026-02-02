import os
import shutil
import numpy as np
import pandas as pd
from library.utils import set_seed, save_submission
from library.feature_pipeline import FeatureProcessor, process_data
from library.model_definitions import get_logistic_regression, get_lda, get_gpc
from library.ensemble_manager import HybridEnsemble, run_pipeline
from library.data_manager import load_and_merge_data

# Constants
DEMO_WORKING_DIR = "./working/demo_run"
DEMO_SUBMISSION_DIR = "./working/demo_submission"
SUBMISSION_FILE = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")
METADATA_DIR = "./metadata"
RANDOM_STATE = 42


def clean_directories():
    """Clean up demo directories to ensure a fresh run."""
    for path in [DEMO_WORKING_DIR, DEMO_SUBMISSION_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def test_feature_processor():
    print("\n=== Testing FeatureProcessor ===")
    # Create synthetic data: 100 samples, 64 features
    X_dummy = np.random.rand(100, 64)
    n_components = 10

    processor = FeatureProcessor(
        n_pca_components=n_components, random_state=RANDOM_STATE
    )

    # Test fit_transform
    X_scaled, X_pca = processor.fit_transform(X_dummy)

    # Assertions
    assert X_scaled.shape == (
        100,
        64,
    ), f"Expected scaled shape (100, 64), got {X_scaled.shape}"
    assert X_pca.shape == (
        100,
        n_components,
    ), f"Expected PCA shape (100, {n_components}), got {X_pca.shape}"

    # Check if scaling worked (mean approx 0, std approx 1)
    assert np.allclose(X_scaled.mean(), 0, atol=1e-1), "Scaler mean is not close to 0"
    assert np.allclose(X_scaled.std(), 1, atol=1e-1), "Scaler std is not close to 1"

    print("FeatureProcessor logic verified.")
    return X_scaled, X_pca


def test_individual_models(X_scaled, X_pca):
    print("\n=== Testing Individual Model Definitions ===")
    # Create synthetic labels: 3 classes
    y_dummy = np.random.randint(0, 3, size=100)

    # 1. Logistic Regression
    lr = get_logistic_regression(random_state=RANDOM_STATE)
    lr.fit(X_scaled, y_dummy)
    probs_lr = lr.predict_proba(X_scaled)
    assert probs_lr.shape == (100, 3), "LR output shape mismatch"
    print("Logistic Regression instantiated and trained successfully.")

    # 2. LDA
    lda = get_lda()
    lda.fit(X_scaled, y_dummy)
    probs_lda = lda.predict_proba(X_scaled)
    assert probs_lda.shape == (100, 3), "LDA output shape mismatch"
    print("LDA instantiated and trained successfully.")

    # 3. GPC
    gpc = get_gpc(random_state=RANDOM_STATE)
    gpc.fit(X_pca, y_dummy)
    probs_gpc = gpc.predict_proba(X_pca)
    assert probs_gpc.shape == (100, 3), "GPC output shape mismatch"
    print("GPC instantiated and trained successfully.")


def test_hybrid_ensemble(X_scaled, X_pca):
    print("\n=== Testing HybridEnsemble ===")
    y_dummy = np.random.randint(0, 3, size=100)

    ensemble = HybridEnsemble(random_state=RANDOM_STATE)

    # Fit
    ensemble.fit(X_scaled, X_pca, y_dummy)

    # Predict
    probs = ensemble.predict_proba(X_scaled, X_pca)

    # Assertions
    assert probs.shape == (100, 3), "Ensemble prediction shape mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"

    print("HybridEnsemble logic verified.")


def test_data_manager():
    print("\n=== Testing Data Manager (Real Data) ===")
    # This tests loading real metadata and merging train+val
    X_train, y_train, X_test, test_ids, label_encoder = load_and_merge_data(
        metadata_dir=METADATA_DIR,
        cache_dir=DEMO_WORKING_DIR,
        load_cached_data=False,  # Force processing
    )

    # Expected counts based on metadata info: Train(712) + Val(179) = 891
    expected_train_samples = 712 + 179

    assert (
        X_train.shape[0] == expected_train_samples
    ), f"Expected {expected_train_samples} training samples, got {X_train.shape[0]}"

    assert len(y_train) == expected_train_samples, "Label count mismatch"
    assert X_test.shape[0] == 99, "Test set size mismatch"

    # Check if label encoder was fitted
    assert hasattr(label_encoder, "classes_"), "LabelEncoder not fitted"
    assert (
        len(label_encoder.classes_) == 99
    ), f"Expected 99 classes, got {len(label_encoder.classes_)}"

    print("Data loading and merging verified.")


def test_full_pipeline():
    print("\n=== Testing Full Pipeline Execution ===")

    # Run the pipeline provided in ensemble_manager
    # This will use the real data, train the ensemble, and save submission
    run_pipeline(
        metadata_dir=METADATA_DIR,
        cache_dir=DEMO_WORKING_DIR,
        submission_path=SUBMISSION_FILE,
        n_pca_components=20,  # Reduced components for speed in demo
        random_state=RANDOM_STATE,
    )

    # Verify submission file
    assert os.path.exists(SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_FILE)

    # Check dimensions: 99 test rows, 1 id col + 99 class cols = 100 cols
    assert df_sub.shape == (
        99,
        100,
    ), f"Submission shape mismatch. Expected (99, 100), got {df_sub.shape}"

    # Check ID column
    assert "id" in df_sub.columns, "id column missing in submission"
    assert df_sub["id"].nunique() == 99, "Duplicate or missing IDs in submission"

    # Check probabilities
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values

    # Allow small floating point tolerance for sum=1 check
    row_sums = probs.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-5
    ), "Submission probabilities do not sum to 1"

    print("Full pipeline execution and submission verification successful.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(RANDOM_STATE)
    clean_directories()

    # 2. Unit Tests with Synthetic Data
    X_scaled_dummy, X_pca_dummy = test_feature_processor()
    test_individual_models(X_scaled_dummy, X_pca_dummy)
    test_hybrid_ensemble(X_scaled_dummy, X_pca_dummy)

    # 3. Integration Tests with Real Data
    test_data_manager()
    test_full_pipeline()

    print("\nAll demonstrations completed successfully.")
