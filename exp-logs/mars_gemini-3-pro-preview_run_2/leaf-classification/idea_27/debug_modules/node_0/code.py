import os
import sys
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import set_seed, setup_logger, suppress_warnings
from library.preprocessing import RobustPreprocessor
from library.models import ExpertLibrary
from library.ensemble_selection import GreedyEnsembleSelector
from library.pipeline import Pipeline

# Setup basic logging
logger = setup_logger("demo_script")
suppress_warnings()


def configure_demo_environment():
    """
    Overrides Config paths to use a separate demo directory in ./working.
    This ensures we don't overwrite existing work and keeps the demo self-contained.
    """
    logger.info("Configuring demo environment...")

    demo_dir = "./working/demo_run"
    submission_dir = "./working/demo_submission"

    # Clean up previous runs if they exist
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)

    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Monkey-patch Config
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = submission_dir
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Update cache paths
    Config.CACHE_ZERNIKE_TRAIN = os.path.join(demo_dir, "zernike_train.parquet")
    Config.CACHE_ZERNIKE_VAL = os.path.join(demo_dir, "zernike_val.parquet")
    Config.CACHE_ZERNIKE_TEST = os.path.join(demo_dir, "zernike_test.parquet")

    logger.info(f"Working directory set to: {Config.WORKING_DIR}")


def demo_preprocessing():
    """
    Demonstrates and validates the RobustPreprocessor.
    """
    logger.info("\n--- Demo: RobustPreprocessor ---")

    # Generate synthetic data (100 samples, 5 features)
    # Include some negative values to test Yeo-Johnson
    X = np.random.randn(100, 5) * 10

    # Introduce a NaN to test sanitization
    X[0, 0] = np.nan

    preprocessor = RobustPreprocessor(method="yeo-johnson")

    # Fit and Transform
    X_trans = preprocessor.fit_transform(X)

    # Assertions
    assert X_trans.shape == X.shape, "Transformed shape mismatch"
    assert not np.isnan(X_trans).any(), "NaNs found in transformed data"
    assert np.isfinite(X_trans).all(), "Infinite values found in transformed data"
    assert X_trans.dtype == np.float64, "Output data type is not float64"

    logger.info(
        "Preprocessing validation passed: Shape preserved, NaNs handled, dtype correct."
    )


def demo_expert_library():
    """
    Demonstrates and validates the ExpertLibrary factory.
    """
    logger.info("\n--- Demo: ExpertLibrary ---")

    lib = ExpertLibrary()

    # Get Tier 1 Experts (LDA)
    tier1 = lib.get_tier1_experts()
    logger.info(f"Tier 1 Experts retrieved: {list(tier1.keys())}")

    # Get Tier 2 Experts (QDA)
    tier2 = lib.get_tier2_experts()
    logger.info(f"Tier 2 Experts retrieved: {list(tier2.keys())}")

    # Assertions
    assert len(tier1) > 0, "No Tier 1 experts returned"
    assert len(tier2) > 0, "No Tier 2 experts returned"

    # Check if specific strategies exist as per config defaults
    assert "LDA_LedoitWolf" in tier1, "LDA_LedoitWolf missing"
    assert any(k.startswith("QDA_Reg") for k in tier2), "QDA experts missing"

    logger.info("ExpertLibrary validation passed: Models instantiated correctly.")


def demo_ensemble_selection():
    """
    Demonstrates and validates the GreedyEnsembleSelector.
    We create a synthetic scenario where one model is clearly better.
    """
    logger.info("\n--- Demo: GreedyEnsembleSelector ---")

    n_samples = 100
    n_classes = 3

    # Ground truth: Class 0 for first 33, Class 1 for next 33, Class 2 for rest
    y_true = np.zeros(n_samples, dtype=int)
    y_true[33:66] = 1
    y_true[66:] = 2

    # Model A: Perfect predictions (One-hot)
    preds_a = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        preds_a[i, y_true[i]] = 1.0

    # Model B: Random uniform predictions
    preds_b = np.random.rand(n_samples, n_classes)
    preds_b = preds_b / preds_b.sum(axis=1, keepdims=True)

    # Model C: Random predictions (worse)
    preds_c = np.random.rand(n_samples, n_classes)
    preds_c = preds_c / preds_c.sum(axis=1, keepdims=True)

    predictions_dict = {
        "Model_Perfect": preds_a,
        "Model_Random1": preds_b,
        "Model_Random2": preds_c,
    }

    selector = GreedyEnsembleSelector(n_iterations=10, tolerance=1e-6)
    selector.fit(predictions_dict, y_true)

    weights = selector.get_selected_experts()
    logger.info(f"Selected Weights: {weights}")

    # Assertions
    assert "Model_Perfect" in weights, "Perfect model was not selected"
    assert weights["Model_Perfect"] > 0.5, "Perfect model should have dominant weight"

    # Test prediction aggregation
    final_preds = selector.predict(predictions_dict)
    assert final_preds.shape == (n_samples, n_classes), "Prediction shape mismatch"
    assert np.allclose(final_preds.sum(axis=1), 1.0), "Probabilities do not sum to 1"

    logger.info("EnsembleSelection validation passed: Correctly identified best model.")


def demo_full_pipeline():
    """
    Runs the full pipeline using the provided data.
    """
    logger.info("\n--- Demo: Full Pipeline Execution ---")

    pipeline = Pipeline()

    # Run pipeline
    # load_cached=False ensures we actually test the Zernike extraction logic
    # and don't rely on pre-existing files in the working dir.
    pipeline.run(load_cached=False)

    # Verify Submission
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    logger.info(f"Submission loaded. Shape: {df_sub.shape}")

    # Load test metadata to verify ID alignment
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    expected_ids = df_test["id"].values

    # Assertions
    assert df_sub.shape[0] == 99, f"Expected 99 rows, got {df_sub.shape[0]}"
    assert (
        df_sub.shape[1] == 100
    ), f"Expected 100 columns (id + 99 species), got {df_sub.shape[1]}"
    assert "id" in df_sub.columns, "id column missing"
    assert np.array_equal(
        df_sub["id"].values, expected_ids
    ), "Submission IDs do not match Test IDs"

    # Check probability range
    probs = df_sub.drop(columns=["id"]).values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    logger.info("Full Pipeline validation passed: Submission generated successfully.")


def main():
    set_seed(42)
    configure_demo_environment()

    # Run individual component demos
    demo_preprocessing()
    demo_expert_library()
    demo_ensemble_selection()

    # Run integration test
    demo_full_pipeline()

    logger.info("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
