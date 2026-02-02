import os
import numpy as np
import pandas as pd
import cv2
import shutil

# Import provided library modules
from library.config import Config
from library.utils import calculate_log_loss, save_submission
from library.image_features import extract_morphometric_features, process_images
from library.data_manager import DataLoader, GaussianPreprocessor
from library.expert_models import get_lda_expert, get_lr_expert
from library.ensemble_selector import GreedyEnsembleSelector, run_selection_phase


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_utils():
    print("\n--- Testing Utils ---")

    # 1. Test calculate_log_loss
    y_true = np.array([0, 1, 2])
    # Create predictions that don't sum to 1 to test rescaling
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    loss = calculate_log_loss(y_true, y_pred)
    print(f"Log Loss (perfect predictions): {loss:.6f}")
    assert loss < 0.5, "Log loss should be low for good predictions"

    # Test clipping logic with extreme values
    y_pred_extreme = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    loss_extreme = calculate_log_loss(y_true, y_pred_extreme)
    print(f"Log Loss (extreme predictions): {loss_extreme:.6e}")
    assert (
        loss_extreme < 1e-10
    ), "Log loss should be near zero for perfect extreme predictions"

    # 2. Test save_submission
    ids = [1, 2, 3]
    classes = ["ClassA", "ClassB", "ClassC"]
    probs = np.random.rand(3, 3)
    # Normalize manually for this test
    probs = probs / probs.sum(axis=1, keepdims=True)

    dummy_sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(ids, probs, classes, output_path=dummy_sub_path)

    assert os.path.exists(dummy_sub_path), "Submission file was not created"
    df_sub = pd.read_csv(dummy_sub_path)
    assert df_sub.shape == (3, 4), f"Submission shape mismatch: {df_sub.shape}"
    assert list(df_sub.columns) == ["id"] + classes, "Submission columns mismatch"
    print("Utils verified successfully.")


def demo_image_features():
    print("\n--- Testing Image Features ---")

    # Use a real image from the dataset if available, otherwise skip logic check
    # Metadata says images/1.jpg exists
    test_image_rel_path = "images/1.jpg"
    test_image_full_path = os.path.join(Config.INPUT_DIR, test_image_rel_path)

    if os.path.exists(test_image_full_path):
        # 1. Test extract_morphometric_features
        features = extract_morphometric_features(test_image_full_path)
        print(f"Extracted feature vector shape: {features.shape}")
        print(f"First 5 features: {features[:5]}")

        expected_len = 7 + 4  # 7 Hu moments + 4 scalars
        assert (
            len(features) == expected_len
        ), f"Expected {expected_len} features, got {len(features)}"
        assert features.dtype == np.float64, "Features must be float64"

        # 2. Test process_images (batch processing)
        # We'll use a list containing the same image twice
        paths = [test_image_rel_path, test_image_rel_path]

        # Force recompute to test extraction logic, ignore cache for this specific call
        # Note: process_images uses Config.WORKING_DIR for cache.
        # We will define a temporary cache path.
        temp_cache = os.path.join(Config.WORKING_DIR, "demo_features.parquet")
        if os.path.exists(temp_cache):
            os.remove(temp_cache)

        df_features = process_images(
            paths, load_cached_data=False, cache_path=temp_cache
        )

        assert len(df_features) == 2, "DataFrame length mismatch"
        assert df_features.shape[1] == expected_len, "DataFrame column count mismatch"
        assert os.path.exists(temp_cache), "Cache file was not created"

        # Test loading from cache
        df_cached = process_images(paths, load_cached_data=True, cache_path=temp_cache)
        pd.testing.assert_frame_equal(df_features, df_cached)
        print("Image features extraction and caching verified.")
    else:
        print(
            f"Warning: Test image {test_image_full_path} not found. Skipping image feature tests."
        )


def demo_data_manager():
    print("\n--- Testing Data Manager ---")

    loader = DataLoader()

    # 1. Test get_phase1_data with subsampling
    # This internally calls process_images, which might take a moment the first time
    print("Loading Phase 1 data (max_samples=50)...")
    data = loader.get_phase1_data(max_samples=50)

    # Verify structure
    assert "train" in data and "val" in data and "classes" in data
    assert "anchor" in data["train"]
    assert "orthogonal" in data["train"]
    assert "synergistic" in data["train"]
    assert "y" in data["train"]

    # Verify Shapes
    n_train = len(data["train"]["y"])
    n_val = len(data["val"]["y"])
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # Anchor: 64 margin + 64 shape + 64 texture = 192 features
    assert data["train"]["anchor"].shape == (n_train, 192)
    # Orthogonal: 11 extracted features
    assert data["train"]["orthogonal"].shape == (n_train, 11)
    # Synergistic: 192 + 11 = 203 features
    assert data["train"]["synergistic"].shape == (n_train, 203)

    # Verify Data Types (Must be float64)
    assert data["train"]["anchor"].dtype == np.float64

    # 2. Test GaussianPreprocessor explicitly
    print("Testing GaussianPreprocessor...")
    gp = GaussianPreprocessor()
    X_dummy = np.random.rand(20, 5)
    X_trans = gp.fit_transform(X_dummy)

    assert X_trans.shape == X_dummy.shape
    assert X_trans.dtype == np.float64
    # Check if mean is roughly 0 and std roughly 1 (property of standardization after Yeo-Johnson)
    print(f"Transformed Mean: {X_trans.mean():.4f}, Std: {X_trans.std():.4f}")
    assert abs(X_trans.mean()) < 0.5, "Standardization mean check failed"

    print("Data Manager verified.")
    return data


def demo_expert_models(data):
    print("\n--- Testing Expert Models ---")

    X_train = data["train"]["anchor"]
    y_train = data["train"]["y"]
    X_val = data["val"]["anchor"]

    # 1. Test LDA Expert
    print("Training LDA Expert...")
    lda = get_lda_expert()
    lda.fit(X_train, y_train)

    preds_lda = lda.predict_proba(X_val)
    assert preds_lda.shape == (len(X_val), len(data["classes"]))
    assert np.allclose(preds_lda.sum(axis=1), 1.0), "LDA probabilities do not sum to 1"

    # 2. Test LR Expert (Backup)
    print("Training LR Expert...")
    # Use C=1.0 for speed in demo
    lr = get_lr_expert(C=1.0, cv=2)
    lr.fit(X_train, y_train)

    preds_lr = lr.predict_proba(X_val)
    assert preds_lr.shape == (len(X_val), len(data["classes"]))

    print("Expert models verified.")
    return preds_lda, preds_lr


def demo_ensemble_selector(data, preds_lda, preds_lr):
    print("\n--- Testing Ensemble Selector ---")

    y_val = data["val"]["y"]

    # Create a dictionary of predictions
    # For demo purposes, we'll pretend we have 3 experts
    predictions_dict = {
        "Expert_A": preds_lda,
        "Expert_B": preds_lr,
        "Expert_C": preds_lda,  # Duplicate just for testing selection logic
    }

    selector = GreedyEnsembleSelector(n_iterations=5)
    weights = selector.fit(predictions_dict, y_val)

    print("Selected Weights:", weights)
    assert isinstance(weights, dict)
    assert sum(weights.values()) == 5, "Total weights should equal n_iterations"

    print("Ensemble Selector verified.")


def demo_full_pipeline():
    print("\n--- Testing Full Pipeline Integration (run_selection_phase) ---")

    loader = DataLoader()

    # This function orchestrates the whole Phase 1:
    # Load Data -> Train 4 Experts -> Predict Val -> Select Weights
    # We use a small subset via modifying the internal call or just relying on the fact
    # that we can't easily pass max_samples to run_selection_phase without modifying it.
    # However, for this demo, we will instantiate a loader that we've already tested
    # and manually run a simplified version of what run_selection_phase does,
    # OR we rely on the fact that run_selection_phase calls get_phase1_data.

    # To keep this demo fast, we will Monkey Patch the loader's get_phase1_data method
    # to return the small dataset we already loaded.

    original_get_data = loader.get_phase1_data

    # Load small data once
    small_data = loader.get_phase1_data(max_samples=50)

    # Mock the method to return small data
    loader.get_phase1_data = lambda: small_data

    try:
        weights = run_selection_phase(loader)
        print("Pipeline Result (Weights):", weights)
        assert len(weights) > 0, "Selection phase returned no weights"
    finally:
        # Restore original method
        loader.get_phase1_data = original_get_data

    print("Full pipeline verified.")


if __name__ == "__main__":
    # Initialize
    set_seed(42)
    Config.setup()

    # Run Demos
    demo_utils()
    demo_image_features()

    # Data Manager returns data needed for subsequent tests
    data_subset = demo_data_manager()

    # Expert Models returns predictions needed for selector test
    p_lda, p_lr = demo_expert_models(data_subset)

    demo_ensemble_selector(data_subset, p_lda, p_lr)

    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
