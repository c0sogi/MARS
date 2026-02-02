import os
import numpy as np
import pandas as pd
import warnings
from library.config import Config
from library.data_loader import DataLoader
from library.feature_streams import DualStreamPreprocessor
from library.model_factory import ModelFactory
from library.ensemble_selection import GreedyEnsembleSelector
from library.utils import save_submission, calculate_log_loss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure reproducibility
np.random.seed(Config.RANDOM_STATE)


def main():
    print("=== Starting Demonstration of Leaf Classification Pipeline ===\n")

    # --------------------------------------------------------------------------
    # 1. Verify Configuration and Preprocessing Logic
    # --------------------------------------------------------------------------
    print("[1/5] Verifying Preprocessing Components...")

    # Create dummy data for testing the preprocessor
    dummy_X = np.random.rand(100, 10)

    # Instantiate and fit preprocessor
    preprocessor = DualStreamPreprocessor(
        pt_method="yeo-johnson", qt_n_quantiles=50, dtype=np.float64
    )
    preprocessor.fit(dummy_X)
    transformed = preprocessor.transform(dummy_X)

    # Assertions
    assert (
        "stream_a" in transformed and "stream_b" in transformed
    ), "Preprocessor must return both streams."
    assert transformed["stream_a"].dtype == np.float64, "Stream A must be float64."
    assert transformed["stream_b"].dtype == np.float64, "Stream B must be float64."
    assert transformed["stream_a"].shape == dummy_X.shape, "Transformed shape mismatch."

    print("   DualStreamPreprocessor logic verified.")

    # --------------------------------------------------------------------------
    # 2. Load Phase 1 Data (Model Selection Split)
    # --------------------------------------------------------------------------
    print("\n[2/5] Loading Phase 1 Data (Train/Val Split)...")

    loader = DataLoader()
    # Force load_cached_data=False to ensure the processing logic runs
    train_data, val_data, classes = loader.load_phase1_data(load_cached_data=False)

    # Verify Data Shapes
    n_train = train_data["y"].shape[0]
    n_val = val_data["y"].shape[0]
    n_features = train_data["stream_a"].shape[1]

    print(f"   Train Samples: {n_train}, Val Samples: {n_val}")
    print(f"   Features: {n_features}, Classes: {len(classes)}")

    assert train_data["stream_a"].shape[0] == n_train
    assert val_data["stream_b"].shape[0] == n_val

    # --------------------------------------------------------------------------
    # 3. Initialize Model Library and Run Ensemble Selection
    # --------------------------------------------------------------------------
    print("\n[3/5] Running Greedy Ensemble Selection...")

    # Generate experts
    experts = ModelFactory.generate_expert_library()
    print(f"   Generated {len(experts)} candidate experts.")

    # Initialize Selector
    # We limit max_iterations for this demo to ensure it completes quickly,
    # although the dataset is small enough for the default 100.
    selector = GreedyEnsembleSelector(max_iterations=15, tolerance=1e-6)

    # Fit Selector (Trains experts -> Predicts Val -> Optimizes Ensemble)
    selector.fit(train_data, val_data, experts, load_cached_data=False)

    # Retrieve learned weights
    ensemble_weights = selector.get_selected_config()
    print(f"   Selected Ensemble Weights: {ensemble_weights}")

    assert len(ensemble_weights) > 0, "Ensemble selection failed to select any models."

    # --------------------------------------------------------------------------
    # 4. Phase 2: Retraining on Full Data and Generating Submission
    # --------------------------------------------------------------------------
    print("\n[4/5] Phase 2: Retraining on Full Data & Generating Test Predictions...")

    # Load Phase 2 Data (Full Train + Test)
    full_data, test_data, classes_p2 = loader.load_phase2_data(load_cached_data=False)

    # Verify alignment
    assert np.array_equal(classes, classes_p2), "Class mismatch between phases."

    # Prepare for aggregation
    n_test_samples = test_data["ids"].shape[0]
    n_classes = len(classes)
    final_probs = np.zeros((n_test_samples, n_classes), dtype=Config.NP_DTYPE)
    total_weight = 0.0

    # Iterate through experts, retrain only the selected ones
    print("   Retraining selected experts...")
    for expert_def in experts:
        eid = expert_def["id"]

        if eid in ensemble_weights:
            weight = ensemble_weights[eid]
            stream_name = expert_def["stream"]
            model = expert_def["model"]

            # Select correct stream from full/test data
            X_full = full_data[stream_name]
            y_full = full_data["y"]
            X_test = test_data[stream_name]

            # Retrain on combined Train+Val
            model.fit(X_full, y_full)

            # Predict on Test
            probs = model.predict_proba(X_test).astype(Config.NP_DTYPE)

            # Weighted Accumulation
            final_probs += probs * weight
            total_weight += weight

    # Normalize Probabilities
    if total_weight > 0:
        final_probs /= total_weight
    else:
        raise RuntimeError("Total ensemble weight is zero.")

    print("   Aggregation complete.")

    # --------------------------------------------------------------------------
    # 5. Save and Verify Submission
    # --------------------------------------------------------------------------
    print("\n[5/5] Saving Submission...")

    save_submission(test_data["ids"], classes, final_probs)

    # Verify the output file
    if os.path.exists(Config.SUBMISSION_FILE_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
        print(f"   File saved to: {Config.SUBMISSION_FILE_PATH}")
        print(f"   Submission Dimensions: {df_sub.shape}")

        # Expected: 99 test samples, 99 classes + 1 id column = 100 columns
        expected_rows = 99
        expected_cols = 100

        assert df_sub.shape == (
            expected_rows,
            expected_cols,
        ), f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {df_sub.shape}"

        # Verify ID column
        assert "id" in df_sub.columns, "Submission missing 'id' column."

        print("   Verification Successful.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
