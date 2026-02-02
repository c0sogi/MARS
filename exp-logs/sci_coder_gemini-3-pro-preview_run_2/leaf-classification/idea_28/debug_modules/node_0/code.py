import os
import numpy as np
import pandas as pd
from library.config import RANDOM_SEED, SUBMISSION_PATH, ID_COL, TARGET_COL
from library.data_loader import DataManager
from library.model_factory import build_expert_library
from library.ensemble_optimizer import GreedySelector


def run_demo():
    # 1. Setup and Initialization
    print("1. Initializing Demo...")
    np.random.seed(RANDOM_SEED)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading and Processing
    print("\n2. Loading and Processing Data...")
    dm = DataManager()

    # We use load_cached_data=True to utilize any existing pre-computed features
    # in ./working/idea_28/processed_data.npz if available, otherwise it computes them.
    data = dm.load_data(load_cached_data=True)

    # Validate Data Dictionary Structure
    required_keys = ["y_train", "y_val", "test_ids", "classes"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in loaded data.")

    y_train = data["y_train"]
    y_val = data["y_val"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(test_ids)
    n_classes = len(classes)

    print(f"   - Training Samples: {n_train}")
    print(f"   - Validation Samples: {n_val}")
    print(f"   - Test Samples: {n_test}")
    print(f"   - Number of Classes: {n_classes}")

    # Validate Feature Views
    views = ["macro", "micro", "synergistic"]
    for view in views:
        key = f"{view}_X_train"
        if key not in data:
            raise KeyError(f"Missing feature view '{key}'")

        # Check dimensions
        assert (
            data[key].shape[0] == n_train
        ), f"{view} train features row count mismatch"
        assert (
            data[f"{view}_X_val"].shape[0] == n_val
        ), f"{view} val features row count mismatch"
        assert (
            data[f"{view}_X_test"].shape[0] == n_test
        ), f"{view} test features row count mismatch"

    # 3. Build Expert Library
    print("\n3. Building Expert Library...")
    experts = build_expert_library()
    print(f"   - Instantiated {len(experts)} experts.")

    if not experts:
        raise ValueError("Expert library is empty.")

    # 4. Train Experts and Generate Predictions
    print("\n4. Training Experts and Generating Predictions...")
    val_preds_dict = {}
    test_preds_dict = {}

    for name, expert_info in experts.items():
        model = expert_info["model"]
        view_name = expert_info["view"]

        # Retrieve specific view data
        X_train = data[f"{view_name}_X_train"]
        X_val = data[f"{view_name}_X_val"]
        X_test = data[f"{view_name}_X_test"]

        # Fit Model
        # Note: LDA/QDA are very fast, so we fit on the full training set
        model.fit(X_train, y_train)

        # Predict
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)

        # Validate Predictions
        assert val_probs.shape == (
            n_val,
            n_classes,
        ), f"Shape mismatch for {name} val preds"
        assert test_probs.shape == (
            n_test,
            n_classes,
        ), f"Shape mismatch for {name} test preds"
        assert not np.isnan(val_probs).any(), f"NaNs detected in predictions for {name}"

        # Store
        val_preds_dict[name] = val_probs
        test_preds_dict[name] = test_probs

    print("   - All experts trained and evaluated.")

    # 5. Ensemble Optimization
    print("\n5. Optimizing Ensemble Weights...")
    # We use a smaller max_iterations for the demo to ensure speed,
    # though the algorithm is efficient enough for more.
    selector = GreedySelector(max_iterations=20, tolerance=1e-5)

    selector.fit(val_preds_dict, y_val)

    selected_weights = selector.get_selected_weights()
    print(f"   - Selected Ensemble Weights: {selected_weights}")

    if not selected_weights:
        raise RuntimeError("Ensemble selection failed to select any models.")

    # 6. Final Prediction and Submission
    print("\n6. Generating Final Submission...")
    final_test_probs = selector.predict_proba(test_preds_dict)

    # Validate final output
    assert final_test_probs.shape == (n_test, n_classes)
    assert np.all((final_test_probs >= 0) & (final_test_probs <= 1))

    # Construct DataFrame
    submission_df = pd.DataFrame(final_test_probs, columns=classes)
    submission_df.insert(0, ID_COL, test_ids)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"   - Submission saved to: {SUBMISSION_PATH}")

    # Final Verification
    print("\n7. Verifying Submission File...")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_check = pd.read_csv(SUBMISSION_PATH)
    expected_cols = [ID_COL] + list(classes)

    # Check shape
    assert df_check.shape == (n_test, n_classes + 1), "Submission shape mismatch"
    # Check columns
    assert list(df_check.columns) == expected_cols, "Submission columns mismatch"
    # Check ID integrity
    assert np.array_equal(df_check[ID_COL].values, test_ids), "Submission IDs mismatch"

    print("   - Verification Successful.")
    print("\nDemo Complete.")


if __name__ == "__main__":
    run_demo()
