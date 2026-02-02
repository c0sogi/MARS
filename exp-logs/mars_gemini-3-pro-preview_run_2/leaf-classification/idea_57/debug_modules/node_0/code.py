import os
import sys
import numpy as np
import pandas as pd

# Import provided library components
from library.utils import set_seed, save_submission, clipped_log_loss
from library.data_handler import DataManager
from library.expert_manager import ExpertLibrary
from library.greedy_ensemble import GreedySelector


def run_demo():
    # 1. Setup Environment
    # ----------------------------------------------------------------
    print(">>> Setting up environment...")
    set_seed(42)

    # Define working directory for this run
    work_dir = "./working/demo_run"
    cache_dir = os.path.join(work_dir, "cache")
    os.makedirs(work_dir, exist_ok=True)

    # 2. Data Loading and Processing
    # ----------------------------------------------------------------
    print("\n>>> Initializing DataManager...")
    # We initialize the DataManager which handles loading CSVs,
    # extracting image features (morphometrics), and creating feature views.
    dm = DataManager(cache_dir=cache_dir)

    print("Loading and processing data (Train/Val/Test)...")
    # get_train_val_split calls get_data internally.
    # We set load_cached_data=False to demonstrate processing from scratch,
    # though in practice True is preferred for speed.
    train_data, val_data, classes = dm.get_train_val_split(load_cached_data=False)

    # Retrieve full data dictionary to access test set later
    all_data = dm.get_data(load_cached_data=True)
    test_data = all_data["test"]

    # --- Verification Steps ---
    print(f"Classes found: {len(classes)}")
    print(f"Train samples: {len(train_data['ids'])}")
    print(f"Val samples:   {len(val_data['ids'])}")
    print(f"Test samples:  {len(test_data['ids'])}")

    assert len(train_data["y"]) == len(train_data["ids"])
    assert "global" in train_data["views"]
    assert train_data["views"]["global"].shape[0] == len(train_data["y"])
    # ----------------------------------------------------------------

    # 3. Expert Library Initialization & Training
    # ----------------------------------------------------------------
    print("\n>>> Initializing ExpertLibrary...")
    # The ExpertLibrary manages a collection of pipelines (Global Linear, Rotational, Robust, etc.)
    expert_lib = ExpertLibrary()
    expert_names = expert_lib.get_expert_names()
    print(f"Defined {len(expert_names)} experts: {expert_names[:3]} ...")

    print("Fitting experts on training data...")
    # fit_all iterates through configured pipelines and fits them on specific feature views
    expert_lib.fit_all(train_data)

    # --- Verification Steps ---
    fitted_count = len(expert_lib.fitted_experts)
    print(f"Successfully fitted {fitted_count} experts.")
    assert fitted_count > 0, "No experts were fitted successfully."
    # ----------------------------------------------------------------

    # 4. Validation & Ensemble Optimization
    # ----------------------------------------------------------------
    print("\n>>> Generating Validation Predictions...")
    # Get probability predictions from all experts on the validation set
    val_preds = expert_lib.predict_all(val_data)

    # Check consistency
    first_expert = next(iter(val_preds.keys()))
    n_val_samples, n_classes = val_preds[first_expert].shape
    assert n_val_samples == len(val_data["y"])
    assert n_classes == len(classes)

    print(">>> Optimizing Ensemble (Greedy Forward Selection)...")
    # Initialize the GreedySelector to find optimal weights for the experts
    # We use a smaller max_iterations for this demo to ensure speed
    selector = GreedySelector(max_iterations=15, tolerance=1e-5, verbose=True)

    # Fit the selector using validation predictions and ground truth
    selector.fit(val_preds, val_data["y"])

    print(f"Selected Ensemble Weights: {selector.weights}")

    # --- Verification Steps ---
    if not selector.weights:
        raise RuntimeError("Ensemble selection failed to select any experts.")

    # Calculate validation score of the ensemble
    val_ensemble_probs = selector.predict(val_preds)
    val_score = clipped_log_loss(val_data["y"], val_ensemble_probs)
    print(f"Final Ensemble Validation LogLoss: {val_score:.5f}")
    # ----------------------------------------------------------------

    # 5. Test Prediction & Submission
    # ----------------------------------------------------------------
    print("\n>>> Generating Test Predictions...")
    # Get expert predictions on test data
    test_expert_preds = expert_lib.predict_all(test_data)

    # Aggregate using the trained ensemble weights
    final_test_probs = selector.predict(test_expert_preds)

    # --- Verification Steps ---
    assert final_test_probs.shape == (len(test_data["ids"]), len(classes))
    # Check probability constraints
    assert np.all(final_test_probs >= 0.0) and np.all(final_test_probs <= 1.0)

    print(">>> Saving Submission...")
    submission_path = os.path.join(work_dir, "submission.csv")
    save_submission(test_data["ids"], classes, final_test_probs, submission_path)

    # Final check
    if os.path.exists(submission_path):
        print(f"Success! Submission file created at: {submission_path}")

        # Validate format briefly
        df_sub = pd.read_csv(submission_path)
        assert df_sub.shape == (
            len(test_data["ids"]),
            len(classes) + 1,
        )  # +1 for id column
        assert "id" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    run_demo()
