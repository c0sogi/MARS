import os
import sys
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Force reload of library modules to ensure config changes are picked up in persistent environment
if "library.config" in sys.modules:
    importlib.reload(sys.modules["library.config"])
if "library.utils" in sys.modules:
    importlib.reload(sys.modules["library.utils"])
if "library.data_loader" in sys.modules:
    importlib.reload(sys.modules["library.data_loader"])
if "library.expert_library" in sys.modules:
    importlib.reload(sys.modules["library.expert_library"])
if "library.ensemble_selection" in sys.modules:
    importlib.reload(sys.modules["library.ensemble_selection"])

# Import from provided library files
from library.utils import set_seed, save_submission, log_loss_metric, clip_probabilities
from library.data_loader import load_data
from library.expert_library import get_expert_library
from library.ensemble_selection import (
    greedy_forward_selection,
    compute_ensemble_prediction,
)
from library.config import SUBMISSION_PATH


def main():
    print("Initializing Leaf Classification Pipeline...")

    # 1. Set Seed for Reproducibility
    set_seed(42)

    # 2. Load Data
    # This handles loading metadata, extracting image features, and merging them.
    print("Loading data...")
    data = load_data(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]
    feature_slices = data["feature_slices"]

    # Validation: Check Data Shapes
    n_classes = len(classes)
    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Number of classes: {n_classes}")

    assert (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ), "Feature count mismatch."
    assert len(y_train) == X_train.shape[0], "Train labels mismatch."

    # 3. Initialize Expert Library
    # Calculate priors for LDA (optional but recommended for imbalanced data, though this dataset is mostly balanced)
    priors = np.bincount(y_train) / len(y_train)

    print("Constructing Expert Library...")
    experts = get_expert_library(feature_slices, priors=priors)
    print(f"Initialized {len(experts)} experts.")

    # 4. Train Experts and Predict on Validation
    print("Training experts and generating validation predictions...")
    val_preds_dict = {}
    fitted_pipelines = {}

    for i, expert in enumerate(experts):
        name = expert["name"]
        pipeline = expert["pipeline"]

        # Fit on training data
        pipeline.fit(X_train, y_train)

        # Predict on validation data
        # LDA predict_proba returns probabilities
        val_probs = pipeline.predict_proba(X_val)

        # Store for ensemble selection
        val_preds_dict[name] = val_probs
        fitted_pipelines[name] = pipeline

        # Basic assertion for probability validity
        assert val_probs.shape == (len(y_val), n_classes), f"Shape mismatch for {name}"
        assert not np.isnan(val_probs).any(), f"NaN predictions in {name}"

    # 5. Ensemble Selection (Greedy Forward Selection)
    print("Running Greedy Forward Selection...")
    # We use a smaller max_iter for speed in this demo, though 100 is standard
    weights, best_val_score = greedy_forward_selection(
        val_preds_dict, y_val, max_iter=50, tol=1e-6, verbose=True
    )

    print(f"Best Validation Log Loss: {best_val_score:.5f}")
    assert weights, "No experts were selected for the ensemble."

    # 6. Generate Test Predictions
    print("Generating Test Predictions...")
    test_preds_dict = {}

    # Only predict with selected experts to save time
    for name in weights.keys():
        pipeline = fitted_pipelines[name]
        test_probs = pipeline.predict_proba(X_test)
        test_preds_dict[name] = test_probs

    # Compute weighted average
    final_test_preds = compute_ensemble_prediction(test_preds_dict, weights)

    # Apply final clipping to satisfy metric requirements strictly
    final_test_preds = clip_probabilities(final_test_preds)

    # Validation: Check Test Predictions
    assert final_test_preds.shape == (
        len(test_ids),
        n_classes,
    ), "Final prediction shape mismatch"
    assert np.all(
        (final_test_preds >= 0) & (final_test_preds <= 1)
    ), "Probabilities out of bounds"

    # 7. Save Submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    save_submission(test_ids, classes, final_test_preds, output_path=SUBMISSION_PATH)

    # Final File Check
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."
    df_sub = pd.read_csv(SUBMISSION_PATH)
    assert df_sub.shape == (
        len(test_ids),
        n_classes + 1,
    ), "Submission file dimensions incorrect."

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
