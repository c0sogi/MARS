import os
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import RANDOM_SEED, SUBMISSION_DIR
from library.data_manager import LeafData
from library.model_engine import ProbabilisticExpert, GreedySelector, WeightedEnsemble
from library.utils import save_submission, clipped_log_loss

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)

# Suppress specific warnings for cleaner output (optional)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def combine_data_dicts(d1, d2):
    """Helper to concatenate two data dictionaries (e.g., train + val)."""
    combined_X = {}
    # Keys expected in the data dictionary
    feature_keys = ["global", "margin", "shape", "texture", "morphometrics"]

    for key in feature_keys:
        if key in d1 and key in d2:
            combined_X[key] = np.concatenate([d1[key], d2[key]], axis=0)

    combined_y = np.concatenate([d1["y"], d2["y"]], axis=0)
    return combined_X, combined_y


def run_demo():
    print("=== Starting Leaf Classification Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("[1] Loading Datasets...")
    data_manager = LeafData()
    # load_cached_data=True will try to use cached morphometrics if available
    datasets = data_manager.load_datasets(load_cached_data=True)

    train_data = datasets["train"]
    val_data = datasets["val"]
    test_data = datasets["test"]
    classes = datasets["classes"]

    print(f"    Train samples: {len(train_data['y'])}")
    print(f"    Val samples:   {len(val_data['y'])}")
    print(f"    Test samples:  {len(test_data['ids'])}")
    print(f"    Number of classes: {len(classes)}")

    # Basic Assertion to ensure data loaded correctly
    assert (
        train_data["global"].shape[1] == 192
    ), "Global features should have 192 columns"
    assert (
        len(train_data["y"]) == train_data["global"].shape[0]
    ), "Mismatch in X and y length"

    # -------------------------------------------------------------------------
    # 2. Single Expert Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Training Single Expert (Global Marginal Linear)...")

    # Define a single expert configuration
    single_expert_config = {
        "name": "Demo_Global_Linear",
        "group": "A",
        "feature_source": "global",
        "pipeline_type": "marginal_linear",
        "shrinkage": "auto",
    }

    # Instantiate and Fit
    expert = ProbabilisticExpert(**single_expert_config)
    expert.fit(train_data, train_data["y"])

    # Predict on Validation
    val_probs = expert.predict_proba(val_data)

    # Evaluate
    loss = clipped_log_loss(val_data["y"], val_probs)
    print(f"    Single Expert Validation Log Loss: {loss:.5f}")

    # Assertions for probability properties
    assert val_probs.shape == (
        len(val_data["y"]),
        len(classes),
    ), "Probability shape mismatch"
    assert np.all(val_probs >= 0) and np.all(
        val_probs <= 1
    ), "Probabilities out of range"
    # Note: Rows might not sum exactly to 1.0 before normalization in loss function,
    # but LDA usually outputs normalized probs.

    # -------------------------------------------------------------------------
    # 3. Ensemble Selection (Greedy Forward Selection)
    # -------------------------------------------------------------------------
    print("\n[3] Running Greedy Forward Selection...")

    # Define a small search space for the demo to ensure speed
    # We mix a global linear model with a texture-based polynomial model
    demo_library_config = [
        {
            "group": "A",
            "name": "Global_Linear",
            "feature_source": "global",
            "pipeline_type": "marginal_linear",
            "shrinkage_grid": [0.1, "auto"],  # Try two shrinkage options
        },
        {
            "group": "C",
            "name": "Texture_Poly",
            "feature_source": "texture",
            "pipeline_type": "component_poly",
            "shrinkage_grid": ["auto"],
        },
    ]

    selector = GreedySelector(demo_library_config, max_steps=5, tolerance=1e-4)

    # Fit selector using Train and Val sets
    selected_experts = selector.fit(
        train_data, train_data["y"], val_data, val_data["y"]
    )

    assert len(selected_experts) > 0, "Greedy selection failed to select any experts"
    print(f"    Selected {len(selected_experts)} experts for the final ensemble.")

    # -------------------------------------------------------------------------
    # 4. Final Retraining & Prediction
    # -------------------------------------------------------------------------
    print("\n[4] Retraining Ensemble on Full Data (Train + Val)...")

    # Combine Train and Val for maximum data usage
    X_full, y_full = combine_data_dicts(train_data, val_data)

    # Initialize Ensemble with selected configuration
    final_ensemble = WeightedEnsemble(selected_experts)

    # Fit on combined data
    final_ensemble.fit(X_full, y_full)

    # Predict on Test data
    print("    Generating predictions for Test set...")
    test_probs = final_ensemble.predict_proba(test_data)

    assert test_probs.shape == (
        len(test_data["ids"]),
        len(classes),
    ), "Test prediction shape mismatch"

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Saving Submission...")

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    save_submission(test_data["ids"], test_probs, classes, submission_path)

    # Verify file creation
    if os.path.exists(submission_path):
        print(f"    SUCCESS: Submission file created at {submission_path}")

        # Quick check of file content
        df_sub = pd.read_csv(submission_path)
        print(f"    Submission shape: {df_sub.shape}")
        assert df_sub.shape == (
            99,
            100,
        ), "Submission should have 99 rows and 100 columns (id + 99 classes)"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
