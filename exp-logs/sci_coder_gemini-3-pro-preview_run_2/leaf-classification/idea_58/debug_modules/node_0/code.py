import os
import numpy as np
import pandas as pd
import sys

# Import provided library modules
from library import config
from library import utils
from library import data_processing
from library import modeling
from library import ensemble
from library import workflow


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup
    utils.set_seed(42)

    # 2. Test Utilities
    print("\n[1/5] Testing Utilities...")
    y_true_sample = np.array([0, 1, 0])
    # Create predictions that are "okay" but not perfect
    y_pred_sample = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    loss = utils.clipped_log_loss(y_true_sample, y_pred_sample)
    print(f"  Calculated Log Loss: {loss:.4f}")
    assert (
        loss < 0.5
    ), "Log loss calculation seems incorrect (too high for decent preds)"

    # 3. Test Data Processing
    print("\n[2/5] Testing Data Processing...")
    dm = data_processing.DatasetManager()

    # Load training data. This handles metadata loading + image feature extraction.
    # Note: This might take a few seconds on the first run as it processes images.
    print("  Loading training data (and extracting physical features)...")
    df_train = dm.get_data("train")

    # Validations
    assert not df_train.empty, "Training dataframe is empty"
    assert "species" in df_train.columns, "Target column 'species' missing"
    assert (
        "phys_aspect_ratio" in df_train.columns
    ), "Physical features were not merged correctly"
    print(f"  Train Data Shape: {df_train.shape}")

    # Test Scope Slicing
    X_margin = dm.get_scope_slice(df_train, config.SCOPE_MARGIN)
    X_phys = dm.get_scope_slice(df_train, config.SCOPE_PHYSICAL)

    assert (
        X_margin.shape[1] == 64
    ), f"Expected 64 Margin features, got {X_margin.shape[1]}"
    assert (
        X_phys.shape[1] == 11
    ), f"Expected 11 Physical features, got {X_phys.shape[1]}"
    print("  Scope slicing verified.")

    # 4. Test Modeling
    print("\n[3/5] Testing Modeling...")
    # Define a simple expert configuration
    expert_def = {
        "scope": config.SCOPE_MARGIN,
        "topology": config.TOPOLOGY_MARGINAL,
        "shrinkage": 0.01,
    }

    # Prepare data
    y_train = dm.get_targets(df_train)
    # Use a small subset for speed in this demo step
    subset_idx = range(100)
    X_sub = X_margin[subset_idx]
    y_sub = y_train[subset_idx]

    print(f"  Training ExpertPipeline ({expert_def['topology']}) on subset...")
    model = modeling.ExpertPipeline(
        topology=expert_def["topology"], shrinkage=expert_def["shrinkage"]
    )
    model.fit(X_sub, y_sub)

    # Predict
    preds = model.predict_proba(X_sub[:5])
    print(f"  Prediction shape: {preds.shape}")

    # Verify Probability Constraints
    row_sums = preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Probabilities out of [0, 1] range"
    print("  Modeling logic verified.")

    # 5. Test Ensemble
    print("\n[4/5] Testing Ensemble Logic...")
    # Synthetic data: 10 samples, 3 classes
    n_samples_ens = 10
    n_classes_ens = 3
    y_ens = np.random.randint(0, n_classes_ens, n_samples_ens)

    # Expert A: Random noise
    preds_a = np.random.rand(n_samples_ens, n_classes_ens)
    preds_a /= preds_a.sum(axis=1, keepdims=True)

    # Expert B: Perfect predictions
    preds_b = np.zeros((n_samples_ens, n_classes_ens))
    preds_b[np.arange(n_samples_ens), y_ens] = 1.0

    preds_dict = {"expert_random": preds_a, "expert_perfect": preds_b}

    print("  Running GreedyForwardSelector on synthetic data...")
    selector = ensemble.GreedyForwardSelector(max_iterations=5, verbose=False)
    weights = selector.fit(preds_dict, y_ens)

    print(f"  Selected Weights: {weights}")
    # The selector should prioritize the perfect expert
    assert "expert_perfect" in weights, "Selector failed to pick the perfect model"
    assert weights.get("expert_perfect", 0) >= weights.get(
        "expert_random", 0
    ), "Perfect model should have higher or equal weight to random model"

    # Test Ensemble Prediction
    ens_preds = selector.predict(preds_dict)
    assert ens_preds.shape == (n_samples_ens, n_classes_ens)
    print("  Ensemble logic verified.")

    # 6. Test Full Workflow
    print("\n[5/5] Testing Full Workflow (ExperimentManager)...")

    # OPTIMIZATION: Reduce the number of experts in the config to ensure the demo finishes quickly.
    # We will pick just 2 diverse experts.
    original_experts = config.EXPERT_DEFINITIONS
    config.EXPERT_DEFINITIONS = [
        # Expert 1: Margin features with Marginal Topology
        {
            "scope": config.SCOPE_MARGIN,
            "topology": config.TOPOLOGY_MARGINAL,
            "shrinkage": 0.01,
        },
        # Expert 2: Shape features with Robust Topology
        {
            "scope": config.SCOPE_SHAPE,
            "topology": config.TOPOLOGY_ROBUST,
            "shrinkage": 0.01,
        },
    ]
    print(
        f"  Temporarily reduced expert count to {len(config.EXPERT_DEFINITIONS)} for speed."
    )

    manager = workflow.ExperimentManager()

    # Phase 1: Selection
    # We set load_cached_preds=False to force the training loop to run for demonstration
    print("  Running Phase 1: Selection...")
    weights = manager.run_selection_phase(load_cached_preds=False)
    print(f"  Phase 1 Weights: {weights}")
    assert len(weights) > 0, "No experts were selected in Phase 1"

    # Phase 2: Final Submission
    print("  Running Phase 2: Final Retraining & Submission...")
    manager.run_final_phase(weights)

    # Verify Output
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission generated successfully.")
    print(f"  Shape: {df_sub.shape}")
    print(f"  Columns: {list(df_sub.columns)[:5]} ...")

    # Restore config (good practice)
    config.EXPERT_DEFINITIONS = original_experts

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
