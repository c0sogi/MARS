import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import from the provided library
from library.config import RANDOM_SEED, SUBMISSION_PATH, FLOAT_PRECISION
from library.utils import set_seed
from library.data_loader import load_dataset
from library.topologies import get_expert_library
from library.training_engine import train_and_predict_experts, retrain_final_ensemble
from library.ensemble_selector import greedy_forward_selection


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)
    print("Initialized with random seed:", RANDOM_SEED)

    # 2. Load Data
    # The load_dataset function handles caching of image feature extraction
    print("\n--- Loading Data ---")
    train_data = load_dataset("train", load_cached_data=True)
    val_data = load_dataset("val", load_cached_data=True)
    test_data = load_dataset("test", load_cached_data=True)

    # Unpack data
    X_train_global = train_data["global_view"]
    X_train_morph = train_data["morph_view"]
    y_train = train_data["y"]

    X_val_global = val_data["global_view"]
    X_val_morph = val_data["morph_view"]
    y_val = val_data["y"]

    X_test_global = test_data["global_view"]
    X_test_morph = test_data["morph_view"]
    test_ids = test_data["ids"]

    # Verify shapes
    print(f"Train Global Shape: {X_train_global.shape}")
    print(f"Train Morph Shape: {X_train_morph.shape}")
    print(f"Val Global Shape: {X_val_global.shape}")
    print(f"Test Global Shape: {X_test_global.shape}")

    assert X_train_global.shape[0] == len(y_train)
    assert X_train_morph.shape[0] == len(y_train)
    assert X_train_morph.shape[1] == 11  # Hu moments + Geometric scalars

    # 3. Get Expert Library
    print("\n--- Initializing Expert Library ---")
    expert_configs = get_expert_library()
    print(f"Loaded {len(expert_configs)} expert topologies.")

    # 4. Phase 1: Train Experts and Get Validation Predictions
    # This step trains all experts on the training set and freezes their hyperparameters
    print("\n--- Phase 1: Training & Validation ---")
    expert_results = train_and_predict_experts(
        X_train_global,
        X_train_morph,
        y_train,
        X_val_global,
        X_val_morph,
        expert_configs,
        load_cached_preds=True,
    )

    # 5. Ensemble Selection
    # Use Greedy Forward Selection to find the best combination of experts
    print("\n--- Ensemble Selection ---")
    selected_experts = greedy_forward_selection(
        expert_results,
        expert_configs,
        y_val,
        max_iter=20,  # Limited iterations for speed in this demo
        tol=1e-5,
    )

    if not selected_experts:
        raise RuntimeError("Ensemble selection failed to select any experts.")

    # 6. Prepare Full Dataset for Retraining
    # Concatenate Train and Val to maximize data for final model
    print("\n--- Preparing Full Dataset ---")
    X_full_global = np.vstack([X_train_global, X_val_global])
    X_full_morph = np.vstack([X_train_morph, X_val_morph])
    y_full = np.concatenate([y_train, y_val])

    print(f"Full Dataset Shape: {X_full_global.shape}")

    # 7. Phase 2: Retrain Selected Experts on Full Data
    print("\n--- Phase 2: Retraining & Testing ---")
    test_predictions_dict = retrain_final_ensemble(
        X_full_global,
        X_full_morph,
        y_full,
        selected_experts,
        X_test_global,
        X_test_morph,
    )

    # 8. Aggregate Predictions
    print("\n--- Aggregating Predictions ---")
    # Get class names from one of the trained pipelines to ensure correct column order
    # We can inspect the first selected expert's pipeline
    first_expert_pipeline = selected_experts[0]["frozen_pipeline"]
    # The pipeline is fitted, but the classes_ attribute belongs to the final estimator step
    # However, since we cloned and refitted in retrain_final_ensemble, we don't have the fitted object directly accessible
    # in the main scope easily without returning it.
    # BUT: The classes are simply the sorted unique values of y_full.
    classes = np.unique(y_full)
    n_classes = len(classes)
    n_test = len(test_ids)

    final_probs = np.zeros((n_test, n_classes), dtype=FLOAT_PRECISION)
    total_weight = 0.0

    for expert_info in selected_experts:
        eid = expert_info["id"]
        weight = expert_info["weight"]
        preds = test_predictions_dict[eid]

        final_probs += preds * weight
        total_weight += weight

    final_probs /= total_weight

    # 9. Generate Submission
    print("\n--- Generating Submission ---")

    # Create DataFrame
    submission_df = pd.DataFrame(final_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    # 10. Validation of Output
    print("\n--- Validating Submission ---")
    loaded_sub = pd.read_csv(SUBMISSION_PATH)

    # Check dimensions
    expected_cols = n_classes + 1
    assert loaded_sub.shape == (
        99,
        expected_cols,
    ), f"Shape mismatch. Expected (99, {expected_cols}), got {loaded_sub.shape}"

    # Check ID integrity
    assert np.all(loaded_sub["id"].values == test_ids), "ID mismatch in submission."

    # Check probability range
    prob_cols = loaded_sub.columns[1:]
    probs = loaded_sub[prob_cols].values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]."

    # Check row sums (should be approximately 1, though clipped log loss logic handles slight deviations)
    row_sums = probs.sum(axis=1)
    # Allow small floating point error
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
