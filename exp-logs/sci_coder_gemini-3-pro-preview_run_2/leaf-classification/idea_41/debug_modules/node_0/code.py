import os
import numpy as np
import pandas as pd
import warnings
from library import data_manager, model_factory, ensemble_selector


# 1. Setup and Configuration
# ==============================================================================
def setup_environment():
    # Set fixed seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    print("Environment setup complete.")


def run_demo():
    setup_environment()

    # 2. Data Loading and Merging
    # ==============================================================================
    print("\n[Step 1] Loading and merging data...")
    # We set load_cached_data=False to demonstrate the full extraction pipeline
    # This uses library.image_utils internally to extract morphometrics
    X_train, y_train, X_val, y_val, X_test, test_ids = data_manager.load_and_merge_data(
        load_cached_data=False
    )

    # Validation of loaded data
    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test shape:  {X_test.shape}, IDs: {test_ids.shape}")

    assert X_train.shape[0] == y_train.shape[0], "Train features and labels mismatch"
    assert X_val.shape[0] == y_val.shape[0], "Val features and labels mismatch"
    assert X_test.shape[0] == len(test_ids), "Test features and IDs mismatch"
    assert not np.isnan(X_train).any(), "NaNs found in training data"

    # 3. Feature Transformation (Topology Application)
    # ==============================================================================
    print("\n[Step 2] Applying topological transformations...")

    # Topology A: Marginal Gaussian Anchors
    X_train_marg, X_val_marg, X_test_marg = data_manager.apply_topology(
        X_train, X_val, X_test, topology_type="marginal"
    )

    # Topology B: Iterative Gaussian Experts
    X_train_iter, X_val_iter, X_test_iter = data_manager.apply_topology(
        X_train, X_val, X_test, topology_type="iterative"
    )

    # Validate transformations
    assert (
        X_train_marg.shape == X_train.shape
    ), "Marginal topology changed shape unexpectedly"
    assert (
        X_train_iter.shape == X_train.shape
    ), "Iterative topology changed shape unexpectedly"

    # 4. Model Training (Expert Generation)
    # ==============================================================================
    print("\n[Step 3] Training experts...")

    shrinkage_candidates = model_factory.get_shrinkage_candidates()
    print(f"Shrinkage candidates: {shrinkage_candidates}")

    # Dictionaries to store predictions from each expert
    # Keys will be unique identifiers for the expert
    val_preds_dict = {}
    test_preds_dict = {}

    # We need to capture the classes from the model to ensure column alignment later
    classes_ = None

    # Define the topologies to iterate over
    topologies = {
        "marginal": (X_train_marg, X_val_marg, X_test_marg),
        "iterative": (X_train_iter, X_val_iter, X_test_iter),
    }

    for topo_name, (X_tr, X_v, X_te) in topologies.items():
        for shrinkage in shrinkage_candidates:
            expert_name = f"lda_{topo_name}_{shrinkage}"

            # Instantiate model
            model = model_factory.get_expert_model(shrinkage=shrinkage)

            # Fit model
            model.fit(X_tr, y_train)

            # Store classes for submission formatting
            if classes_ is None:
                classes_ = model.classes_
            else:
                # verify consistency across models
                assert np.array_equal(
                    classes_, model.classes_
                ), "Inconsistent classes between models"

            # Predict
            p_val = model.predict_proba(X_v)
            p_test = model.predict_proba(X_te)

            # Store predictions
            val_preds_dict[expert_name] = p_val
            test_preds_dict[expert_name] = p_test

    print(f"Trained {len(val_preds_dict)} experts.")

    # 5. Ensemble Selection
    # ==============================================================================
    print("\n[Step 4] Running Greedy Forward Selection...")

    # Instantiate selector
    # Using a small max_iter for demonstration speed
    selector = ensemble_selector.GreedySelector(max_iter=10, tol=1e-5)

    # Fit selector on validation data
    selector.fit(val_preds_dict, y_val)

    # Validate selection
    if not selector.selected_experts:
        raise RuntimeError("Selector failed to select any experts.")

    print(f"Selected experts: {selector.selected_experts}")

    # 6. Prediction and Submission
    # ==============================================================================
    print("\n[Step 5] Generating final predictions and submission...")

    # Predict on test set using the learned ensemble weights
    final_test_probs = selector.predict(test_preds_dict)

    # Validate output shape
    n_test_samples = len(test_ids)
    n_classes = len(classes_)
    assert final_test_probs.shape == (
        n_test_samples,
        n_classes,
    ), f"Output shape mismatch. Expected {(n_test_samples, n_classes)}, got {final_test_probs.shape}"

    # Create Submission DataFrame
    # Columns must be: id, <class_names...>
    submission_df = pd.DataFrame(final_test_probs, columns=classes_)
    submission_df.insert(0, "id", test_ids)

    # Save to working directory
    submission_path = "./working/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")

    # Final check of the file
    saved_df = pd.read_csv(submission_path)
    print(f"Saved submission shape: {saved_df.shape}")
    assert saved_df.shape == (
        99,
        100,
    ), "Submission file does not have expected shape (99 rows, 100 cols)"


if __name__ == "__main__":
    run_demo()
