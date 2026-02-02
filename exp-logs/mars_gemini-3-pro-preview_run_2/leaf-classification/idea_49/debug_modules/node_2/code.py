import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Import provided library modules
from library.utils import set_seed, clipped_log_loss
import library.data_manager as dm
import library.pipeline_factory as pf
from library.ensemble_selection import GreedyEnsemble

# Force reload of data_manager to ensure cache validation logic is active
importlib.reload(dm)


def run_demo():
    # 1. Setup
    print("Initializing demonstration...")
    set_seed(42)

    # 2. Data Loading & Feature Extraction
    # This step uses data_manager to merge provided tabular features with
    # extracted image morphometrics (handled by image_features.py).
    print("Loading and processing data...")

    # Load stratified train/val split
    # Note: First run will process images; subsequent runs use cache in ./working/idea_49/
    X_train, y_train, X_val, y_val = dm.get_train_val_split(load_cached_data=True)

    # Load test data
    X_test, ids_test = dm.get_test_data(load_cached_data=True)

    # Verify Data Shapes
    # Expected: 192 provided features + 11 extracted morphometric features = 203 columns
    n_features_expected = 203
    assert (
        X_train.shape[1] == n_features_expected
    ), f"Expected {n_features_expected} features, got {X_train.shape[1]}"
    assert X_val.shape[1] == n_features_expected
    assert X_test.shape[1] == n_features_expected
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    print(
        f"Data loaded successfully. Train shape: {X_train.shape}, Val shape: {X_val.shape}"
    )

    # 3. Pipeline Instantiation (Topologies)
    print("Instantiating model topologies...")

    # We use the factory to create 4 different variations of LDA pipelines
    # Topology A: Marginal Statistical Anchors (Global Features -> PowerTransform -> LDA)
    # Topology B: Rotational Statistical Experts (Global -> PT -> PCA -> PT -> LDA)
    # Topology C: Discriminative-Interaction Experts (Global -> PT -> LDA Proj -> Poly -> PT -> LDA)
    # Topology D: Polynomial Physical Experts (Morphometric Features -> PT -> Poly -> PT -> LDA)

    pipelines = {
        "Topo_A": pf.get_topology_a(solver="lsqr", shrinkage="auto"),
        "Topo_B": pf.get_topology_b(solver="lsqr", shrinkage="auto"),
        "Topo_C": pf.get_topology_c(solver="lsqr", shrinkage="auto"),
        "Topo_D": pf.get_topology_d(solver="lsqr", shrinkage="auto"),
    }

    # 4. Training and Validation (Selection Phase)
    print("Training models on split data...")

    val_preds = {}
    test_preds_individual = {}

    for name, pipeline in pipelines.items():
        # Fit on training split
        pipeline.fit(X_train, y_train)

        # Predict on validation split
        y_pred_val = pipeline.predict_proba(X_val)
        val_preds[name] = y_pred_val

        # Validate predictions
        assert y_pred_val.shape == (len(y_val), 99), f"{name} output shape mismatch"
        assert np.all(
            (y_pred_val >= 0) & (y_pred_val <= 1)
        ), f"{name} probs out of range"

        # Calculate individual score
        score = clipped_log_loss(y_val, y_pred_val)
        acc = accuracy_score(y_val, pipeline.predict(X_val))
        print(f"  {name} - Val LogLoss: {score:.5f}, Accuracy: {acc:.4f}")

    # 5. Ensemble Selection
    print("Optimizing ensemble weights...")

    # Initialize Greedy Forward Selection
    ensemble = GreedyEnsemble(max_size=20, tol=1e-4)

    # Fit ensemble on validation predictions
    ensemble.fit(val_preds, y_val)

    # Verify ensemble selected something
    assert len(ensemble.selected_experts) > 0, "Ensemble failed to select any experts"
    print(f"Selected Experts: {ensemble.weights}")
    print(f"Best Ensemble Val Score: {ensemble.best_score:.5f}")

    # 6. Final Retraining & Inference
    print("Retraining selected models on full dataset...")

    # Get combined train + val data
    X_full, y_full = dm.get_full_train_data(load_cached_data=True)

    # Identify unique models needed for the ensemble
    needed_models = list(ensemble.weights.keys())

    final_test_preds_dict = {}

    for name in needed_models:
        # Refit pipeline on full data
        # Note: In a real scenario, we might want to clone the pipeline first
        # but here we just refit the existing instance for speed/simplicity.
        pipeline = pipelines[name]
        pipeline.fit(X_full, y_full)

        # Predict on test set
        final_test_preds_dict[name] = pipeline.predict_proba(X_test)

    # 7. Generate Ensemble Predictions
    print("Generating final ensemble predictions...")

    # Use the ensemble to compute weighted average of test predictions
    y_test_pred_ensemble = ensemble.predict(final_test_preds_dict)

    # Validation of final output
    assert y_test_pred_ensemble.shape == (len(X_test), 99)
    # Row sums should be approx 1.0
    row_sums = y_test_pred_ensemble.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Final probabilities do not sum to 1"

    # 8. Create Submission File
    print("Creating submission file...")

    # Get class names from one of the models (all share same classes)
    classes = pipelines[needed_models[0]].classes_

    # Create DataFrame
    submission = pd.DataFrame(y_test_pred_ensemble, columns=classes)
    submission.insert(0, "id", ids_test)

    # Save to file
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")

    # Verify file content
    df_check = pd.read_csv(submission_path)
    assert df_check.shape == (99, 100)  # 99 rows, 99 classes + 1 id
    assert not df_check.isnull().values.any(), "Submission contains NaNs"

    print("Demonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
