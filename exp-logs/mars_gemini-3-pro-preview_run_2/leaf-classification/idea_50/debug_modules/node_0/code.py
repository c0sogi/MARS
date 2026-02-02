import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Import provided library functions
from library.utils import set_seed, clipped_log_loss
from library.data_loader import load_dataset
from library.expert_pipelines import build_pipeline
from library.ensemble_selection import GreedyEnsembleSelector


def run_demo():
    print("Initializing Demo...")
    set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading Dataset...")
    # load_dataset returns:
    # (X_train_global, X_train_morph, y_train),
    # (X_val_global, X_val_morph, y_val),
    # (X_test_global, X_test_morph, test_ids, classes)
    train_data, val_data, test_data = load_dataset(load_cached_data=True)

    X_train_global, X_train_morph, y_train = train_data
    X_val_global, X_val_morph, y_val = val_data
    X_test_global, X_test_morph, test_ids, classes = test_data

    print(f"Training Samples: {X_train_global.shape[0]}")
    print(f"Validation Samples: {X_val_global.shape[0]}")
    print(f"Test Samples: {X_test_global.shape[0]}")
    print(f"Number of Classes: {len(classes)}")

    # -------------------------------------------------------------------------
    # 2. Expert Pipeline Training & Prediction
    # -------------------------------------------------------------------------
    print("\n[Step 2] Training Expert Pipelines...")

    # Define configurations for the experts
    # Topologies A-D use global features (192 dims)
    # Topology E uses morphometric features (11 dims)
    expert_configs = [
        {"name": "Expert_A", "topology": "A", "features": "global", "shrinkage": 0.2},
        {"name": "Expert_B", "topology": "B", "features": "global", "shrinkage": 0.2},
        {"name": "Expert_C", "topology": "C", "features": "global", "shrinkage": 0.2},
        {"name": "Expert_D", "topology": "D", "features": "global", "shrinkage": 0.2},
        {
            "name": "Expert_E",
            "topology": "E",
            "features": "morph",
            "shrinkage": None,
        },  # E uses auto shrinkage
    ]

    val_predictions = {}
    test_predictions = {}

    for config in expert_configs:
        name = config["name"]
        topo = config["topology"]
        feat_type = config["features"]
        shrinkage = config["shrinkage"]

        print(f"  Processing {name} (Topology {topo})...")

        # Select appropriate feature set
        if feat_type == "global":
            X_train = X_train_global
            X_val = X_val_global
            X_test = X_test_global
        else:
            X_train = X_train_morph
            X_val = X_val_morph
            X_test = X_test_morph

        # Build Pipeline
        # Note: n_components_lda is optional for D, defaults to 25
        pipeline = build_pipeline(topology=topo, shrinkage=shrinkage)

        # Fit Pipeline
        pipeline.fit(X_train, y_train)

        # Predict on Validation
        y_val_pred = pipeline.predict_proba(X_val)
        val_predictions[name] = y_val_pred

        # Predict on Test
        y_test_pred = pipeline.predict_proba(X_test)
        test_predictions[name] = y_test_pred

        # Validation Checks
        assert y_val_pred.shape == (
            len(y_val),
            len(classes),
        ), f"Shape mismatch for {name}: {y_val_pred.shape}"
        assert np.all(
            (y_val_pred >= 0) & (y_val_pred <= 1)
        ), f"Probabilities out of bounds for {name}"

        # Calculate Score
        score = clipped_log_loss(y_val, y_val_pred)
        acc = accuracy_score(y_val, np.argmax(y_val_pred, axis=1))
        print(f"    -> Log Loss: {score:.4f}, Accuracy: {acc:.4f}")

    # -------------------------------------------------------------------------
    # 3. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Metric Logic...")
    # Test clipped_log_loss with extreme values to ensure stability
    y_true_dummy = np.array([0, 1])
    # Prediction with absolute 0 and 1, which would cause log(0) = -inf without clipping
    y_pred_dummy = np.array([[0.9999999999999999, 0.0], [0.0, 1.0]])

    # The function expects y_true as labels or one-hot. Here we use labels.
    # We need to ensure y_pred_dummy matches the shape expected (2 samples, 2 classes)
    loss = clipped_log_loss(y_true_dummy, y_pred_dummy)

    assert np.isfinite(
        loss
    ), "Metric returned non-finite value for extreme probabilities."
    print(f"  Metric check passed. Loss for extreme values: {loss:.6f}")

    # -------------------------------------------------------------------------
    # 4. Ensemble Selection
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Greedy Ensemble Selection...")

    selector = GreedyEnsembleSelector(max_iterations=20, tolerance=1e-5)

    # Fit the selector
    weights, best_score = selector.fit(val_predictions, y_val)

    # Validate results
    assert len(weights) > 0, "Ensemble selector failed to select any experts."
    assert best_score > 0, "Best score should be positive."

    # Check that ensemble score is at least as good as the best single model (approx)
    # Note: Greedy selection starts with the best single model, so it shouldn't be worse.
    best_single_score = min(
        [clipped_log_loss(y_val, p) for p in val_predictions.values()]
    )
    print(f"  Best Single Model Score: {best_single_score:.5f}")
    print(f"  Ensemble Score:          {best_score:.5f}")

    assert (
        best_score <= best_single_score + 1e-9
    ), "Ensemble score is worse than the best single model."

    # -------------------------------------------------------------------------
    # 5. Final Prediction Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Final Test Predictions...")

    final_test_probs = selector.predict(test_predictions)

    assert final_test_probs.shape == (
        len(test_ids),
        len(classes),
    ), "Final test prediction shape is incorrect."

    # Create submission dataframe structure
    submission_df = pd.DataFrame(final_test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    print("  Submission DataFrame created successfully.")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Head:\n{submission_df.head(2)}")

    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
