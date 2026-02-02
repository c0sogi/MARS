import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Add library path to system path to enable imports
sys.path.append("./library")

from library.utils import set_seed, save_submission, calculate_metric
from library.data_loader import load_data
from library.models import get_expert_pipeline, get_fixed_pipeline
from library.ensemble import GreedySelector


def run_leaf_classification_demo():
    # 1. Setup Environment
    # --------------------
    print("1. Setting up environment...")
    set_seed(42)

    # 2. Data Loading
    # --------------------
    print("\n2. Loading and Preprocessing Data...")
    # We use debug=False to load the full dataset (approx 712 train samples).
    # This is small enough to train quickly but ensures all 99 classes are represented.
    # load_cached_data=False ensures we demonstrate the full preprocessing pipeline.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        load_cached_data=False, debug=False
    )

    print(f"   Train Set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"   Val Set:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"   Test Set:  {X_test.shape}")
    print(f"   Number of Classes: {len(classes)}")

    # Assertions to verify data integrity
    assert X_train.shape[1] == 192, "Expected 192 features (Margin + Shape + Texture)"
    assert len(classes) == 99, "Expected 99 plant species"
    assert not np.isnan(X_train).any(), "Training data contains NaNs"

    # 3. Model Training (Phase 1)
    # ---------------------------
    print("\n3. Training Expert Models (Phase 1)...")

    expert_names = ["Expert_A", "Expert_B", "Expert_C"]
    trained_pipelines = {}
    val_predictions = {}
    test_predictions = {}

    # Metric calculation requires integer labels for y_true and list of label indices for classes
    metric_classes = list(range(len(classes)))

    for name in expert_names:
        print(f"   Training {name}...")

        # Instantiate untrained pipeline
        pipeline = get_expert_pipeline(name, random_state=42)

        # Fit on training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Generate predictions
        val_probs = pipeline.predict_proba(X_val)
        test_probs = pipeline.predict_proba(X_test)

        val_predictions[name] = val_probs
        test_predictions[name] = test_probs

        # Evaluate
        score = calculate_metric(y_val, val_probs, classes=metric_classes)
        print(f"     -> {name} Validation Log Loss: {score:.5f}")

        # Verify prediction shape
        assert val_probs.shape == (
            len(y_val),
            len(classes),
        ), f"{name} val prediction shape mismatch"

    # 4. Pipeline Fixing (Phase 2)
    # ----------------------------
    print("\n4. Converting to Fixed Pipelines (Phase 2)...")
    # Expert_B uses LogisticRegressionCV. We demonstrate extracting the best hyperparameters
    # to create a fixed, lighter pipeline for deployment.

    expert_b_cv = trained_pipelines["Expert_B"]
    expert_b_fixed = get_fixed_pipeline(expert_b_cv)

    # Verify the structure change
    steps_cv = dict(expert_b_cv.steps)
    steps_fixed = dict(expert_b_fixed.steps)

    assert isinstance(
        steps_cv["lr"], LogisticRegressionCV
    ), "Original Expert_B should use CV"
    assert isinstance(
        steps_fixed["lr"], LogisticRegression
    ), "Fixed Expert_B should use standard LR"
    assert not isinstance(
        steps_fixed["lr"], LogisticRegressionCV
    ), "Fixed Expert_B should not use CV"

    print("   Successfully converted Expert_B (CV) to Expert_B (Fixed).")

    # 5. Ensemble Selection
    # ---------------------
    print("\n5. Selecting Optimal Ensemble...")
    # Initialize Greedy Selector with a tolerance for improvement
    selector = GreedySelector(tolerance=1e-5)

    # Fit selector on validation predictions
    selector.fit(val_predictions, y_val, classes=metric_classes)

    selected = selector.selected_experts
    print(f"   Selected Experts: {selected}")
    assert len(selected) > 0, "Ensemble selection failed to pick any model"

    # 6. Final Prediction & Submission
    # --------------------------------
    print("\n6. Generating Final Submission...")

    # Compute ensemble predictions on test set
    final_test_probs = selector.predict(test_predictions)

    # Define output path
    output_file = "./working/demo_submission.csv"

    # Save submission
    save_submission(test_ids, classes, final_test_probs, output_path=output_file)

    # Verify Output
    if os.path.exists(output_file):
        df_sub = pd.read_csv(output_file)
        print(f"   Submission saved to {output_file}")
        print(f"   Submission shape: {df_sub.shape}")

        # Validate format
        assert df_sub.shape == (
            len(test_ids),
            len(classes) + 1,
        ), "Submission shape mismatch"
        assert df_sub.columns[0] == "id", "First column must be 'id'"
        assert list(df_sub.columns[1:]) == list(
            classes
        ), "Column headers must match class names"

        # Validate probability constraints
        probs_only = df_sub.iloc[:, 1:].values
        assert (probs_only >= 0).all() and (
            probs_only <= 1
        ).all(), "Probabilities out of range [0, 1]"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_leaf_classification_demo()
