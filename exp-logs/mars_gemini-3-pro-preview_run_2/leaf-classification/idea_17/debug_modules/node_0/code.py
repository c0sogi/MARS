import os
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

# Import provided library modules
from library.utils import set_seed, clip_and_score
from library.data_loader import load_dataset
from library.model_factory import get_linear_lda, get_kernel_lda, get_discriminative_lr
from library.ensemble_selection import GreedySelector
from library.training_engine import (
    run_selection_phase,
    run_retraining_phase,
    generate_submission_predictions,
)

# Constants
RANDOM_STATE = 42
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")


def run_demo():
    # 1. Setup
    print("=== Setting up environment ===")
    set_seed(RANDOM_STATE)
    # Suppress warnings for cleaner output (e.g. convergence warnings)
    warnings.filterwarnings("ignore")

    # 2. Verify Utils
    print("\n=== Verifying Library: utils.py ===")
    # Test clip_and_score with known values
    y_true_dummy = np.array([0, 1, 0])
    # Create predictions that are "perfect" but unnormalized
    y_pred_dummy = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    # The function expects one-hot encoded y_true or label encoded?
    # Looking at utils.py, it uses sklearn.metrics.log_loss which handles label encoded y_true
    # if y_pred is probabilities.
    # However, let's look at utils.py again. It passes y_true directly to log_loss.
    # Sklearn log_loss supports y_true as array of shape (n_samples,) containing labels.

    score = clip_and_score(y_true_dummy, y_pred_dummy)
    print(f"Calculated Log Loss on dummy data: {score:.4f}")
    assert score < 0.3, "Log loss should be low for good predictions"
    assert score >= 0, "Log loss must be non-negative"

    # 3. Verify Data Loader
    print("\n=== Verifying Library: data_loader.py ===")
    # Load a small subset to verify functionality
    sample_size = 200
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=False,  # Force reload from metadata to test logic
        sample_size=sample_size,
        random_state=RANDOM_STATE,
    )

    print(f"Subset Train Shape: {X_train.shape}")
    print(f"Subset Val Shape: {X_val.shape}")
    print(f"Classes: {len(classes)}")

    assert (
        X_train.shape[0] == sample_size
    ), f"Expected {sample_size} training samples, got {X_train.shape[0]}"
    assert (
        X_train.shape[1] == 192
    ), f"Expected 192 features (64*3), got {X_train.shape[1]}"
    assert (
        len(test_ids) == X_test.shape[0]
    ), "Mismatch between test IDs and test features"

    # 4. Verify Model Factory
    print("\n=== Verifying Library: model_factory.py ===")
    # Instantiate experts
    expert_a = get_linear_lda(RANDOM_STATE)
    expert_b = get_kernel_lda(RANDOM_STATE)
    expert_c = get_discriminative_lr(RANDOM_STATE)

    assert isinstance(expert_a, Pipeline), "Expert A must be a Pipeline"
    assert isinstance(expert_b, Pipeline), "Expert B must be a Pipeline"
    assert isinstance(expert_c, Pipeline), "Expert C must be a Pipeline"
    print("All expert pipelines instantiated successfully.")

    # 5. Verify Ensemble Selection
    print("\n=== Verifying Library: ensemble_selection.py ===")
    # Create synthetic predictions for 3 models, 10 samples, 2 classes
    n_samples_sel = 10
    n_classes_sel = 2

    preds_1 = np.random.rand(n_samples_sel, n_classes_sel)
    preds_1 /= preds_1.sum(axis=1, keepdims=True)

    preds_2 = np.random.rand(n_samples_sel, n_classes_sel)
    preds_2 /= preds_2.sum(axis=1, keepdims=True)

    preds_dict = {"Model1": preds_1, "Model2": preds_2}
    y_true_sel = np.random.randint(0, n_classes_sel, n_samples_sel)

    # Run selector with few iterations for speed
    selector = GreedySelector(iterations=5, random_state=RANDOM_STATE)
    selector.fit(preds_dict, y_true_sel)

    weights = selector.get_weights()
    print(f"Selected Weights: {weights}")

    total_weight = sum(weights.values())
    assert np.isclose(total_weight, 1.0), f"Weights must sum to 1.0, got {total_weight}"

    # Test predict
    final_preds = selector.predict(preds_dict)
    assert final_preds.shape == (
        n_samples_sel,
        n_classes_sel,
    ), "Ensemble prediction shape mismatch"

    # 6. Verify Training Engine (End-to-End Integration)
    print("\n=== Verifying Library: training_engine.py (Full Pipeline) ===")
    # We use sample_size=None to use the full dataset (N~900) to ensure
    # LogisticRegressionCV has enough samples per class for 5-fold CV.
    # N=900 is small enough to run quickly.

    # Phase 1: Selection
    print("Running Phase 1: Selection...")
    weights, best_score, data_tuple = run_selection_phase(
        load_cached_data=True,  # Use cache if available for speed
        sample_size=None,
        random_state=RANDOM_STATE,
    )

    (
        X_train_full,
        y_train_full,
        X_val_full,
        y_val_full,
        X_test_full,
        test_ids_full,
        classes_full,
    ) = data_tuple

    print(f"Phase 1 Complete. Best Validation Score: {best_score:.4f}")
    assert len(weights) > 0, "No weights returned from selection phase"

    # Phase 2: Retraining
    print("Running Phase 2: Retraining...")
    final_models = run_retraining_phase(
        weights,
        X_train_full,
        y_train_full,
        X_val_full,
        y_val_full,
        random_state=RANDOM_STATE,
    )

    # Verify models are fitted
    for name, model in final_models.items():
        # Check if the last step of the pipeline is fitted
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
            check_is_fitted(estimator)
    print("Retraining Complete. Models fitted.")

    # Phase 3: Submission
    print("Running Phase 3: Submission Generation...")
    generate_submission_predictions(
        final_models, weights, X_test_full, test_ids_full, classes_full
    )

    # 7. Final Output Check
    print("\n=== Final Validation ===")
    if os.path.exists(SUBMISSION_FILE):
        df_sub = pd.read_csv(SUBMISSION_FILE)
        print(f"Submission file found at {SUBMISSION_FILE}")
        print(f"Submission Shape: {df_sub.shape}")

        # Expected shape: 99 test samples, 1 id col + 99 species cols = 100 columns
        assert df_sub.shape == (
            99,
            100,
        ), f"Expected shape (99, 100), got {df_sub.shape}"
        assert "id" in df_sub.columns, "Submission missing 'id' column"

        # Check probability constraints
        prob_cols = [c for c in df_sub.columns if c != "id"]
        probs = df_sub[prob_cols].values

        # Check range [0, 1]
        assert np.all(probs >= 0) and np.all(
            probs <= 1
        ), "Probabilities out of range [0, 1]"

        # Check sum to 1 (approximate due to float precision)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

        print("Submission file passed validation checks.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
