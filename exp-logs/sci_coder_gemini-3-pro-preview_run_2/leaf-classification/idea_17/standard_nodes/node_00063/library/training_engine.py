import os
import numpy as np
import pandas as pd
from sklearn.base import clone

from library.utils import set_seed, clip_and_score
from library.data_loader import load_dataset
from library.model_factory import get_linear_lda, get_discriminative_lr
from library.ensemble_selection import GreedySelector

# Constants
SUBMISSION_DIR = "./submission"


def run_selection_phase(load_cached_data=True, sample_size=None, random_state=42):
    """
    Executes Phase 1: Expert Training & Ensemble Selection.

    1. Loads data.
    2. Trains candidate experts on the training split.
    3. Evaluates experts on the validation split.
    4. Runs Greedy Forward Selection to find optimal ensemble weights.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        sample_size (int, optional): limit training size for debugging.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (weights, best_score, data_tuple)
               where data_tuple is (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    set_seed(random_state)

    # 1. Load Data
    print("Loading dataset for Selection Phase...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=load_cached_data,
        sample_size=sample_size,
        random_state=random_state,
    )

    print(
        f"Data Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 2. Define Expert Pool
    # We instantiate fresh models from the factory
    # Removed Kernel_LDA (Cite 00061)
    experts = {
        "Linear_LDA": get_linear_lda(random_state=random_state),
        "Linear_LR": get_discriminative_lr(random_state=random_state),
    }

    val_predictions = {}

    print("\n--- Training Experts and Generating Validation Predictions ---")
    for name, model in experts.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predict on Validation
        probs = model.predict_proba(X_val)
        val_predictions[name] = probs

        # Score
        loss = clip_and_score(y_val, probs)
        print(f"  {name} Validation LogLoss: {loss}")

    # 3. Ensemble Selection
    print("\n--- Running Greedy Forward Selection ---")
    selector = GreedySelector(iterations=100, random_state=random_state)
    selector.fit(val_predictions, y_val)

    weights = selector.get_weights()
    best_score = selector.best_score_

    print(f"Selection Complete. Best Ensemble LogLoss: {best_score}")
    print("Ensemble Weights:")
    for name, w in weights.items():
        if w > 0:
            print(f"  {name}: {w}")

    return (
        weights,
        best_score,
        (X_train, y_train, X_val, y_val, X_test, test_ids, classes),
    )


def run_retraining_phase(weights, X_train, y_train, X_val, y_val, random_state=42):
    """
    Executes Phase 2: Final Retraining.

    1. Combines Training and Validation data.
    2. Retrains only the selected experts (weight > 0) on the full dataset.

    Args:
        weights (dict): Ensemble weights from Phase 1.
        X_train, y_train, X_val, y_val: Data splits.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary of retrained model objects.
    """
    set_seed(random_state)
    print("\n--- Phase 2: Final Retraining on Full Dataset ---")

    # Combine data
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    print(f"Full Training Set Shape: {X_full.shape}")

    # Instantiate fresh experts to clone from
    # We need the factory definitions again to ensure we have clean models
    base_experts = {
        "Linear_LDA": get_linear_lda(random_state=random_state),
        "Linear_LR": get_discriminative_lr(random_state=random_state),
    }

    final_models = {}

    for name, w in weights.items():
        if w > 0:
            print(f"Retraining {name} (Weight: {w})...")
            # Clone ensures we have a fresh estimator with the same parameters
            model = clone(base_experts[name])
            model.fit(X_full, y_full)
            final_models[name] = model
        else:
            print(f"Skipping {name} (Weight: 0.0)")

    return final_models


def generate_submission_predictions(final_models, weights, X_test, test_ids, classes):
    """
    Executes Phase 3: Inference and Submission Generation.

    1. Generates predictions on Test set using retrained models.
    2. Computes weighted average.
    3. Normalizes probabilities.
    4. Saves to CSV.

    Args:
        final_models (dict): Dictionary of retrained models.
        weights (dict): Ensemble weights.
        X_test (np.array): Test features.
        test_ids (np.array): Test IDs.
        classes (np.array): Class names.
    """
    print("\n--- Phase 3: Generating Submission Predictions ---")

    n_samples = X_test.shape[0]
    n_classes = len(classes)

    # Initialize weighted sum
    weighted_probs = np.zeros((n_samples, n_classes))

    for name, model in final_models.items():
        w = weights.get(name, 0.0)
        if w > 0:
            print(f"Predicting with {name}...")
            probs = model.predict_proba(X_test)
            weighted_probs += w * probs

    # Normalize rows to sum to 1 (handling potential floating point drift)
    row_sums = weighted_probs.sum(axis=1)
    # Avoid division by zero (unlikely given softmax/prob outputs but good practice)
    row_sums[row_sums == 0] = 1.0
    final_probs = weighted_probs / row_sums[:, np.newaxis]

    # Create Submission DataFrame
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    df_sub = pd.DataFrame(final_probs, columns=classes)
    df_sub.insert(0, "id", test_ids)

    print(f"Saving submission to {submission_path}...")
    df_sub.to_csv(submission_path, index=False)
    print("Submission saved successfully.")
