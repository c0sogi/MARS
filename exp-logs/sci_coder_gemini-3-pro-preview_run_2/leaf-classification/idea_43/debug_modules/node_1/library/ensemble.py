import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from collections import Counter

from library.config import (
    WORKING_DIR,
    SUBMISSION_FILE,
    LDA_SHRINKAGE_PARAMS,
    MAX_ENSEMBLE_ITERATIONS,
    SELECTION_PATIENCE,
    FLOAT_PRECISION,
    RANDOM_SEED,
)
from library.data import get_datasets
from library.transforms import apply_topology
from library.models import get_lda_expert, postprocess_probabilities

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def get_expert_key(topology, view, shrinkage):
    """Generates a unique string key for an expert configuration."""
    return f"{topology}|{view}|{shrinkage}"


def parse_expert_key(key):
    """Parses an expert key back into configuration components."""
    parts = key.split("|")
    topology = parts[0]
    view = parts[1]
    shrinkage = parts[2]

    # Convert shrinkage back to correct type
    if shrinkage == "auto":
        pass
    else:
        try:
            shrinkage = float(shrinkage)
        except ValueError:
            pass  # Keep as string if it fails, though unlikely given config

    return topology, view, shrinkage


# ==============================================================================
# GREEDY SELECTOR CLASS
# ==============================================================================


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement to optimize the ensemble.
    """

    def __init__(
        self, max_iterations=MAX_ENSEMBLE_ITERATIONS, patience=SELECTION_PATIENCE
    ):
        self.max_iterations = max_iterations
        self.patience = patience
        self.selected_experts = []
        self.best_score = float("inf")
        self.history = []

    def fit(self, predictions_dict, y_true):
        """
        Selects experts to minimize log loss.

        Args:
            predictions_dict (dict): Key -> np.ndarray (n_samples, n_classes)
            y_true (np.ndarray): True labels (n_samples,)

        Returns:
            list: List of selected expert keys (with duplicates implying weight).
        """
        available_keys = list(predictions_dict.keys())
        current_ensemble_sum = np.zeros_like(list(predictions_dict.values())[0])
        ensemble_size = 0

        # Initial best score (using uniform probas or similar is worse, so we start inf)
        self.best_score = float("inf")
        patience_counter = 0

        print(
            f"Starting Greedy Selection on {len(available_keys)} candidate experts..."
        )

        for i in range(self.max_iterations):
            best_iter_score = float("inf")
            best_iter_key = None

            # Try adding each expert to the current ensemble
            for key in available_keys:
                candidate_pred = predictions_dict[key]

                # Calculate temporary ensemble prediction
                # New Average = (Current Sum + Candidate) / (N + 1)
                temp_ensemble_pred = (current_ensemble_sum + candidate_pred) / (
                    ensemble_size + 1
                )

                # Post-process for metric stability
                temp_ensemble_pred = postprocess_probabilities(temp_ensemble_pred)

                score = log_loss(y_true, temp_ensemble_pred)

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_key = key

            # Check for improvement
            if best_iter_score < self.best_score:
                self.best_score = best_iter_score
                self.selected_experts.append(best_iter_key)
                current_ensemble_sum += predictions_dict[best_iter_key]
                ensemble_size += 1
                patience_counter = 0

                print(
                    f"Iter {i+1}/{self.max_iterations}: Added {best_iter_key}, Score: {self.best_score:.15f}"
                )
                self.history.append((best_iter_key, self.best_score))
            else:
                patience_counter += 1
                # print(f"Iter {i+1}: No improvement. Best iter score: {best_iter_score:.6f} vs Global Best: {self.best_score:.6f}")

            if patience_counter >= self.patience:
                print(f"Stopping early. No improvement for {self.patience} iterations.")
                break

        print(f"Selection Complete. Selected {len(self.selected_experts)} experts.")
        return self.selected_experts


# ==============================================================================
# PHASE 1: LIBRARY GENERATION & SELECTION
# ==============================================================================


def run_phase_1_selection(load_cached_data=True):
    """
    Generates the expert library, trains on Train, predicts on Val,
    and runs Greedy Selection.
    """
    # Define grid
    topologies = ["marginal", "rotational"]
    views = ["global", "combined"]
    shrinkages = LDA_SHRINKAGE_PARAMS

    val_predictions = {}
    y_val_global = None

    # Iterate through Data Views and Topologies first to minimize data loading/transform overhead
    for view in views:
        # Load Raw Data for this view
        (X_train_raw, y_train, _), (X_val_raw, y_val, _), _, _, _ = get_datasets(
            view=view, load_cached_data=load_cached_data
        )

        if y_val_global is None:
            y_val_global = y_val

        for topology in topologies:
            # Apply Topology Transform
            # This handles caching internally
            cache_name = f"{view}"
            X_train_trans, X_val_trans, _ = apply_topology(
                X_train_raw,
                X_val_raw,
                X_val_raw,  # Pass dummy for test here as we don't need it yet
                topology_name=topology,
                cache_name=cache_name,
                load_cached_data=load_cached_data,
            )

            # Iterate through Estimators
            for shrinkage in shrinkages:
                key = get_expert_key(topology, view, shrinkage)

                # Check if prediction is cached
                pred_cache_path = os.path.join(WORKING_DIR, f"pred_val_{key}.npy")

                if load_cached_data and os.path.exists(pred_cache_path):
                    # print(f"Loading cached predictions for {key}")
                    probas = np.load(pred_cache_path)
                else:
                    # Train and Predict
                    # print(f"Training expert: {key}")
                    clf = get_lda_expert(shrinkage)
                    clf.fit(X_train_trans, y_train)
                    probas = clf.predict_proba(X_val_trans)

                    # Cache prediction
                    np.save(pred_cache_path, probas)

                val_predictions[key] = probas

    # Run Selection
    selector = GreedySelector()
    selected_keys = selector.fit(val_predictions, y_val_global)

    return selected_keys


# ==============================================================================
# PHASE 2: RETRAINING & INFERENCE
# ==============================================================================


def run_phase_2_inference(selected_keys, load_cached_data=True):
    """
    Retrains selected experts on Full Train (Train+Val) and predicts on Test.
    """
    if not selected_keys:
        raise ValueError("No experts selected! Cannot proceed to inference.")

    # Count occurrences of each expert (for weighting)
    expert_counts = Counter(selected_keys)
    unique_experts = list(expert_counts.keys())

    print(f"\nStarting Phase 2: Retraining {len(unique_experts)} unique experts...")

    # To optimize, we group experts by (Topology, View)
    # structure: {(topology, view): [shrinkage_params...]}
    expert_groups = {}
    for key in unique_experts:
        t, v, s = parse_expert_key(key)
        if (t, v) not in expert_groups:
            expert_groups[(t, v)] = []
        expert_groups[(t, v)].append(s)

    # Initialize aggregated probabilities
    # We need to know n_samples and n_classes.
    # We'll get dimensions from the first processed batch.
    final_probas = None
    total_weight = 0
    test_ids = None
    class_names = None

    for (topology, view), shrinkage_list in expert_groups.items():
        # Load Full Data
        _, _, (X_test_raw, _, ids), (X_train_full_raw, y_train_full, _), classes = (
            get_datasets(view=view, load_cached_data=load_cached_data)
        )

        if test_ids is None:
            test_ids = ids
            class_names = classes

        # Apply Topology Transform (Train Full -> Test)
        # Note: apply_topology expects X_train, X_val, X_test.
        # We will pass X_train_full as X_train, and X_test as X_test. X_val is dummy.
        cache_name = f"{view}_full"

        # We need a slightly modified call or reuse apply_topology carefully.
        # apply_topology fits on the first arg.
        # We want to fit on X_train_full_raw.
        X_train_full_trans, _, X_test_trans = apply_topology(
            X_train_full_raw,
            X_test_raw,
            X_test_raw,  # 2nd arg is dummy
            topology_name=topology,
            cache_name=cache_name,
            load_cached_data=load_cached_data,
        )

        # Train specific shrinkage experts
        for shrinkage in shrinkage_list:
            key = get_expert_key(topology, view, shrinkage)
            weight = expert_counts[key]

            # print(f"Retraining {key} (Weight: {weight})...")

            clf = get_lda_expert(shrinkage)
            clf.fit(X_train_full_trans, y_train_full)
            probas = clf.predict_proba(X_test_trans)

            # Initialize final_probas if first iteration
            if final_probas is None:
                final_probas = np.zeros_like(probas, dtype=FLOAT_PRECISION)

            # Add weighted prediction
            final_probas += probas * weight
            total_weight += weight

    # Normalize by total weight
    final_probas /= total_weight

    # Post-process (Clip/Normalize)
    final_probas = postprocess_probabilities(final_probas)

    return test_ids, final_probas, class_names


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def run_ensemble_pipeline(load_cached_data=True):
    """
    Orchestrates the full Dynamic Ensemble Selection pipeline.
    """
    # Phase 1: Selection
    selected_keys = run_phase_1_selection(load_cached_data=load_cached_data)

    # Phase 2: Inference
    test_ids, predictions, class_names = run_phase_2_inference(
        selected_keys, load_cached_data=load_cached_data
    )

    # Save Submission
    print(f"Saving submission to {SUBMISSION_FILE}...")

    # Construct DataFrame
    df_sub = pd.DataFrame(predictions, columns=class_names)
    df_sub.insert(0, "id", test_ids)

    # Save
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print("Submission saved successfully.")
