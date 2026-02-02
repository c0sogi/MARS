import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from library.utils import calculate_log_loss
from library.config import WORKING_DIR, MARGIN_COLS, SHAPE_COLS, TEXTURE_COLS


class GreedySelector:
    """
    Implements Greedy Forward Selection with Replacement for ensemble optimization.
    Iteratively adds the expert that maximizes the improvement in Log Loss.
    """

    def __init__(self, n_iterations=100, tolerance=1e-6, verbose=True):
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.selected_experts = []
        self.weights = {}
        self.best_loss = float("inf")
        self.history = []

    def fit(self, predictions_dict, y_true):
        """
        Runs the greedy selection process.

        Args:
            predictions_dict (dict): Map of {expert_name: probability_matrix (N, C)}.
            y_true (np.array): Ground truth labels (N,). Must be integer encoded 0..C-1.
        """
        expert_names = sorted(list(predictions_dict.keys()))
        if not expert_names:
            raise ValueError("No predictions provided.")

        n_samples, n_classes = predictions_dict[expert_names[0]].shape

        # Initialize ensemble accumulator
        # We maintain the sum of probabilities of selected experts
        current_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        current_size = 0

        self.selected_experts = []
        self.best_loss = float("inf")
        self.history = []

        for it in range(self.n_iterations):
            best_iter_loss = float("inf")
            best_expert = None

            # Try adding each expert to the current ensemble
            for name in expert_names:
                pred = predictions_dict[name]

                # Calculate potential new ensemble probability
                # New Avg = (Sum + New_Pred) / (Size + 1)
                # Note: We calculate loss on the average, not the sum
                temp_proba = (current_sum + pred) / (current_size + 1)

                loss = calculate_log_loss(y_true, temp_proba)

                if loss < best_iter_loss:
                    best_iter_loss = loss
                    best_expert = name

            # Check for improvement
            if best_expert is None:
                if self.verbose:
                    print("No valid expert found to add.")
                break

            improvement = self.best_loss - best_iter_loss

            if self.verbose:
                print(
                    f"Iteration {it+1}: Best Expert={best_expert}, Loss={best_iter_loss}, Improvement={improvement}"
                )

            # Stop if improvement is negligible, but ensure at least one expert is selected
            if improvement < self.tolerance and current_size > 0:
                if self.verbose:
                    print(
                        f"Stopping: Improvement {improvement} < Tolerance {self.tolerance}"
                    )
                break

            # Update state
            self.best_loss = best_iter_loss
            self.selected_experts.append(best_expert)
            current_sum += predictions_dict[best_expert]
            current_size += 1

            self.history.append(
                {
                    "iteration": it + 1,
                    "added_expert": best_expert,
                    "loss": best_iter_loss,
                }
            )

        # Calculate final weights
        self.weights = dict(Counter(self.selected_experts))

        if self.verbose:
            print("Selection Complete.")
            print(f"Selected Experts: {self.weights}")

        return self

    def predict(self, predictions_dict):
        """
        Computes the weighted average prediction using selected experts.
        """
        if not self.weights:
            raise ValueError("Selector not fitted or no experts selected.")

        final_pred = None
        total_weight = 0.0

        for name, weight in self.weights.items():
            if name not in predictions_dict:
                # If a selected expert is missing from input (e.g. during inference if not all computed)
                raise KeyError(
                    f"Selected expert '{name}' not found in predictions dictionary."
                )

            pred = predictions_dict[name]

            if final_pred is None:
                final_pred = np.zeros_like(pred, dtype=np.float64)

            final_pred += pred * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight is zero.")

        return final_pred / total_weight


def _get_input_data(df, input_type):
    """
    Extracts the appropriate feature matrix from the dataframe based on input_type.
    """
    if input_type == "provided_features":
        cols = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS
        return df[cols].values
    elif input_type == "morphometrics":
        # Columns defined in features.py
        cols = [f"hu_{i}" for i in range(7)] + [
            "aspect_ratio",
            "solidity",
            "extent",
            "eccentricity",
        ]
        # Ensure columns exist
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing morphometric columns: {missing}")
        return df[cols].values
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def run_selection_phase(
    X_train, y_train, X_val, y_val, experts_list, load_cached_data=True
):
    """
    Orchestrates Phase 1:
    1. Trains all experts on Training set.
    2. Generates predictions on Validation set.
    3. Runs Greedy Forward Selection.

    Handles caching of validation predictions to speed up re-runs of selection logic.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        experts_list: List of expert configuration dicts.
        load_cached_data: Whether to load predictions from cache.

    Returns:
        selector: Fitted GreedySelector instance.
        le: Fitted LabelEncoder instance.
    """
    cache_path = os.path.join(WORKING_DIR, "val_predictions_cache.npz")

    # Encode Labels
    # We fit on all unique labels to ensure consistency across train/val
    le = LabelEncoder()
    all_labels = pd.concat([y_train, y_val]).unique()
    le.fit(all_labels)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    val_preds = {}

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached validation predictions from {cache_path}")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # Convert npz to dict
            val_preds = {k: loaded[k] for k in loaded.files}

            # Verify all experts in list are in cache
            missing_experts = [
                exp["name"] for exp in experts_list if exp["name"] not in val_preds
            ]
            if missing_experts:
                print(f"Cache missing experts: {missing_experts}. Re-computing all.")
                val_preds = {}  # Reset to force re-compute
        except Exception as e:
            print(f"Error loading cache: {e}. Re-computing.")
            val_preds = {}

    # 2. Compute if not cached
    if not val_preds:
        print("Training experts and generating validation predictions...")

        for expert in experts_list:
            name = expert["name"]
            print(f"Processing Expert: {name}")

            # Prepare Data
            X_tr_np = _get_input_data(X_train, expert["input_type"])
            X_val_np = _get_input_data(X_val, expert["input_type"])

            # Construct Model
            # Clone to ensure fresh start
            # Pipeline is preprocessing, Estimator is classifier
            model = make_pipeline(clone(expert["pipeline"]), clone(expert["estimator"]))

            # Fit
            model.fit(X_tr_np, y_train_enc)

            # Predict
            probs = model.predict_proba(X_val_np)
            val_preds[name] = probs

        # Save to cache
        os.makedirs(WORKING_DIR, exist_ok=True)
        np.savez(cache_path, **val_preds)
        print(f"Saved validation predictions to {cache_path}")

    # 3. Run Selection
    print("Running Greedy Forward Selection...")
    selector = GreedySelector(verbose=True)
    selector.fit(val_preds, y_val_enc)

    return selector, le
