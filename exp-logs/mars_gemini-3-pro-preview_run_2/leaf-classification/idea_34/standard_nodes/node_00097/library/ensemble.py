import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.base import clone
from library import config


class GreedySelector:
    """
    Implements Greedy Forward Selection for Dynamic Ensemble Selection.

    Selects a subset of experts from a library by iteratively adding the model
    that minimizes the ensemble's Log Loss on the validation set.
    """

    def __init__(self, max_iterations=20, tolerance=1e-6):
        """
        Args:
            max_iterations (int): Maximum number of experts to add to the ensemble.
            tolerance (float): Minimum improvement in Log Loss required to continue adding experts.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.selected_experts = (
            []
        )  # List of dicts: {'expert_def': dict, 'instance': model}
        self.weights = []  # List of integer weights corresponding to selected_experts
        self.classes = None
        self.best_score = float("inf")

    def _get_X(self, data_split, view):
        """Helper to retrieve the correct feature matrix based on view."""
        if view == "global":
            return data_split["X_global"]
        elif view == "macro":
            return data_split["X_macro"]
        else:
            raise ValueError(f"Unknown view: {view}")

    def fit(self, experts, data):
        """
        Trains all experts on the training set, evaluates them on the validation set,
        and performs greedy forward selection to determine the optimal ensemble.

        Args:
            experts (list): List of expert definitions from expert_factory.
            data (dict): Data dictionary containing 'train' and 'val' splits.
        """
        print(f"Starting Greedy Selection with {len(experts)} candidate experts...")

        self.classes = data["classes"]
        y_val = data["val"]["y"]

        # ---------------------------------------------------------------------
        # 1. Train Library and Generate Validation Predictions
        # ---------------------------------------------------------------------
        val_predictions = []
        trained_instances = []

        for i, expert in enumerate(experts):
            # Clone to ensure fresh start
            model = clone(expert["model"])
            view = expert["view"]

            X_train = self._get_X(data["train"], view)
            y_train = data["train"]["y"]
            X_val = self._get_X(data["val"], view)

            # Fit model
            model.fit(X_train, y_train)

            # Predict on validation
            # Ensure float64 precision
            probs = model.predict_proba(X_val).astype(config.FLOAT_PRECISION)

            # Store
            val_predictions.append(probs)
            trained_instances.append(model)

        val_predictions = np.array(val_predictions, dtype=config.FLOAT_PRECISION)

        # ---------------------------------------------------------------------
        # 2. Greedy Forward Selection
        # ---------------------------------------------------------------------
        # Initialize ensemble prediction (starts as None)
        current_ens_pred = None
        selected_indices = []

        print("-" * 40)
        print(
            f"{'Iter':<5} | {'Expert Added':<25} | {'Val LogLoss':<15} | {'Improvement':<15}"
        )
        print("-" * 40)

        for it in range(self.max_iterations):
            best_iter_score = float("inf")
            best_idx = -1

            # Try adding each expert to the current ensemble
            for idx in range(len(experts)):
                candidate_pred = val_predictions[idx]

                if current_ens_pred is None:
                    # First iteration: ensemble is just this candidate
                    temp_ens_pred = candidate_pred
                else:
                    # Weighted average: (current_sum + candidate) / (current_count + 1)
                    # We maintain the running sum to avoid re-dividing constantly
                    # current_ens_pred here represents the AVERAGE of previous steps.
                    # To update efficiently: New_Avg = (Old_Avg * k + New_Pred) / (k + 1)
                    k = len(selected_indices)
                    temp_ens_pred = (current_ens_pred * k + candidate_pred) / (k + 1)

                # Clip probabilities for stability before scoring
                # Note: sklearn log_loss does this internally, but we do it explicitly for consistency
                temp_ens_pred_clipped = np.clip(
                    temp_ens_pred, config.CLIP_MIN, config.CLIP_MAX
                )

                # Renormalize to ensure sum to 1 (metric requirement)
                temp_ens_pred_clipped /= temp_ens_pred_clipped.sum(
                    axis=1, keepdims=True
                )

                score = log_loss(y_val, temp_ens_pred_clipped, labels=self.classes)

                if score < best_iter_score:
                    best_iter_score = score
                    best_idx = idx

            # Check for improvement
            improvement = self.best_score - best_iter_score

            if improvement > self.tolerance or (it == 0):
                self.best_score = best_iter_score
                selected_indices.append(best_idx)

                # Update current ensemble prediction
                candidate_pred = val_predictions[best_idx]
                if current_ens_pred is None:
                    current_ens_pred = candidate_pred
                else:
                    k = len(selected_indices) - 1  # Count before adding
                    current_ens_pred = (current_ens_pred * k + candidate_pred) / (k + 1)

                expert_id = experts[best_idx]["id"]
                print(
                    f"{it+1:<5} | {expert_id:<25} | {self.best_score:.15f} | {improvement:.15f}"
                )
            else:
                print(
                    f"Stopping: Improvement {improvement:.6e} < Tolerance {self.tolerance}"
                )
                break

        # ---------------------------------------------------------------------
        # 3. Finalize Selection
        # ---------------------------------------------------------------------
        # Count occurrences of each expert to determine weights
        unique_indices, counts = np.unique(selected_indices, return_counts=True)

        self.selected_experts = []
        self.weights = []

        print("-" * 40)
        print("Final Ensemble Composition:")
        for idx, count in zip(unique_indices, counts):
            expert_def = experts[idx]
            self.selected_experts.append(
                {"expert_def": expert_def, "instance": None}  # Will be refitted later
            )
            self.weights.append(count)
            print(f"  - {expert_def['id']}: Weight {count}")

        print(f"Final Validation Log Loss: {self.best_score:.15f}")

    def refit(self, data):
        """
        Retrains the selected experts on the combined Training + Validation dataset.

        Args:
            data (dict): Data dictionary containing 'train' and 'val' splits.
        """
        print("Refitting selected experts on combined (Train + Val) data...")

        # Combine data
        X_global_full = np.vstack((data["train"]["X_global"], data["val"]["X_global"]))
        X_macro_full = np.vstack((data["train"]["X_macro"], data["val"]["X_macro"]))
        y_full = np.concatenate((data["train"]["y"], data["val"]["y"]))

        # Helper to get full data based on view
        def get_full_X(view):
            if view == "global":
                return X_global_full
            elif view == "macro":
                return X_macro_full
            else:
                raise ValueError(f"Unknown view: {view}")

        # Refit each unique selected expert
        for item in self.selected_experts:
            expert_def = item["expert_def"]
            print(f"  Refitting {expert_def['id']}...")

            # Create a fresh clone
            model = clone(expert_def["model"])
            X_full = get_full_X(expert_def["view"])

            model.fit(X_full, y_full)
            item["instance"] = model

    def predict(self, data):
        """
        Generates predictions for the test set using the ensemble.

        Args:
            data (dict): Data dictionary containing 'test' split.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        print("Generating ensemble predictions on Test set...")

        test_predictions = np.zeros(
            (len(data["test"]["ids"]), len(self.classes)), dtype=config.FLOAT_PRECISION
        )
        total_weight = sum(self.weights)

        for item, weight in zip(self.selected_experts, self.weights):
            model = item["instance"]
            view = item["expert_def"]["view"]

            X_test = self._get_X(data["test"], view)

            # Predict
            probs = model.predict_proba(X_test).astype(config.FLOAT_PRECISION)

            # Add weighted prediction
            test_predictions += probs * weight

        # Normalize
        test_predictions /= total_weight

        # Clip and Renormalize (Final Safety)
        test_predictions = np.clip(test_predictions, config.CLIP_MIN, config.CLIP_MAX)
        test_predictions /= test_predictions.sum(axis=1, keepdims=True)

        return test_predictions
