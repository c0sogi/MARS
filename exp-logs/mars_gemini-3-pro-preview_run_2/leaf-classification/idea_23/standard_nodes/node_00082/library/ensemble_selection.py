import numpy as np
from sklearn.base import clone
from library.utils import clipped_log_loss, set_seed
from library.config import SELECTION_ITERATIONS, RANDOM_SEED


class GreedySelector:
    """
    Implements Greedy Forward Selection for ensemble optimization.

    This class manages the lifecycle of the ensemble:
    1. Training candidate experts.
    2. Selecting the optimal subset of experts based on validation log loss.
    3. Retraining selected experts on full data.
    4. Generating weighted predictions.
    """

    def __init__(
        self,
        experts,
        n_iterations=SELECTION_ITERATIONS,
        seed=RANDOM_SEED,
        patience=10,
        tolerance=1e-6,
    ):
        """
        Args:
            experts (list): List of dictionaries defining the experts (model, name, view).
            n_iterations (int): Maximum number of experts to select (sum of weights).
            seed (int): Random seed for reproducibility.
            patience (int): Number of iterations to wait for improvement before early stopping.
            tolerance (float): Minimum improvement required to reset patience.
        """
        self.experts = experts
        self.n_iterations = n_iterations
        self.seed = seed
        self.patience = patience
        self.tolerance = tolerance

        # State
        self.selected_indices = []  # List of indices of experts in self.experts
        self.fitted_models = {}  # Map: expert_index -> trained model object
        self.best_score = float("inf")
        self.ensemble_history = []  # To track score progression

    def fit(self, data_map):
        """
        Trains all experts on training splits and runs greedy selection using validation splits.

        Args:
            data_map (dict): Dictionary mapping view names to (X_train, y_train, X_val, y_val).
                             Example: {'Global': (X_tr, y_tr, X_val, y_val), ...}
        """
        set_seed(self.seed)
        print(f"Initializing Greedy Selector with {len(self.experts)} experts.")

        # 1. Train all experts and generate validation predictions
        # We store predictions in memory: (n_experts, n_val_samples, n_classes)
        val_preds = []

        # We assume y_val is consistent across views. Retrieve it from the first available view.
        first_view_key = next(iter(data_map.keys()))
        y_val_target = data_map[first_view_key][3]

        print("Training experts and generating validation predictions...")
        for i, expert in enumerate(self.experts):
            view_name = expert["view"]
            if view_name not in data_map:
                raise ValueError(
                    f"View '{view_name}' required by expert '{expert['name']}' not found in data_map."
                )

            X_train, y_train, X_val, y_val = data_map[view_name]

            # Sanity check for target consistency
            if not np.array_equal(y_val, y_val_target):
                # In a robust system we might handle this, but here we assume DataManager consistency
                pass

            # Clone and train
            model = clone(expert["model"])
            model.fit(X_train, y_train)

            # Predict
            # Ensure float64 for precision
            p = model.predict_proba(X_val).astype(np.float64)
            val_preds.append(p)

        val_preds = np.array(val_preds)  # Shape: (n_experts, n_samples, n_classes)

        # 2. Run Greedy Forward Selection
        print(f"Starting Selection Loop (Max Iterations: {self.n_iterations})...")

        # Initialize ensemble accumulator
        # We accumulate sum of probabilities, then divide by k (current size) for averaging
        current_sum = np.zeros_like(val_preds[0])
        self.selected_indices = []
        self.best_score = float("inf")

        patience_counter = 0

        for k in range(1, self.n_iterations + 1):
            best_iter_score = float("inf")
            best_expert_idx = -1

            # Try adding each expert to the current ensemble
            for i in range(len(self.experts)):
                # Calculate potential new ensemble prediction
                # New Average = (Current Sum + Candidate Pred) / k
                trial_pred = (current_sum + val_preds[i]) / k

                # Evaluate
                score = clipped_log_loss(y_val_target, trial_pred)

                if score < best_iter_score:
                    best_iter_score = score
                    best_expert_idx = i

            # Check for improvement
            improvement = self.best_score - best_iter_score

            # Update Ensemble
            self.selected_indices.append(best_expert_idx)
            current_sum += val_preds[best_expert_idx]

            expert_name = self.experts[best_expert_idx]["name"]
            self.ensemble_history.append((k, best_iter_score, expert_name))

            print(
                f"Iter {k}: Selected {expert_name} | Val Log Loss: {best_iter_score:.6f} | Improvement: {improvement:.2e}"
            )

            # Update best score
            if best_iter_score < self.best_score:
                self.best_score = best_iter_score

            # Early Stopping Logic
            if improvement < self.tolerance:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f"Early stopping triggered at iteration {k}. No significant improvement for {self.patience} steps."
                    )
                    break
            else:
                patience_counter = 0

        print(f"Selection Complete. Final Ensemble Size: {len(self.selected_indices)}")
        print(f"Best Validation Log Loss: {self.best_score:.6f}")

    def refit(self, full_data_map):
        """
        Retrains the selected experts on the full dataset (Train + Val).

        Args:
            full_data_map (dict): Dictionary mapping view names to (X_full, y_full).
        """
        if not self.selected_indices:
            raise ValueError("No experts selected. Call fit() first.")

        print("Retraining selected experts on full dataset...")
        set_seed(self.seed)

        # Identify unique experts to avoid redundant training
        # We still keep track of weights via self.selected_indices, but we only need one model instance per unique expert
        unique_indices = set(self.selected_indices)
        self.fitted_models = {}

        for idx in unique_indices:
            expert = self.experts[idx]
            view_name = expert["view"]

            if view_name not in full_data_map:
                raise ValueError(f"View '{view_name}' not found in full_data_map.")

            X_full, y_full = full_data_map[view_name]

            # Clone and Fit
            model = clone(expert["model"])
            model.fit(X_full, y_full)

            self.fitted_models[idx] = model

        print(f"Retrained {len(self.fitted_models)} unique models.")

    def predict(self, test_data_map):
        """
        Generates predictions for the test set using the refitted ensemble.

        Args:
            test_data_map (dict): Dictionary mapping view names to X_test.

        Returns:
            np.ndarray: Weighted probability predictions of shape (n_test_samples, n_classes).
        """
        if not self.fitted_models:
            raise ValueError("Models not fitted. Call refit() before predict().")

        # Determine output shape from the first expert
        first_idx = self.selected_indices[0]
        first_view = self.experts[first_idx]["view"]
        n_samples = test_data_map[first_view].shape[0]

        # Get number of classes from the fitted model
        n_classes = len(self.fitted_models[first_idx].classes_)

        # Initialize accumulator
        final_sum = np.zeros((n_samples, n_classes), dtype=np.float64)

        # Cache predictions for unique models to optimize inference
        unique_preds = {}

        for idx in self.fitted_models:
            model = self.fitted_models[idx]
            view_name = self.experts[idx]["view"]
            X_test = test_data_map[view_name]

            # Predict and cast to float64
            unique_preds[idx] = model.predict_proba(X_test).astype(np.float64)

        # Aggregate based on selection frequency (weights)
        for idx in self.selected_indices:
            final_sum += unique_preds[idx]

        # Average
        final_pred = final_sum / len(self.selected_indices)

        return final_pred
