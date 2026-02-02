import numpy as np
from sklearn.metrics import log_loss
import library.config as conf


class GreedyEnsembleSelector:
    """
    Implements Greedy Forward Selection (Caruana et al.) to find optimal
    ensemble weights by minimizing log loss on a validation set.
    """

    def __init__(self, n_iterations=100, random_state=conf.RANDOM_SEED):
        """
        Args:
            n_iterations (int): Number of iterations for the greedy selection process.
                                Higher values allow for finer-grained weights.
            random_state (int): Seed for reproducibility (if needed, though this alg is deterministic
                                given a fixed order of keys).
        """
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights_ = {}
        self.best_score_ = float("inf")
        self.selected_models_ = []

    def fit(self, predictions_dict, y_true):
        """
        Fits the ensemble weights using greedy forward selection.

        Args:
            predictions_dict (dict): Dictionary where keys are model names and
                                     values are numpy arrays of shape (n_samples, n_classes).
            y_true (array-like): True class labels or indices.
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)

        if n_models == 0:
            raise ValueError("predictions_dict cannot be empty.")

        # 1. Find the single best model to start with
        best_single_score = float("inf")
        best_single_model = None

        print(f"Evaluating {n_models} base models on validation set...")

        for name, preds in predictions_dict.items():
            # Use sklearn's log_loss which handles the clipping (eps=1e-15 default)
            score = log_loss(y_true, preds, labels=list(range(preds.shape[1])))
            print(f"  Model: {name:<15} | Log Loss: {score:.15f}")

            if score < best_single_score:
                best_single_score = score
                best_single_model = name

        print(
            f"\nBest single model: {best_single_model} (Score: {best_single_score:.15f})"
        )

        # 2. Initialize ensemble with the best single model
        self.selected_models_ = [best_single_model]
        # Current sum of probabilities (unnormalized)
        current_sum = predictions_dict[best_single_model].copy()
        self.best_score_ = best_single_score

        # 3. Greedy Selection Loop
        print(
            f"\nStarting Greedy Forward Selection ({self.n_iterations} iterations)..."
        )

        for i in range(1, self.n_iterations + 1):
            best_iter_score = float("inf")
            best_iter_model = None

            # Try adding each model to the current ensemble
            # The new ensemble size will be len(self.selected_models_) + 1
            current_size = len(self.selected_models_)
            new_size = current_size + 1

            for name in model_names:
                preds = predictions_dict[name]

                # Calculate potential new average
                # (current_sum + preds) / new_size
                temp_ensemble_preds = (current_sum + preds) / new_size

                score = log_loss(
                    y_true, temp_ensemble_preds, labels=list(range(preds.shape[1]))
                )

                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_model = name

            # Update ensemble with the winner of this round
            self.selected_models_.append(best_iter_model)
            current_sum += predictions_dict[best_iter_model]
            self.best_score_ = best_iter_score

            # Verbose logging
            if i <= 10 or i % 10 == 0 or i == self.n_iterations:
                print(
                    f"  Iter {i:3d}/{self.n_iterations} | Added: {best_iter_model:<15} | Ensemble Log Loss: {self.best_score_:.15f}"
                )

        # 4. Compute final weights
        self.weights_ = {name: 0 for name in model_names}
        for name in self.selected_models_:
            self.weights_[name] += 1

        print("\nFinal Ensemble Weights:")
        for name, weight in self.weights_.items():
            if weight > 0:
                print(
                    f"  {name:<15}: {weight}/{self.n_iterations} ({weight/self.n_iterations:.4f})"
                )

        return self

    def predict(self, predictions_dict):
        """
        Generates weighted average predictions using the fitted weights.

        Args:
            predictions_dict (dict): Dictionary where keys are model names and
                                     values are numpy arrays of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Weighted average probabilities of shape (n_samples, n_classes).
        """
        if not self.weights_:
            raise RuntimeError("Model must be fitted before calling predict.")

        # Initialize result array
        # Get shape from the first model in the dict
        first_preds = next(iter(predictions_dict.values()))
        weighted_sum = np.zeros_like(first_preds)
        total_weight = 0

        for name, weight in self.weights_.items():
            if weight > 0:
                if name not in predictions_dict:
                    raise KeyError(
                        f"Model '{name}' found in weights but missing from input predictions."
                    )

                weighted_sum += weight * predictions_dict[name]
                total_weight += weight

        if total_weight == 0:
            raise RuntimeError("Total ensemble weight is zero. Check fit process.")

        return weighted_sum / total_weight
