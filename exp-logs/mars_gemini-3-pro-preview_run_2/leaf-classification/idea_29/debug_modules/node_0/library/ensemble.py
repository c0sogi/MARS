import numpy as np
from collections import Counter
from library.utils import clipped_log_loss, set_seed


class GreedyForwardSelector:
    """
    Implements the Greedy Forward Selection algorithm to optimize ensemble weights.
    Iteratively adds the model that maximizes the improvement in the evaluation metric.
    """

    def __init__(self, selection_iterations=50, random_seed=42):
        """
        Args:
            selection_iterations (int): The number of models to add to the ensemble (sum of weights).
            random_seed (int): Seed for reproducibility.
        """
        self.selection_iterations = selection_iterations
        self.random_seed = random_seed
        self.selected_weights = {}
        self.best_score = float("inf")

    def fit(self, preds_dict, y_true):
        """
        Runs the greedy forward selection process.

        Args:
            preds_dict (dict): A dictionary where keys are model identifiers (str) and values
                               are numpy arrays of probabilities with shape (n_samples, n_classes).
            y_true (np.ndarray): The ground truth labels.

        Returns:
            dict: A dictionary mapping model identifiers to their selected integer weights.
        """
        set_seed(self.random_seed)

        # Extract keys to ensure consistent ordering
        model_keys = list(preds_dict.keys())

        # Stack predictions into a tensor for efficient iteration: (n_models, n_samples, n_classes)
        # Ensure float64 for precision
        candidate_preds = np.array(
            [preds_dict[k] for k in model_keys], dtype=np.float64
        )

        n_models, n_samples, n_classes = candidate_preds.shape

        # Initialize ensemble accumulator (sum of probabilities)
        ensemble_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        selected_indices = []

        print(
            f"Starting Greedy Forward Selection with {n_models} candidates for {self.selection_iterations} iterations..."
        )

        for i in range(self.selection_iterations):
            iteration_best_loss = float("inf")
            iteration_best_idx = -1

            # Current size of the ensemble (number of models added so far)
            current_size = len(selected_indices)

            # Try adding each candidate model to the current ensemble
            for idx in range(n_models):
                # Calculate the potential new ensemble sum
                temp_sum = ensemble_sum + candidate_preds[idx]

                # Calculate average probabilities
                # Note: clipped_log_loss handles row normalization, but we average here
                # to keep the scale consistent with the probability definition.
                temp_avg = temp_sum / (current_size + 1)

                # Evaluate metric
                loss = clipped_log_loss(y_true, temp_avg)

                if loss < iteration_best_loss:
                    iteration_best_loss = loss
                    iteration_best_idx = idx

            # Update the ensemble with the best candidate found in this iteration
            if iteration_best_idx != -1:
                selected_indices.append(iteration_best_idx)
                ensemble_sum += candidate_preds[iteration_best_idx]
                self.best_score = iteration_best_loss

                best_model_name = model_keys[iteration_best_idx]
                print(
                    f"Iteration {i+1}: Added {best_model_name}, Validation Log Loss: {self.best_score:.15f}"
                )
            else:
                print(f"Iteration {i+1}: No improvement possible.")
                break

        # Aggregate indices into weights
        counts = Counter(selected_indices)
        self.selected_weights = {
            model_keys[idx]: count for idx, count in counts.items()
        }

        print("\nFinal Optimized Ensemble Weights:")
        for name, weight in self.selected_weights.items():
            print(f"  {name}: {weight}")

        return self.selected_weights
