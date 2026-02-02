import numpy as np
from collections import Counter
from library.utils import compute_log_loss, set_seed
from library.config import DTYPE


class GreedySelector:
    """
    Implements Greedy Forward Selection (Caruana Ensemble Selection) to optimize
    ensemble weights based on validation log loss.
    """

    def __init__(self, n_iterations=100, random_state=42):
        """
        Args:
            n_iterations (int): The number of models to select (ensemble size).
                                Allows for integer weighting (selection with replacement).
            random_state (int): Seed for reproducibility (if ties occur).
        """
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.selected_models = []
        self.best_loss_history = []

    def fit(self, library_preds, y_true, classes):
        """
        Fits the ensemble selection by iteratively adding the model that minimizes
        log loss of the ensemble mean.

        Args:
            library_preds (dict): Dictionary where keys are model names and values are
                                  prediction matrices (np.ndarray) of shape (n_samples, n_classes).
            y_true (np.ndarray): True class labels (strings or ints).
            classes (list): List of class names corresponding to the columns of prediction matrices.
        """
        set_seed(self.random_state)

        model_names = list(library_preds.keys())
        n_models = len(model_names)

        if n_models == 0:
            raise ValueError("library_preds dictionary is empty.")

        # Ensure all predictions are float64
        for name in model_names:
            library_preds[name] = library_preds[name].astype(DTYPE)

        # Initialize ensemble sum (accumulates probabilities)
        # We use a sum accumulator to avoid repeated division/mean calculation overhead
        n_samples, n_classes = library_preds[model_names[0]].shape
        current_ensemble_sum = np.zeros((n_samples, n_classes), dtype=DTYPE)

        print(
            f"Starting Greedy Forward Selection with {self.n_iterations} iterations..."
        )

        for i in range(1, self.n_iterations + 1):
            best_iter_loss = float("inf")
            best_model_name = None

            # Try adding each candidate model to the current ensemble
            for name in model_names:
                candidate_pred = library_preds[name]

                # Calculate temporary mean: (current_sum + candidate) / i
                temp_ensemble_pred = (current_ensemble_sum + candidate_pred) / i

                # Compute metric
                loss = compute_log_loss(y_true, temp_ensemble_pred, classes=classes)

                if loss < best_iter_loss:
                    best_iter_loss = loss
                    best_model_name = name

            # Update ensemble with the winner of this round
            self.selected_models.append(best_model_name)
            current_ensemble_sum += library_preds[best_model_name]
            self.best_loss_history.append(best_iter_loss)

            # Print progress periodically
            if i % 10 == 0 or i == 1 or i == self.n_iterations:
                print(
                    f"Iteration {i}/{self.n_iterations}: Selected '{best_model_name}' - Ensemble Log Loss: {best_iter_loss:.15f}"
                )

        print("Selection complete.")

    def get_best_weights(self):
        """
        Returns the weights (counts) of each selected model.

        Returns:
            dict: Mapping of model_name -> weight (int).
        """
        return dict(Counter(self.selected_models))


def aggregate_predictions(library_preds, weights):
    """
    Computes the weighted average of predictions based on the provided weights.

    Args:
        library_preds (dict): Dictionary of prediction matrices (e.g., for test set).
        weights (dict): Dictionary mapping model names to integer weights.

    Returns:
        np.ndarray: The aggregated probability matrix.
    """
    if not weights:
        raise ValueError("Weights dictionary is empty.")

    # Get shape from first model
    first_model = next(iter(weights))
    n_samples, n_classes = library_preds[first_model].shape

    weighted_sum = np.zeros((n_samples, n_classes), dtype=DTYPE)
    total_weight = 0.0

    for name, weight in weights.items():
        if name not in library_preds:
            raise KeyError(
                f"Model '{name}' found in weights but missing from prediction library."
            )

        # Add weighted contribution
        weighted_sum += library_preds[name].astype(DTYPE) * weight
        total_weight += weight

    if total_weight == 0:
        raise ValueError("Total weight is zero.")

    # Compute mean
    final_pred = weighted_sum / total_weight

    return final_pred
