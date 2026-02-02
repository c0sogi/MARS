import os
import numpy as np
from sklearn.metrics import log_loss
from library import config


class EnsembleOptimizer:
    """
    Optimizes the scalar weights for the Multi-Fidelity Ensemble branches
    using a discrete grid search on the simplex.
    """

    def __init__(self, step_size=0.01):
        """
        Initialize the optimizer.

        Args:
            step_size (float): The granularity of the grid search (default: 0.01).
                               Smaller values provide finer tuning but increase runtime.
        """
        self.step_size = step_size
        self.weights_cache_path = os.path.join(config.CACHE_DIR, "ensemble_weights.npy")

    def _generate_simplex_grid(self, n_weights):
        """
        Generates weight combinations that sum to 1.0.
        Recursive implementation to support arbitrary number of branches.

        Args:
            n_weights (int): Number of weights to generate.

        Yields:
            list: A list of floats summing to 1.0.
        """

        def generate(target_sum, n, current_weights):
            if n == 1:
                # The last weight takes whatever is remaining to ensure exact sum
                yield current_weights + [target_sum]
            else:
                # Iterate w from 0 to target_sum
                # Use int steps to avoid floating point accumulation errors during iteration
                n_steps = int(round(target_sum / self.step_size))
                for i in range(n_steps + 1):
                    w = i * self.step_size
                    # Floating point correction
                    if w > target_sum:
                        w = target_sum

                    # Recurse for the remaining weights
                    yield from generate(target_sum - w, n - 1, current_weights + [w])

        yield from generate(1.0, n_weights, [])

    def optimize(self, oof_preds_list, y_true, load_cached_data=True):
        """
        Finds the optimal weight combination that minimizes log loss on OOF predictions.

        Args:
            oof_preds_list (list of np.ndarray): List of probability matrices from each branch.
                                                 Shape of each: (N_samples, N_classes).
                                                 Order corresponds to config.PCA_THRESHOLDS.
            y_true (np.ndarray): True class indices. Shape: (N_samples,).
            load_cached_data (bool): If True, attempts to load weights from cache.

        Returns:
            list: Optimal weights corresponding to the input list order.
        """
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.weights_cache_path), exist_ok=True)

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(self.weights_cache_path):
            try:
                cached_weights = np.load(self.weights_cache_path)
                if len(cached_weights) == len(oof_preds_list):
                    print(
                        f"Loading cached ensemble weights from {self.weights_cache_path}..."
                    )
                    print(f"Cached Weights: {cached_weights.tolist()}")
                    return cached_weights.tolist()
                else:
                    print(
                        "Cached weights dimension mismatch (different number of branches). Re-optimizing..."
                    )
            except Exception as e:
                print(f"Error loading cache: {e}. Re-optimizing...")

        print(
            f"Starting Ensemble Weight Optimization (Grid Search, Step Size: {self.step_size})..."
        )

        n_branches = len(oof_preds_list)
        if n_branches == 0:
            raise ValueError("No OOF predictions provided for optimization.")

        # Prepare labels for log_loss (assumes all classes 0..K-1 are possible)
        n_classes = oof_preds_list[0].shape[1]
        labels = np.arange(n_classes)

        best_loss = float("inf")
        best_weights = [1.0 / n_branches] * n_branches

        # Stack predictions for vectorized weighted sum
        # Shape: (n_branches, n_samples, n_classes)
        stacked_preds = np.stack(oof_preds_list, axis=0)

        # 2. Grid Search
        grid = self._generate_simplex_grid(n_branches)
        count = 0

        for weights in grid:
            count += 1

            # Compute weighted average
            # Reshape weights for broadcasting: (n_branches, 1, 1)
            w_arr = np.array(weights).reshape(-1, 1, 1)
            p_ensemble = (stacked_preds * w_arr).sum(axis=0)

            # Normalize (Row sum = 1)
            # Required because floating point math might drift slightly from 1.0
            # and the task specification implies rescaling.
            row_sums = p_ensemble.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            p_ensemble /= row_sums

            # Clip probabilities to avoid log(0)
            # Using the epsilon defined in config
            p_ensemble = np.clip(
                p_ensemble, config.PROB_CLIP_EPS, 1.0 - config.PROB_CLIP_EPS
            )

            # Calculate Metric
            loss = log_loss(y_true, p_ensemble, labels=labels)

            if loss < best_loss:
                best_loss = loss
                best_weights = weights

        print(f"Optimization Complete. Evaluated {count} combinations.")
        print(f"Best Log Loss: {best_loss}")
        print(f"Best Weights: {best_weights}")

        # 3. Save to Cache
        try:
            np.save(self.weights_cache_path, np.array(best_weights))
            print(f"Weights saved to {self.weights_cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save weights to cache: {e}")

        return best_weights
