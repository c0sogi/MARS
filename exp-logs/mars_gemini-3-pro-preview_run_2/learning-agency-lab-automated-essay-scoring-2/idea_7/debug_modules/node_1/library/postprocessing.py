import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.utils import compute_qwk


class ThresholdOptimizer:
    """
    Optimizes decision boundaries (thresholds) for converting continuous regression
    predictions into integer scores (1-6) to maximize the Quadratic Weighted Kappa (QWK).
    Uses the Nelder-Mead algorithm to find the optimal cutoffs.
    """

    def __init__(self, initial_coefs=None):
        """
        Initialize the optimizer with starting thresholds.

        Args:
            initial_coefs (list or np.ndarray, optional): Initial thresholds.
                Defaults to [1.5, 2.5, 3.5, 4.5, 5.5] for the 1-6 scale.
        """
        if initial_coefs is not None:
            self.coef_ = np.array(initial_coefs)
        else:
            # Default midpoints for 1-6 scale
            self.coef_ = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    def _kappa_loss(self, coef, X, y):
        """
        Loss function to minimize: Negative QWK.

        Args:
            coef (np.ndarray): Current thresholds.
            X (np.ndarray): Continuous predictions.
            y (np.ndarray): True integer labels.

        Returns:
            float: Negative QWK score.
        """
        # Ensure coefficients are sorted for valid binning
        c = np.sort(coef)

        # Define bins: (-inf, c[0], c[1], ..., c[4], inf)
        # This creates 6 intervals corresponding to scores 1 through 6
        bins = [-np.inf] + list(c) + [np.inf]

        # Labels for the bins
        labels = [1, 2, 3, 4, 5, 6]

        # Digitize predictions
        # pd.cut handles the binning robustly
        preds_discrete = pd.cut(X, bins=bins, labels=labels).astype(int)

        # Compute QWK
        # We negate it because we are using a minimizer
        score = compute_qwk(y, preds_discrete)
        return -score

    def fit(self, X, y):
        """
        Fit the thresholds to the data using Nelder-Mead optimization.

        Args:
            X (np.ndarray): Continuous predictions.
            y (np.ndarray): True integer labels.
        """
        # Define partial function for the optimizer
        loss_partial = lambda coef: self._kappa_loss(coef, X, y)

        # Run optimization
        # Nelder-Mead is chosen as the objective function is non-differentiable (step function)
        # and we need a derivative-free optimization method.
        result = minimize(
            loss_partial,
            self.coef_,
            method="nelder-mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 1000},
        )

        # Update coefficients with the optimized values
        self.coef_ = np.sort(result.x)

        # Calculate final score for reporting
        final_score = -result.fun
        print(f"Threshold optimization complete. Best QWK: {final_score}")

    def predict(self, X):
        """
        Convert continuous predictions to integer scores using the optimized thresholds.

        Args:
            X (np.ndarray): Continuous predictions.

        Returns:
            np.ndarray: Integer scores (1-6).
        """
        c = np.sort(self.coef_)
        bins = [-np.inf] + list(c) + [np.inf]
        labels = [1, 2, 3, 4, 5, 6]

        # Apply thresholds
        preds_discrete = pd.cut(X, bins=bins, labels=labels).astype(int)

        # Return as numpy array
        return np.array(preds_discrete)
