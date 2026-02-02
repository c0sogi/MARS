import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import cohen_kappa_score
from library.config import Config
from library.utils import save_artifact, load_artifact
from library.metrics import compute_qwk


def _kappa_loss(coef, X, y):
    """
    Loss function for threshold optimization (Negative QWK).
    """
    X_p = np.digitize(X, coef) + 1
    return -cohen_kappa_score(y, X_p, weights="quadratic")


class ScoreRegressor:
    """
    A wrapper class for Ridge Regression model to predict essay scores.
    Implements training with hyperparameter tuning (alpha selection) and prediction with clipping.
    """

    def __init__(self, **kwargs):
        """
        Initializes the regressor.

        Args:
            kwargs: Arguments to override default Ridge parameters from Config.
        """
        self.model = None

        # Load default parameters from Config
        self.ridge_params = Config.RIDGE_PARAMS.copy()
        self.ridge_params.update(kwargs)

        # Separate alpha from other parameters as it might be tuned
        if "alpha" in self.ridge_params:
            self.default_alpha = self.ridge_params.pop("alpha")
        else:
            self.default_alpha = 1.0

        # Thresholds for optimized rounding
        self.thresholds = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model.

        If validation data (X_val, y_val) is provided, it performs a grid search over
        Config.ALPHA_GRID to find the alpha that maximizes the QWK score on the validation set.

        If validation data is NOT provided, it uses RidgeCV with 5-fold cross-validation
        to select the best alpha based on Mean Squared Error.

        Args:
            X_train: Training features (sparse matrix).
            y_train: Training targets.
            X_val: Validation features (sparse matrix), optional.
            y_val: Validation targets, optional.

        Returns:
            self
        """
        alphas = Config.ALPHA_GRID

        if X_val is not None and y_val is not None:
            print(f"Training with validation set. Tuning alpha over: {alphas}")

            best_score = -float("inf")
            best_model = None
            best_alpha = None

            for alpha in alphas:
                # Initialize Ridge with specific alpha and other config params
                # We use the solver specified in Config (likely sparse-efficient)
                model = Ridge(alpha=alpha, **self.ridge_params)

                # Fit on training data
                model.fit(X_train, y_train)

                # Predict on validation data
                y_pred_val = model.predict(X_val)

                # Clip predictions to valid range [1, 6]
                y_pred_val = np.clip(y_pred_val, Config.SCORE_MIN, Config.SCORE_MAX)

                # Round to nearest integer for QWK calculation
                y_pred_val_int = np.round(y_pred_val)

                # Compute QWK
                score = compute_qwk(y_val, y_pred_val_int)

                print(f"Alpha: {alpha}, Val QWK: {score}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_alpha = alpha

            print(f"Best Alpha: {best_alpha}")
            print(f"Best Val QWK: {best_score}")
            self.model = best_model

            # --- Threshold Optimization ---
            # Cite solution_lesson_node_00001: Addressing "regression to the mean" and tail performance
            print("Optimizing thresholds to maximize QWK...")

            # Get continuous predictions from the best model
            y_pred_val = self.model.predict(X_val)

            # Initial thresholds (standard rounding boundaries)
            initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5]

            # Optimize
            opt_res = minimize(
                _kappa_loss,
                initial_coef,
                args=(y_pred_val, y_val),
                method="nelder-mead",
                tol=1e-3,
            )

            self.thresholds = opt_res.x
            print(f"Optimized Thresholds: {self.thresholds}")

            # Check score with optimized thresholds
            opt_score = -opt_res.fun
            print(f"Val QWK with Optimized Thresholds: {opt_score}")

        else:
            print("No validation set provided. Using RidgeCV with 5-fold CV.")
            # Use RidgeCV for automatic tuning based on MSE
            # RidgeCV does not accept 'solver' or 'random_state' in the same way as Ridge
            # We pass alphas and cv.
            self.model = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
            self.model.fit(X_train, y_train)
            print(f"RidgeCV selected alpha: {self.model.alpha_}")

        return self

    def predict(self, X):
        """
        Generates predictions for the input features.

        Args:
            X: Input features (sparse matrix).

        Returns:
            np.array: Predicted scores.
                      If thresholds are optimized, returns integers (1-6).
                      Otherwise, returns continuous values clipped to [1, 6].
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Generate continuous predictions
        preds = self.model.predict(X)

        if self.thresholds is not None:
            # Use optimized thresholds
            # np.digitize returns indices 0..5, we map to 1..6
            return np.digitize(preds, self.thresholds) + 1
        else:
            # Fallback to standard clipping
            return np.clip(preds, Config.SCORE_MIN, Config.SCORE_MAX)

    def save(self, path):
        """
        Saves the trained model and thresholds to the specified path.

        Args:
            path (str): File path to save the model.
        """
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")

        # Save a dictionary containing the model and thresholds
        artifact = {"model": self.model, "thresholds": self.thresholds}
        save_artifact(artifact, path)

    @classmethod
    def load(cls, path):
        """
        Loads a trained model from the specified path.

        Args:
            path (str): File path to load the model from.

        Returns:
            ScoreRegressor: An instance of ScoreRegressor with the loaded model.
        """
        artifact = load_artifact(path)

        # Create a new instance
        instance = cls()

        # Handle backward compatibility or dict format
        if isinstance(artifact, dict) and "model" in artifact:
            instance.model = artifact["model"]
            instance.thresholds = artifact.get("thresholds")
        else:
            # Assume it's just the sklearn model (legacy)
            instance.model = artifact
            instance.thresholds = None

        # Attempt to retrieve alpha if available (for info)
        if hasattr(instance.model, "alpha_"):
            instance.default_alpha = instance.model.alpha_
        elif hasattr(instance.model, "alpha"):
            instance.default_alpha = instance.model.alpha

        return instance
