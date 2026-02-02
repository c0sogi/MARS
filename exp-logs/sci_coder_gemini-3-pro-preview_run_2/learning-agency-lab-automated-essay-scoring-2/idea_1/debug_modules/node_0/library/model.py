import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from library.config import Config
from library.utils import save_artifact, load_artifact
from library.metrics import compute_qwk


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
            np.array: Predicted scores, clipped to [Config.SCORE_MIN, Config.SCORE_MAX].
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Generate predictions
        preds = self.model.predict(X)

        # Clip to ensure valid range
        preds = np.clip(preds, Config.SCORE_MIN, Config.SCORE_MAX)

        return preds

    def save(self, path):
        """
        Saves the trained underlying sklearn model to the specified path.

        Args:
            path (str): File path to save the model.
        """
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        save_artifact(self.model, path)

    @classmethod
    def load(cls, path):
        """
        Loads a trained model from the specified path.

        Args:
            path (str): File path to load the model from.

        Returns:
            ScoreRegressor: An instance of ScoreRegressor with the loaded model.
        """
        sklearn_model = load_artifact(path)

        # Create a new instance
        instance = cls()
        instance.model = sklearn_model

        # Attempt to retrieve alpha if available (for info)
        if hasattr(sklearn_model, "alpha_"):
            instance.default_alpha = sklearn_model.alpha_
        elif hasattr(sklearn_model, "alpha"):
            instance.default_alpha = sklearn_model.alpha

        return instance
