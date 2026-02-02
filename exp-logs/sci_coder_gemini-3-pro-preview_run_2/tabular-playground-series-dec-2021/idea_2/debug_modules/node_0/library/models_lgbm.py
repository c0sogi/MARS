import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
import library.config as config


class LGBMWrapper:
    def __init__(self, params=None):
        """
        Initialize the LightGBM wrapper.

        Args:
            params (dict, optional): Hyperparameters for LightGBM.
                                     Defaults to config.LGBM_PARAMS.
        """
        self.params = params if params is not None else config.LGBM_PARAMS.copy()
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LightGBM model with early stopping.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (pd.DataFrame or np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation labels.
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Extract training control parameters from config/params or set defaults
        num_boost_round = self.params.pop("n_estimators", 3000)
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        verbose_eval = 100  # Print progress every 100 rounds

        # Train the model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=verbose_eval),
            ],
        )

        # Calculate and print final validation accuracy with full precision
        if X_val is not None and y_val is not None:
            # Predict returns probabilities for multiclass
            y_pred_prob = self.model.predict(
                X_val, num_iteration=self.model.best_iteration
            )
            y_pred_class = np.argmax(y_pred_prob, axis=1)

            acc = accuracy_score(y_val, y_pred_class)
            print(f"Final Validation Accuracy: {acc}")

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X, num_iteration=self.model.best_iteration)
