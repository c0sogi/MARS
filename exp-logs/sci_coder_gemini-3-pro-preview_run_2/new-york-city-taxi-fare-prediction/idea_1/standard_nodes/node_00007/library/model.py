import os
import numpy as np
import lightgbm as lgb
from library.config import Config


class TaxiFareModel:
    """
    A Gradient Boosting Decision Tree model for NYC Taxi Fare Prediction.
    Wraps LightGBM with physics-informed constraints (min fare).
    Cite {solution_lesson_node_00003}: Tree-based models are robust to outliers
    and do not extrapolate errors like linear models.
    """

    def __init__(self):
        self.params = Config.get_model_params()
        self.model = lgb.LGBMRegressor(**self.params)
        self.min_fare = Config.MIN_FARE

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with early stopping.
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=50),
        ]

        self.model.fit(
            X_train, y_train, eval_set=eval_set, eval_metric="rmse", callbacks=callbacks
        )

    def predict(self, X):
        """
        Predicts fare amounts and enforces minimum fare constraint.
        """
        predictions = self.model.predict(X)
        # Enforce lower bound (Physics/Domain constraint)
        return np.maximum(predictions, self.min_fare)

    def save(self, path):
        # Placeholder for compatibility, though not strictly used in current runfile
        pass

    def load(self, path):
        # Placeholder for compatibility
        pass
