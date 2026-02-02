import xgboost as xgb
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_log_mae
from library.features import COUPLING_TYPES


class CouplingPredictor:
    """
    A wrapper for the XGBoost Regressor to predict scalar coupling constants.
    Encapsulates model initialization, training with early stopping, and
    competition-specific metric evaluation.
    """

    def __init__(self):
        """
        Initializes the XGBoost model using the configuration parameters.
        """
        self.model = xgb.XGBRegressor(**Config.XGB_PARAMS)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
    ) -> None:
        """
        Trains the XGBoost model.

        If validation data is provided, it is used for early stopping and to
        calculate the Log MAE metric at the end of training.

        Args:
            X_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training target values.
            X_val (pd.DataFrame, optional): Validation feature set.
            y_val (pd.Series, optional): Validation target values.
        """
        eval_set = []
        if X_val is not None and y_val is not None:
            # XGBoost expects a list of (X, y) tuples for evaluation
            # We include train set to monitor overfitting
            eval_set = [(X_train, y_train), (X_val, y_val)]

        print("Starting XGBoost training...")
        self.model.fit(X_train, y_train, eval_set=eval_set, **Config.XGB_FIT_PARAMS)

        # After training, evaluate using the specific competition metric
        if X_val is not None and y_val is not None:
            self._evaluate_log_mae(X_val, y_val)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given features.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Predicted scalar coupling constants.
        """
        return self.model.predict(X)

    def _evaluate_log_mae(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Internal helper to calculate and print the Log MAE metric.

        This is necessary because the model is trained on 'type_enc' (integer),
        but the metric calculation requires the original string 'type' (e.g., '1JHC').
        We reconstruct the string column using the COUPLING_TYPES list.
        """
        print("\nCalculating Log MAE on Validation Set...")

        # Generate predictions
        preds = self.predict(X_val)

        # Prepare DataFrame for metric calculation
        # We create a copy to avoid modifying the input X_val
        eval_df = pd.DataFrame({Config.TARGET_COL: y_val.values, "pred": preds})

        # Reconstruct the 'type' column from 'type_enc'
        # COUPLING_TYPES is the sorted list used for encoding in library.features
        if Config.TYPE_ENC_COL in X_val.columns:
            # Create a decoder map: index -> type_string
            type_decoder = {i: t for i, t in enumerate(COUPLING_TYPES)}

            # Map the encoded integers back to strings
            # Ensure we cast to int to handle any potential float types from NaNs (though unlikely here)
            eval_df[Config.TYPE_COL] = (
                X_val[Config.TYPE_ENC_COL].astype(int).map(type_decoder)
            )

            # Calculate and print the metric
            score = calculate_log_mae(
                eval_df,
                pred_col="pred",
                target_col=Config.TARGET_COL,
                type_col=Config.TYPE_COL,
                verbose=True,
            )
        else:
            print(
                f"Warning: '{Config.TYPE_ENC_COL}' column not found in validation data. Skipping Log MAE calculation."
            )
