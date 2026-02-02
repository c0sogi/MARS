import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS


def train_target_model(
    X_train, y_train, X_val, y_val, target_name, early_stopping_rounds=100
):
    """
    Trains an XGBoost Regressor for a single target variable using the provided training and validation data.

    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target vector for training (log-transformed).
        X_val (pd.DataFrame): Feature matrix for validation.
        y_val (pd.Series): Target vector for validation (log-transformed).
        target_name (str): Name of the target variable (for logging purposes).
        early_stopping_rounds (int): Number of rounds with no improvement to trigger early stopping.

    Returns:
        xgb.XGBRegressor: The trained XGBoost model.
    """
    print(f"\n========== Training Model for Target: {target_name} ==========")

    # Initialize the model with hyperparameters from config
    model = xgb.XGBRegressor(**XGB_PARAMS)

    # Fit the model
    # We pass both train and val sets to eval_set to monitor overfitting
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=200,  # Print progress every 200 rounds
    )

    # --- Evaluation ---
    # Generate predictions on validation set (log space)
    y_pred_log = model.predict(X_val)

    # Calculate RMSLE (Root Mean Squared Logarithmic Error) - since targets are already log1p
    mse_log = mean_squared_error(y_val, y_pred_log)
    rmse_log = np.sqrt(mse_log)

    # Calculate RMSE in original space for physical interpretability
    # Inverse transform: exp(y) - 1
    y_val_orig = np.expm1(y_val)
    y_pred_orig = np.expm1(y_pred_log)
    mse_orig = mean_squared_error(y_val_orig, y_pred_orig)
    rmse_orig = np.sqrt(mse_orig)

    print(f"\n>>> Metrics for {target_name}:")
    print(f"    Validation RMSLE (log-space): {rmse_log}")
    print(f"    Validation RMSE (original-space): {rmse_orig}")
    print("============================================================\n")

    return model


def make_predictions(model, X_test):
    """
    Generates predictions for the test set using the trained model.

    Args:
        model (xgb.XGBRegressor): The trained model.
        X_test (pd.DataFrame): Feature matrix for testing.

    Returns:
        np.ndarray: Predicted values in log-space.
    """
    return model.predict(X_test)
