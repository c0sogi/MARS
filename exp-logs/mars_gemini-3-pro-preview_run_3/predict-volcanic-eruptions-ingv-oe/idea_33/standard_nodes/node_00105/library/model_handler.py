import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from library.config import MODEL_PARAMS


def train_model(X_train, y_train, X_val, y_val):
    """
    Trains a High-Capacity LightGBM Regressor using parameters from config.
    Implements Early Stopping based on Validation MAE.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.

    Returns:
        model: Trained LightGBM model.
    """
    # Copy parameters to avoid modifying the global config
    params = MODEL_PARAMS.LGBM_PARAMS.copy()

    # Extract early_stopping_rounds to use in callbacks (standard for LightGBM >= 4.0)
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)

    # Initialize the model
    # Note: 'objective' and 'metric' are already in params
    model = lgb.LGBMRegressor(**params)

    # Define callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    print("Starting LightGBM training...")

    # Train the model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric="mae",
        callbacks=callbacks,
    )

    # Generate validation predictions to print final metric
    # LightGBM automatically uses the best iteration for prediction if early stopping was used
    val_preds = model.predict(X_val)
    final_mae = mean_absolute_error(y_val, val_preds)

    print(f"Training finished.")
    print(f"Final Validation MAE: {final_mae}")

    return model


def predict_model(model, X):
    """
    Generates predictions using the trained model.

    Args:
        model: Trained LightGBM model.
        X (pd.DataFrame): Features to predict on.

    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(X)
