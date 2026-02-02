import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class XGBoostRegressorWrapper:
    """
    Wrapper for XGBoost Regressor with feature pruning and specific hyperparameters.
    Encapsulates the model definition and training process.
    """

    def __init__(
        self,
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        early_stopping_rounds=50,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.valid_features = None

    def fit_model(self, X_train, y_train, X_val, y_val):
        """
        Fits the XGBoost model with feature pruning (removing constant features)
        and early stopping.

        Args:
            X_train, X_val: Feature matrices (pandas DataFrame or numpy array)
            y_train, y_val: Target vectors (log-transformed)
        """
        # Ensure inputs are DataFrames for reliable column handling
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            # Attempt to align columns if X_val is array
            X_val = pd.DataFrame(X_val)
            if X_val.shape[1] == X_train.shape[1]:
                X_val.columns = X_train.columns

        # Feature Pruning: Identify constant features (zero variance)
        # We calculate variance on the training set to avoid data leakage
        variances = X_train.var()
        # Keep features with variance > 0
        self.valid_features = variances[variances > 0].index.tolist()

        # Apply pruning to both sets
        X_train_pruned = X_train[self.valid_features]
        X_val_pruned = X_val[self.valid_features]

        # Initialize XGBoost model with specified hyperparameters
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=-1,
            random_state=self.random_state,
            objective="reg:squarederror",
            tree_method="hist",  # Use histogram-based algorithm for efficiency
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=0,
        )

        # Fit model with early stopping
        self.model.fit(
            X_train_pruned,
            y_train,
            eval_set=[(X_train_pruned, y_train), (X_val_pruned, y_val)],
            verbose=False,
        )

        # Calculate and print evaluation metrics (on log-transformed data)
        train_preds = self.model.predict(X_train_pruned)
        val_preds = self.model.predict(X_val_pruned)

        train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))

        print(f"Training RMSE (log-space): {train_rmse}")
        print(f"Validation RMSE (log-space): {val_rmse}")

    def predict_model(self, X):
        """
        Generates predictions using the trained model.

        Args:
            X: Feature matrix (pandas DataFrame)

        Returns:
            np.array: Predicted values (log-space)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Select only the valid features identified during training
        try:
            X_pruned = X[self.valid_features]
        except KeyError as e:
            raise KeyError(f"Input data is missing features required by the model: {e}")

        return self.model.predict(X_pruned)


def train_models(X_train, y_train, X_val, y_val, random_state=42):
    """
    Orchestrates the training of separate XGBoost models for each target variable.

    Args:
        X_train, X_val: Feature DataFrames
        y_train, y_val: Target DataFrames (containing both targets, log-transformed)
        random_state: Seed for reproducibility

    Returns:
        dict: Dictionary of trained XGBoostRegressorWrapper objects, keyed by target name.
    """
    targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    models = {}

    for target in targets:
        print(f"Training model for target: {target}")

        # Instantiate the wrapper
        model = XGBoostRegressorWrapper(random_state=random_state)

        # Extract the specific target series
        y_tr = y_train[target]
        y_v = y_val[target]

        # Train the model
        model.fit_model(X_train, y_tr, X_val, y_v)

        models[target] = model

    return models


def predict_models(models, X_test):
    """
    Generates final predictions for all targets using the trained models.
    Applies the inverse log transformation (expm1) to return values to the original scale.

    Args:
        models (dict): Dictionary of trained models.
        X_test (pd.DataFrame): Test features.

    Returns:
        pd.DataFrame: DataFrame containing predictions for each target.
    """
    predictions = {}

    for target, model in models.items():
        # Predict in log space
        log_preds = model.predict_model(X_test)
        # Apply inverse transformation (exp(x) - 1)
        predictions[target] = np.expm1(log_preds)

    return pd.DataFrame(predictions)
