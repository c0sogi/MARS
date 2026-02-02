import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from library import config, utils


def train_single_regressor(df_train, df_val, type_name, params):
    """
    Trains a single XGBoost regressor for a specific coupling type.

    Args:
        df_train (pd.DataFrame): Training data for this specific type.
        df_val (pd.DataFrame): Validation data for this specific type.
        type_name (str): The coupling type (e.g., '1JHC').
        params (dict): XGBoost hyperparameters.
    """
    print(f"\n{'='*40}")
    print(f"Training Model for Type: {type_name}")
    print(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}")
    print(f"{'='*40}")

    # Identify feature columns
    # Exclude metadata and target columns
    exclude_cols = {
        "id",
        "molecule_name",
        "type",
        "scalar_coupling_constant",
        "file_path",
    }
    features = [c for c in df_train.columns if c not in exclude_cols]

    print(f"Number of features: {len(features)}")

    # Prepare Data
    X_train = df_train[features]
    y_train = df_train["scalar_coupling_constant"]
    X_val = df_val[features]
    y_val = df_val["scalar_coupling_constant"]

    # Create DMatrix
    # enable_categorical=True is not strictly necessary if we handled cats,
    # but good practice if object types remain.
    # Here we assume data is numeric/float32 from data_loader.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose_eval=config.VERBOSE_EVAL,
    )

    # Save Model
    utils.save_model(model, type_name)

    # Cleanup to free GPU memory
    del dtrain, dval, model, X_train, X_val, y_train, y_val
    gc.collect()


class StratifiedEnsemble:
    """
    Manages a collection of XGBoost models, one for each scalar coupling type.
    """

    def __init__(self):
        self.types = list(config.TYPE_SPECIFIC_PARAMS.keys())

    def fit(self, train_df, val_df):
        """
        Trains separate models for each coupling type found in the configuration.

        Args:
            train_df (pd.DataFrame): The complete training dataset.
            val_df (pd.DataFrame): The complete validation dataset.
        """
        for type_name in self.types:
            # Get parameters for this type
            params = config.TYPE_SPECIFIC_PARAMS[type_name]

            # Filter data for this type
            train_subset = train_df[train_df["type"] == type_name]
            val_subset = val_df[val_df["type"] == type_name]

            if len(train_subset) == 0:
                print(
                    f"Warning: No training data found for type {type_name}. Skipping."
                )
                continue

            # Train the model
            train_single_regressor(train_subset, val_subset, type_name, params)

    def predict(self, test_df):
        """
        Generates predictions for the test set using the stratified models.

        Args:
            test_df (pd.DataFrame): The test dataset containing features and 'type'.

        Returns:
            pd.Series: Predictions aligned with the input test_df index.
        """
        print("\nStarting Inference...")

        # Initialize results series with index matching test_df
        final_predictions = pd.Series(index=test_df.index, dtype=np.float32)
        final_predictions[:] = np.nan

        # Identify feature columns (same logic as training)
        exclude_cols = {
            "id",
            "molecule_name",
            "type",
            "scalar_coupling_constant",
            "file_path",
        }
        features = [c for c in test_df.columns if c not in exclude_cols]

        for type_name in self.types:
            # Identify rows belonging to this type
            mask = test_df["type"] == type_name

            if not mask.any():
                continue

            print(f"Predicting for {type_name}: {mask.sum()} samples")

            # Load model
            try:
                model = utils.load_model(type_name)
            except FileNotFoundError:
                print(
                    f"Warning: Model for {type_name} not found. Skipping predictions for this type."
                )
                continue

            # Prepare data
            X_test = test_df.loc[mask, features]
            dtest = xgb.DMatrix(X_test)

            # Predict
            preds = model.predict(dtest)

            # Assign predictions to the correct rows
            final_predictions.loc[mask] = preds

            # Cleanup
            del model, dtest, X_test
            gc.collect()

        return final_predictions
