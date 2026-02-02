import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.preprocessing import TargetTransformer, FeatureCleaner


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """
    Trains XGBoost models for each target variable defined in Config.
    Applies feature cleaning and target transformation.

    Args:
        train_df: DataFrame containing training features and targets.
        val_df: DataFrame containing validation features and targets.

    Returns:
        dict: A dictionary containing trained models and the fitted feature cleaner.
              Structure: {'models': {target: model}, 'cleaner': feature_cleaner, 'target_transformer': transformer}
    """
    # Identify feature columns (exclude id and targets)
    exclude_cols = ["id"] + Config.TARGET_COLS
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    # Initialize and fit FeatureCleaner
    # This imputes missing values and removes constant features to prevent feature dilution
    print("Fitting FeatureCleaner on training data...")
    cleaner = FeatureCleaner(constant_threshold=0.0)
    X_train_clean = cleaner.fit_transform(X_train)
    X_val_clean = cleaner.transform(X_val)
    print(f"Feature count reduced from {X_train.shape[1]} to {X_train_clean.shape[1]}")

    # Initialize TargetTransformer for log(1+y) transformation
    target_transformer = TargetTransformer()

    trained_models = {}

    for target_col in Config.TARGET_COLS:
        print(f"\nTraining model for target: {target_col}")

        # Extract targets
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        # Transform targets (log1p)
        y_train_trans = target_transformer.transform(y_train)
        y_val_trans = target_transformer.transform(y_val)

        # Initialize XGBoost Regressor with config parameters
        # Cite debug_lesson_1: Update XGBoost Early Stopping Syntax for Versions 1.6+
        xgb_params = Config.XGB_PARAMS.copy()
        xgb_params["early_stopping_rounds"] = Config.EARLY_STOPPING_ROUNDS
        model = xgb.XGBRegressor(**xgb_params)

        # Fit model with early stopping
        # Note: early_stopping_rounds is passed to fit() for sklearn-compatible API
        model.fit(
            X_train_clean,
            y_train_trans,
            eval_set=[(X_val_clean, y_val_trans)],
            verbose=Config.VERBOSE_EVAL,
        )

        # Predict on validation set to print metrics
        preds_trans = model.predict(X_val_clean)

        # Calculate RMSE on transformed data
        rmse_trans = np.sqrt(mean_squared_error(y_val_trans, preds_trans))
        print(f"Validation RMSE (Log-Transformed): {rmse_trans}")

        # Calculate RMSE on original scale for interpretability
        preds_orig = target_transformer.inverse_transform(preds_trans)
        rmse_orig = np.sqrt(mean_squared_error(y_val, preds_orig))
        print(f"Validation RMSE (Original Scale): {rmse_orig}")

        trained_models[target_col] = model

    return {
        "models": trained_models,
        "cleaner": cleaner,
        "target_transformer": target_transformer,
    }


def predict_model(models_dict: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictions for the test set using trained models.

    Args:
        models_dict: Dictionary returned by train_model containing models and preprocessors.
        test_df: DataFrame containing test features (and id).

    Returns:
        pd.DataFrame: DataFrame containing 'id' and predicted target columns.
    """
    cleaner = models_dict["cleaner"]
    target_transformer = models_dict["target_transformer"]
    models = models_dict["models"]

    # Identify feature columns in test set
    # We assume the test set has the same feature columns as train set (excluding id)
    exclude_cols = ["id"]
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]

    X_test = test_df[feature_cols]

    # Apply the fitted cleaner (impute and select features)
    # This ensures the test set has the exact same columns as the model expects
    X_test_clean = cleaner.transform(X_test)

    # Initialize submission DataFrame
    submission_df = pd.DataFrame()
    submission_df["id"] = test_df["id"]

    for target_col in Config.TARGET_COLS:
        if target_col in models:
            model = models[target_col]

            # Generate predictions in log-transformed space
            preds_trans = model.predict(X_test_clean)

            # Inverse transform to original space (expm1)
            preds_orig = target_transformer.inverse_transform(preds_trans)

            submission_df[target_col] = preds_orig

    return submission_df


def save_submission(
    submission_df: pd.DataFrame, output_path: str = Config.SUBMISSION_PATH
):
    """
    Saves the submission DataFrame to a CSV file.

    Args:
        submission_df: DataFrame containing predictions.
        output_path: Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
