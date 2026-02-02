import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import library.config as config
from library.data_loader import prepare_features_target


def train_regressor(
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators=None,
    max_samples=None,
    random_state=config.SEED,
):
    """
    Trains a LightGBM Regressor with early stopping and optional subsampling.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        n_estimators (int, optional): Override for the number of boosting iterations.
        max_samples (int, optional): If provided and less than the training set size,
                                     subsamples the training data for faster execution.
        random_state (int): Random seed for reproducibility.

    Returns:
        model: The trained LightGBM model.
        float: The Mean Absolute Error (MAE) on the validation set.
    """
    # Set global seed for numpy to ensure reproducibility in sampling
    np.random.seed(random_state)

    # Apply subsampling if requested for debugging or faster iteration
    if max_samples is not None and max_samples < len(X_train):
        X_train = X_train.sample(n=max_samples, random_state=random_state)
        y_train = y_train.loc[X_train.index]

    # Prepare model parameters
    params = config.LGBM_PARAMS.copy()
    params["random_state"] = random_state

    # Allow overriding n_estimators (training steps)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    # Initialize the LightGBM Regressor
    model = lgb.LGBMRegressor(**params)

    # Configure callbacks for early stopping and logging control
    # log_evaluation(period=0) suppresses verbose output during training
    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    # Fit the model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=callbacks,
    )

    # Generate predictions on the validation set
    val_preds = model.predict(X_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, val_preds)

    # Print validation metric with full precision
    print(f"Validation MAE: {mae}")

    return model, mae


def predict(model, X):
    """
    Generates predictions using the trained model.

    Args:
        model: Trained LightGBM model.
        X (pd.DataFrame): Feature matrix.

    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(X)


def generate_submission(model, test_df, output_path=config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        model: Trained LightGBM model.
        test_df (pd.DataFrame): The processed test dataset containing 'segment_id' and features.
        output_path (str): File path to save the submission CSV.
    """
    # Isolate features for prediction using the data_loader helper
    # This ensures 'segment_id' is removed from the feature set
    X_test, _ = prepare_features_target(
        test_df, target_col="time_to_eruption", id_col="segment_id"
    )

    # Generate predictions
    predictions = predict(model, X_test)

    # Construct the submission DataFrame
    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": predictions}
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
