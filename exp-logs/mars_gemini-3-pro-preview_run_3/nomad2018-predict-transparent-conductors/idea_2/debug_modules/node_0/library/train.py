import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.data import load_data
from library.model import EnergyPredictor


def train_model(load_cached_data=True):
    """
    Orchestrates the training process: loads data, trains the model,
    and evaluates it on the validation set.

    Args:
        load_cached_data (bool): Whether to load features from cache.

    Returns:
        tuple: (trained_model, validation_metrics)
    """
    # 1. Load Data
    # load_data handles feature extraction, GNN inference, and caching internally
    # It returns log-transformed targets if Config.LOG_TRANSFORM_TARGETS is True
    (X_train, y_train), (X_val, y_val), X_test, test_ids = load_data(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = EnergyPredictor()

    # 3. Train Model
    # The fit method in EnergyPredictor handles the loop over targets and XGBoost training
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Evaluate Model
    metrics = evaluate_model(model, X_val, y_val)

    # 5. Generate Submission
    generate_submission(model, X_test, test_ids)

    return model, metrics


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the trained model on the validation set using RMSLE.

    Args:
        model (EnergyPredictor): Trained model instance.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.DataFrame): Validation targets (potentially log-transformed).

    Returns:
        dict: Dictionary containing RMSLE for each target and the mean RMSLE.
    """
    print("\n--- Evaluating Model ---")

    preds_log = model.predict(X_val)

    # Calculate RMSLE
    # Since inputs y_val are already log1p transformed (if Config.LOG_TRANSFORM_TARGETS is True),
    # and the model predicts in that space, RMSE on these values IS the RMSLE.
    # RMSLE = sqrt(mean((log(1+p) - log(1+a))^2))
    # Here p_log = log(1+p) and a_log = log(1+a)
    # So RMSE(p_log, a_log) = RMSLE(p, a)

    metrics = {}
    total_rmsle = 0.0

    for target in Config.TARGET_COLS:
        # Calculate RMSE on the log-scale values
        # This is equivalent to RMSLE on original scale
        mse = mean_squared_error(y_val[target], preds_log[target])
        rmsle = np.sqrt(mse)

        metrics[f"RMSLE_{target}"] = rmsle
        total_rmsle += rmsle

        print(f"Target: {target}")
        print(f"  RMSLE: {rmsle}")

        # Optional: Calculate MAE on original scale for interpretability
        if Config.LOG_TRANSFORM_TARGETS:
            y_true_orig = np.expm1(y_val[target])
            y_pred_orig = np.expm1(preds_log[target])
            mae_orig = np.mean(np.abs(y_true_orig - y_pred_orig))
            print(f"  MAE (Original Scale): {mae_orig}")

    mean_rmsle = total_rmsle / len(Config.TARGET_COLS)
    metrics["Mean_RMSLE"] = mean_rmsle
    print(f"\nOverall Mean RMSLE: {mean_rmsle}")

    return metrics


def generate_submission(model, X_test, test_ids):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (EnergyPredictor): Trained model instance.
        X_test (pd.DataFrame): Test features.
        test_ids (pd.Series): IDs for the test samples.
    """
    print("\n--- Generating Submission ---")

    # Predict (returns log-scale predictions if Config.LOG_TRANSFORM_TARGETS is True)
    preds_log = model.predict(X_test)

    # Inverse transform if necessary
    if Config.LOG_TRANSFORM_TARGETS:
        print("Applying expm1 to inverse transform predictions...")
        preds_final = np.expm1(preds_log)
    else:
        preds_final = preds_log

    # Construct submission DataFrame
    submission = pd.DataFrame()
    submission["id"] = test_ids

    for target in Config.TARGET_COLS:
        submission[target] = preds_final[target].values

    # Save to file
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(submission.head())
