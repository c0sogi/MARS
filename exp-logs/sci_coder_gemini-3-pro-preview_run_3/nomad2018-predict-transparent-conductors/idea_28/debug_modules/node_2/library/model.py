import numpy as np
import pandas as pd
import xgboost as xgb
import os
from library.config import (
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    TARGET_COLS,
    TEST_METADATA_PATH,
)
from library.data_pipeline import generate_features
from library.utils import calculate_rmsle, save_submission, setup_logger

logger = setup_logger("model")


class DualTargetRegressor:
    """
    Wrapper for training separate XGBoost regressors for each target.
    Handles log-transformation of targets internally to ensure non-negative predictions
    and better handle the distribution of energy values.
    """

    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS.copy()
        self.models = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fits the models. Targets are log-transformed (log1p) before training.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets.
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame, optional): Validation targets.
        """
        # Log-transform targets: z = log(1 + y)
        y_train_log = np.log1p(y_train)

        # Prepare validation set if provided
        y_val_log = None
        if y_val is not None:
            y_val_log = np.log1p(y_val)

        for target in TARGET_COLS:
            logger.info(f"Training model for target: {target}")

            # Cite debug_lesson_1: Update XGBoost Early Stopping Syntax
            # Cite debug_lesson_15: Disable XGBoost Early Stopping When Fitting Without Validation Sets
            current_params = self.params.copy()
            eval_set = []

            if X_val is not None and y_val_log is not None:
                eval_set = [(X_val, y_val_log[target])]
                current_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            else:
                # Ensure early_stopping_rounds is not set if no eval_set
                current_params.pop("early_stopping_rounds", None)

            # Instantiate model with parameters for this run
            model = xgb.XGBRegressor(**current_params)

            model.fit(
                X_train,
                y_train_log[target],
                eval_set=eval_set if eval_set else None,
                verbose=VERBOSE_EVAL,
            )

            self.models[target] = model

            # Evaluate on validation set in original scale for logging purposes
            if X_val is not None and y_val is not None:
                # Predict log values
                pred_log = model.predict(X_val)
                # Inverse transform: y = exp(z) - 1
                pred = np.expm1(pred_log)
                # Enforce physical constraint (non-negative energy)
                pred = np.maximum(0, pred)

                score = calculate_rmsle(y_val[target], pred)
                logger.info(f"Validation RMSLE for {target}: {score}")

    def predict(self, X):
        """
        Predicts targets. Output is inverse-transformed (expm1) to original scale.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            pd.DataFrame: DataFrame containing predictions for all targets.
        """
        predictions = {}
        for target in TARGET_COLS:
            model = self.models[target]
            pred_log = model.predict(X)
            pred = np.expm1(pred_log)
            pred = np.maximum(0, pred)  # Ensure non-negative
            predictions[target] = pred

        return pd.DataFrame(predictions)


def train_model(load_cached_data=True, debug=False, n_estimators=None):
    """
    Loads data, trains the dual regressor, and evaluates it.

    Args:
        load_cached_data (bool): Whether to load features from parquet cache.
        debug (bool): If True, subsamples data for quick debugging.
        n_estimators (int): Override default n_estimators for XGBoost.

    Returns:
        DualTargetRegressor: The trained model instance.
    """
    logger.info("Loading training data...")
    X_train, y_train = generate_features(
        data_type="train", load_cached_data=load_cached_data
    )

    logger.info("Loading validation data...")
    X_val, y_val = generate_features(data_type="val", load_cached_data=load_cached_data)

    if debug:
        logger.info("Debug mode: Subsampling data...")
        # Use a small subset for debugging
        X_train = X_train.iloc[:100]
        y_train = y_train.iloc[:100]
        X_val = X_val.iloc[:20]
        y_val = y_val.iloc[:20]

    # Configure parameters
    params = XGB_PARAMS.copy()
    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    # Initialize and train
    model = DualTargetRegressor(params=params)
    model.fit(X_train, y_train, X_val, y_val)

    # Final evaluation on the full validation set (even if debug was True for training,
    # though usually debug implies we don't care about full val score,
    # but here we use the loaded X_val which was sliced if debug=True)
    logger.info("Evaluating model performance...")
    val_preds = model.predict(X_val)
    overall_score = calculate_rmsle(y_val, val_preds)
    logger.info(f"Overall Validation RMSLE: {overall_score}")

    return model


def generate_submission_file(model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (DualTargetRegressor): Trained model.
        load_cached_data (bool): Whether to load test features from cache.
    """
    logger.info("Loading test data...")
    X_test, _ = generate_features(data_type="test", load_cached_data=load_cached_data)

    # Load IDs from metadata since X_test doesn't have them (dropped in pipeline)
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    test_meta = pd.read_csv(TEST_METADATA_PATH)
    test_ids = test_meta["id"].values

    # Ensure alignment
    if len(test_ids) != len(X_test):
        raise ValueError(
            f"Mismatch between test metadata IDs ({len(test_ids)}) and test features rows ({len(X_test)})."
        )

    logger.info("Generating predictions for test set...")
    preds_df = model.predict(X_test)

    save_submission(
        ids=test_ids,
        formation_energy=preds_df["formation_energy_ev_natom"],
        bandgap_energy=preds_df["bandgap_energy_ev"],
    )
