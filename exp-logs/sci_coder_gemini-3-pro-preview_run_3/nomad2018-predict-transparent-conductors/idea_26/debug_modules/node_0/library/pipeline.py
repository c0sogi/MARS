import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from library.config import Config
from library.features import process_data
from library.model import train_model, predict_model, save_submission


def run_training_pipeline(load_cached_data: bool = True):
    """
    Orchestrates the training pipeline: loads data, generates features, trains models,
    and evaluates performance.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.

    Returns:
        dict: Dictionary containing trained models and preprocessors.
    """
    print("--- Starting Training Pipeline ---")

    # 1. Feature Generation / Loading
    # The process_data function handles caching logic (check existence, compute, save)
    print("Processing Training Data...")
    df_train = process_data(split="train", load_cached_data=load_cached_data)

    print("Processing Validation Data...")
    df_val = process_data(split="val", load_cached_data=load_cached_data)

    # 2. Model Training
    print("Training Models...")
    # train_model handles feature cleaning and target transformation internally
    # It returns a dictionary with 'models', 'cleaner', and 'target_transformer'
    artifacts = train_model(train_df=df_train, val_df=df_val)

    # 3. Evaluation (Explicit RMSLE Calculation)
    print("--- Validation Metrics ---")
    models = artifacts["models"]
    cleaner = artifacts["cleaner"]
    target_transformer = artifacts["target_transformer"]

    # Prepare validation features (apply the same cleaning as training data)
    exclude_cols = ["id"] + Config.TARGET_COLS
    feature_cols = [c for c in df_val.columns if c not in exclude_cols]
    X_val = df_val[feature_cols]
    X_val_clean = cleaner.transform(X_val)

    rmsle_scores = []

    for target_col in Config.TARGET_COLS:
        if target_col in models:
            model = models[target_col]

            # Actual targets (original scale)
            y_true = df_val[target_col].values

            # Transform actuals to log scale for RMSLE calculation
            # RMSLE = RMSE(log(1+y_pred), log(1+y_true))
            # The model predicts log(1+y), so we compare against log(1+y_true)
            y_true_log = target_transformer.transform(y_true)

            # Predict (model output is already log-transformed)
            y_pred_log = model.predict(X_val_clean)

            # Calculate RMSE on log scale which is equivalent to RMSLE on original scale
            rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
            rmsle_scores.append(rmsle)

            # Print full precision as requested
            print(f"Target: {target_col}, RMSLE: {rmsle}")

    if rmsle_scores:
        mean_rmsle = np.mean(rmsle_scores)
        print(f"Mean Column-wise RMSLE: {mean_rmsle}")

    return artifacts


def run_inference_pipeline(models_dict: dict, load_cached_data: bool = True):
    """
    Orchestrates the inference pipeline: loads test data, generates features,
    predicts targets, and saves submission.

    Args:
        models_dict (dict): Dictionary containing trained models and preprocessors.
        load_cached_data (bool): Whether to load features from cache if available.
    """
    print("--- Starting Inference Pipeline ---")

    # 1. Feature Generation / Loading for Test Set
    print("Processing Test Data...")
    df_test = process_data(split="test", load_cached_data=load_cached_data)

    # 2. Prediction
    # predict_model handles feature cleaning and inverse transformation of predictions
    print("Generating Predictions...")
    submission_df = predict_model(models_dict=models_dict, test_df=df_test)

    # 3. Save Submission
    save_submission(submission_df)
    print("Inference Pipeline Completed.")
