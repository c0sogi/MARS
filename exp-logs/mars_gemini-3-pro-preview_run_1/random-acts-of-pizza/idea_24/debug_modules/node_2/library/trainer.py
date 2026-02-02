import pandas as pd
import numpy as np
import library.config as config
from library.config import ID_COL, TARGET_COL, RANDOM_SEED
from library.utils import seed_everything, save_submission
from library.data_loader import load_data
from library.model_rf import RFModel
from library.model_mlp import MLPModel


def train_rf_stream(train_df, val_df, load_cached_data=True):
    """
    Instantiates and trains the Random Forest model stream.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        load_cached_data (bool): Whether to use cached features.

    Returns:
        tuple: (trained_model, validation_auc)
    """
    print("Initializing RF Stream...")
    model = RFModel()
    auc = model.train(train_df, val_df, load_cached_data=load_cached_data)
    print(f"RF Stream Validation AUC: {auc}")
    return model, auc


def train_mlp_stream(train_df, val_df, load_cached_data=True):
    """
    Instantiates and trains the MLP model stream.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        load_cached_data (bool): Whether to use cached features.

    Returns:
        tuple: (trained_model, validation_auc)
    """
    print("Initializing MLP Stream...")
    model = MLPModel()
    auc = model.train(train_df, val_df, load_cached_data=load_cached_data)
    print(f"MLP Stream Validation AUC: {auc}")
    return model, auc


def generate_predictions(rf_model, mlp_model, test_df, load_cached_data=True):
    """
    Generates predictions from both models, ensembles them via averaging,
    and formats the output DataFrame.

    Args:
        rf_model: Trained RFModel instance.
        mlp_model: Trained MLPModel instance.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to use cached features for inference.

    Returns:
        pd.DataFrame: Submission DataFrame with request_id and probabilities.
    """
    print("Generating predictions from RF model...")
    rf_preds = rf_model.predict_proba(test_df, load_cached_data=load_cached_data)

    print("Generating predictions from MLP model...")
    mlp_preds = mlp_model.predict_proba(test_df, load_cached_data=load_cached_data)

    # Ensemble: Simple Weighted Average (0.5 / 0.5)
    final_preds = (rf_preds + mlp_preds) / 2.0

    submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: final_preds})

    return submission_df


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the full training and inference pipeline:
    1. Loads data.
    2. Trains RF Stream.
    3. Trains MLP Stream.
    4. Generates Ensemble Predictions.
    5. Saves Submission.
    """
    # Ensure reproducibility
    seed_everything(RANDOM_SEED)

    # 1. Load Data
    print("Loading data...")
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # 2. Train RF Stream
    rf_model, rf_auc = train_rf_stream(
        train_df, val_df, load_cached_data=load_cached_data
    )

    # 3. Train MLP Stream
    mlp_model, mlp_auc = train_mlp_stream(
        train_df, val_df, load_cached_data=load_cached_data
    )

    # 4. Generate Predictions
    print("Generating ensemble predictions...")
    submission_df = generate_predictions(
        rf_model, mlp_model, test_df, load_cached_data=load_cached_data
    )

    # 5. Save Submission
    save_submission(submission_df, config.SUBMISSION_PATH)

    print("Pipeline completed successfully.")
    print(f"Final RF AUC: {rf_auc}")
    print(f"Final MLP AUC: {mlp_auc}")
