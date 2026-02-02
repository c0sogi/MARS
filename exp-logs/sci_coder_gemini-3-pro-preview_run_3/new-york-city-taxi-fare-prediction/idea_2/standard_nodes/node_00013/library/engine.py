import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data_loader import create_dataloaders
from library.gbdt_model import GBDTPredictor
from library.nn_model import NNPredictor


def train_gbdt(train_df, val_df):
    """
    Trains the Gradient Boosting Decision Tree model.

    Args:
        train_df (pd.DataFrame): The training dataset.
        val_df (pd.DataFrame): The validation dataset.

    Returns:
        GBDTPredictor: The trained GBDT model instance.
    """
    set_seed()
    print("Initializing GBDT training...")

    # Initialize the predictor with default config parameters
    gbdt_model = GBDTPredictor()

    # Fit the model
    # The GBDTPredictor handles feature selection and internal validation
    gbdt_model.fit(train_df, val_df)

    # Save the model artifact
    gbdt_model.save(Config.GBDT_MODEL_PATH)

    return gbdt_model


def train_nn(train_df, val_df, test_df):
    """
    Trains the Neural Network model.

    This function handles:
    1. Creation of PyTorch DataLoaders and Scalers.
    2. Initialization of the NNPredictor.
    3. Execution of the training loop via the predictor.
    4. Persistence of the model.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data (needed for scaling consistency).

    Returns:
        tuple: (trained_nn_model, test_loader)
    """
    set_seed()
    print("Initializing Neural Network training...")

    # Create DataLoaders
    # This handles scaling of continuous features and batching
    # We pass load_cached_scaler=True to utilize existing scalers if available
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(
        train_df, val_df, test_df, load_cached_scaler=True
    )

    # Initialize the Neural Network Predictor
    nn_model = NNPredictor()

    # Fit the model (handles training loop, backprop, early stopping)
    nn_model.fit(train_loader, val_loader)

    # Save the model artifact
    nn_model.save(Config.NN_MODEL_PATH)

    return nn_model, test_loader


def generate_predictions(gbdt_model, nn_model, test_df, test_loader):
    """
    Generates predictions using the ensemble of GBDT and NN models.
    Saves the result to the submission file.

    Args:
        gbdt_model (GBDTPredictor): Trained GBDT model.
        nn_model (NNPredictor): Trained NN model (Can be None).
        test_df (pd.DataFrame): Raw test dataframe (for GBDT and IDs).
        test_loader (DataLoader): Test DataLoader (for NN, Can be None).

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    set_seed()
    print("Generating predictions...")

    # 1. GBDT Predictions
    # GBDT uses the dataframe directly
    print("Predicting with GBDT...")
    gbdt_preds = gbdt_model.predict(test_df)

    # 2. NN Predictions (Optional)
    if nn_model is not None and Config.WEIGHT_NN > 0:
        print("Predicting with Neural Network...")
        nn_preds = nn_model.predict(test_loader)

        # 3. Ensemble (Weighted Average)
        print(
            f"Ensembling with weights: GBDT={Config.WEIGHT_GBDT}, NN={Config.WEIGHT_NN}"
        )
        final_preds = (Config.WEIGHT_GBDT * gbdt_preds) + (Config.WEIGHT_NN * nn_preds)
    else:
        print("Using GBDT predictions only.")
        final_preds = gbdt_preds

    # 4. Format Submission
    submission = pd.DataFrame(
        {"key": test_df[Config.ID_COL], "fare_amount": final_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission
