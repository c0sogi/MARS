import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.gbdt_model import GBDTPredictor


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


def generate_predictions(gbdt_model, test_df):
    """
    Generates predictions using the GBDT model.
    Saves the result to the submission file.

    Args:
        gbdt_model (GBDTPredictor): Trained GBDT model.
        test_df (pd.DataFrame): Raw test dataframe.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    set_seed()
    print("Generating predictions...")

    # 1. GBDT Predictions
    print("Predicting with GBDT...")
    final_preds = gbdt_model.predict(test_df)

    # 2. Format Submission
    submission = pd.DataFrame(
        {"key": test_df[Config.ID_COL], "fare_amount": final_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission
