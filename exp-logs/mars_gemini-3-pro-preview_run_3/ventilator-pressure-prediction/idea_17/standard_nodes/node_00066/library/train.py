import os
import sys
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_datasets
from library.model import train_model, predict_and_submit


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    force_recompute=False,
):
    """
    Orchestrates the training pipeline for the Ventilator Pressure Prediction task.

    This function manages the configuration, data preparation, model training,
    and submission generation processes using the provided library components.

    Args:
        epochs (int): The number of training epochs. Defaults to Config.EPOCHS.
        batch_size (int): The batch size for data loaders. Defaults to Config.BATCH_SIZE.
        debug (bool): If True, runs in debug mode with a data subset. Defaults to Config.DEBUG.
        force_recompute (bool): If True, forces feature engineering to run from scratch,
                                ignoring cached files. Defaults to False.
    """
    # 1. Update Configuration
    # We update the Config class attributes directly so that the imported library
    # functions (which reference Config) see the updated values.
    Config.EPOCHS = epochs
    Config.BATCH_SIZE = batch_size
    Config.DEBUG = debug

    # 2. Set Random Seeds
    # Ensures reproducibility of the run.
    seed_everything(Config.SEED)

    print(f"Starting training pipeline...")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Debug={Config.DEBUG}"
    )

    # 3. Prepare Datasets
    # This step handles feature engineering, caching, and DataLoader creation.
    # It returns train, validation, and test loaders.
    train_loader, val_loader, test_loader = prepare_datasets(
        batch_size=Config.BATCH_SIZE, force_recompute=force_recompute
    )

    # 4. Train Model
    # Executes the training loop, validation, learning rate scheduling,
    # and early stopping. Returns the model with the best validation weights.
    model = train_model(train_loader, val_loader)

    # 5. Generate Submission
    # Runs inference on the test set and saves the submission.csv file.
    predict_and_submit(model, test_loader)

    print("Training pipeline completed successfully.")
