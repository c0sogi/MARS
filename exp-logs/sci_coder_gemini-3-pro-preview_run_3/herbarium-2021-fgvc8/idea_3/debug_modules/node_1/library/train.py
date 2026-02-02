import os
import pandas as pd
from library.config import Config
from library.dataset import get_dataloaders
from library.model import train_model, generate_submission


def run_training(debug=False, epochs=None, load_cached_data=True):
    """
    Orchestrates the training and submission pipeline for the Plant Species Classification task.

    This function:
    1. Updates configuration based on arguments (e.g., debug mode, epochs).
    2. Loads metadata from CSV files.
    3. Initializes DataLoaders with weighted sampling.
    4. Trains the ConvNeXt model (including SWA and early stopping).
    5. Generates the submission file for the test set.

    Args:
        debug (bool): If True, enables debug mode (uses fewer samples).
        epochs (int, optional): If provided, overrides the number of training epochs in Config.
        load_cached_data (bool): If True, attempts to load cached sampler weights.
                                 If False or cache missing, recomputes and saves weights.
    """
    # 1. Update Configuration
    if debug:
        Config.DEBUG = True
        print(f"Debug mode enabled. Using {Config.DEBUG_SAMPLE_SIZE} samples.")

    if epochs is not None:
        Config.EPOCHS = epochs
        print(f"Overriding total epochs to: {Config.EPOCHS}")

    # 2. Load Metadata
    # We rely on the pre-generated metadata CSVs in ./metadata
    if (
        not os.path.exists(Config.TRAIN_CSV)
        or not os.path.exists(Config.VAL_CSV)
        or not os.path.exists(Config.TEST_CSV)
    ):
        raise FileNotFoundError(
            f"Metadata files not found in {Config.METADATA_DIR}. "
            "Please ensure metadata generation script has been run."
        )

    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Initialize DataLoaders
    # get_dataloaders handles the creation of datasets, transforms, and the weighted sampler.
    # It passes 'load_cached_data' down to the sampler utility.
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 4. Train Model
    # train_model handles the full training loop, including:
    # - Mixed Precision (AMP)
    # - Label Smoothing
    # - Stochastic Weight Averaging (SWA)
    # - Early Stopping
    # - Model Checkpointing
    print("Starting training pipeline...")
    model = train_model(train_loader, val_loader)

    # 5. Generate Submission
    # Generates predictions on the test set and saves to Config.SUBMISSION_PATH
    print("Generating submission...")
    generate_submission(model, test_loader)

    print("Run completed successfully.")
