import os
import pandas as pd
from library.config import Config
from library.trainer import Trainer


def generate_submission(debug: bool = Config.DEBUG):
    """
    Generates the submission file for the iNaturalist 2019 competition.

    This function:
    1. Initializes the Trainer (which sets up the model architecture and device).
    2. Loads the test data loader.
    3. Calls the trainer's predict method to run inference, format predictions,
       and save the result to the configured submission path.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data
                      as defined in Config.DEBUG_SAMPLE_SIZE. Defaults to Config.DEBUG.
    """
    # Initialize the Trainer.
    # This sets the random seed, creates the model architecture, and moves it to the device.
    # It also initializes the optimizer/scheduler, which are unused for inference but harmless.
    trainer = Trainer()

    # Retrieve DataLoaders.
    # get_dataloaders returns a tuple: (train_loader, val_loader, test_loader).
    # We only need the test_loader for inference.
    print(f"Retrieving test dataloader (debug={debug})...")
    _, _, test_loader = trainer.get_dataloaders(debug=debug)

    # Run prediction.
    # The predict method handles:
    # - Loading the best model checkpoint from Config.MODEL_CHECKPOINT.
    # - Running the forward pass in mixed precision (if enabled).
    # - Extracting top-5 class indices.
    # - Formatting the output string.
    # - Saving the DataFrame to Config.SUBMISSION_FILE.
    print("Starting inference...")
    trainer.predict(test_loader)

    print("Inference complete.")
