import torch
import torch.optim as optim
from library.config import Config
from library.model import TransUNet1D
from library.data_loader import get_dataloaders
from library.trainer import Trainer, generate_submission
from library.utils import set_seed


def run_inference(debug: bool = False):
    """
    Orchestrates the inference pipeline for the test dataset.

    Args:
        debug (bool): If True, runs in debug mode using a subset of data
                      (controlled by Config.DEBUG_SAMPLE_SIZE).
    """
    # Override Config.DEBUG if specified via argument
    if debug:
        Config.DEBUG = True

    # 1. Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Load data
    # get_dataloaders returns (train_loader, val_loader, test_loader).
    # We only need the test_loader for inference.
    # load_cached_data=True ensures we use preprocessed parquet files if they exist.
    print("Loading dataloaders...")
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    # The model architecture is defined in library.model
    model = TransUNet1D().to(Config.DEVICE)

    # 4. Initialize Trainer
    # The Trainer class requires an optimizer to be initialized, even though
    # we won't be performing any optimization steps during inference.
    # We use the hyperparameters from Config.
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize the Trainer wrapper
    trainer = Trainer(model, Config.DEVICE, optimizer)

    # 5. Generate Submission
    # This function (from library.trainer) performs the following:
    # - Loads the best model weights from Config.WORKING_DIR/model_weights.pth
    # - Runs the model on the test_loader
    # - Converts predicted ENU residuals back to Latitude/Longitude
    # - Merges predictions with the sample submission format
    # - Saves the result to Config.SUBMISSION_DIR/submission.csv
    generate_submission(trainer, test_loader)
