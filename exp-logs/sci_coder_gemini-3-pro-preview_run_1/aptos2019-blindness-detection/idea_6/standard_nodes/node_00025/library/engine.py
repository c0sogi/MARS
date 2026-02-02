import os
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import run_training, generate_submission


def run_experiment(debug=False, epochs=Config.EPOCHS):
    """
    Main function to execute the training and inference pipeline.

    This function orchestrates the following steps:
    1. Sets random seeds for reproducibility.
    2. Initializes DataLoaders (optionally with a subset for debugging).
    3. Runs the training loop (including validation and early stopping).
    4. Generates the submission file using the best saved model.

    Args:
        debug (bool): If True, runs the pipeline on a small subset of the data for debugging purposes.
        epochs (int): The number of epochs to train the model. Defaults to Config.EPOCHS.

    Returns:
        float: The best Quadratic Weighted Kappa (QWK) score achieved on the validation set.
    """
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # Ensure necessary directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Prepare Data
    # Load data loaders. If debug is True, a small subset (Config.DEBUG_SUBSET_SIZE) is used.
    subset_size = Config.DEBUG_SUBSET_SIZE
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, subset_size=subset_size
    )

    # 3. Train Model
    # The run_training function encapsulates the training loop, validation logic,
    # scheduler updates, and saves the model with the best QWK score.
    best_qwk = run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        save_path=Config.BEST_MODEL_PATH,
    )

    # 4. Generate Submission
    # Predicts on the test set using the best model checkpoint and saves to submission.csv.
    generate_submission(
        test_loader=test_loader,
        model_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
    )

    return best_qwk
