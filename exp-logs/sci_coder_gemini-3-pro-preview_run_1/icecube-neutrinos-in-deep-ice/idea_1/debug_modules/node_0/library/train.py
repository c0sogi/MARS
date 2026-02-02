import torch
from library.config import Config, set_seed
from library.data import get_dataloaders, get_test_dataloader
from library.model import TemporalCNN, train_model, predict_and_submit


class Trainer:
    """
    Manages the training and submission pipeline for the Neutrino Direction Prediction task.
    Encapsulates data loading, model initialization, training, and inference.
    """

    def __init__(self):
        """
        Initializes the Trainer and sets the random seed for reproducibility.
        """
        set_seed(Config.SEED)
        self.device = Config.DEVICE

    def run(self, num_epochs=None, debug=None, subset_size=None):
        """
        Executes the full pipeline: Data Loading -> Training -> Prediction -> Submission.

        Args:
            num_epochs (int, optional): Override for the number of training epochs.
            debug (bool, optional): Override for the debug flag (train on subset).
            subset_size (int, optional): Override for the debug subset size.
        """
        # 1. Update Configuration based on arguments for flexibility
        if num_epochs is not None:
            Config.NUM_EPOCHS = num_epochs
        if debug is not None:
            Config.DEBUG = debug
        if subset_size is not None:
            Config.DEBUG_SUBSET_SIZE = subset_size

        print(
            f"Trainer Configured: Epochs={Config.NUM_EPOCHS}, Debug={Config.DEBUG}, Device={self.device}"
        )

        # 2. Load Training and Validation Data
        # get_dataloaders handles metadata reading and caching internally
        print("Loading training data...")
        train_loader, val_loader = get_dataloaders()

        # 3. Initialize Model
        print("Initializing TemporalCNN model...")
        model = TemporalCNN()

        # 4. Train Model
        # train_model manages the optimizer, loss calculation, backprop,
        # validation metrics, early stopping, and saving the best model.
        print("Starting training process...")
        best_model = train_model(model, train_loader, val_loader, self.device)

        # 5. Generate Submission
        # Load test data
        print("Loading test data...")
        test_loader = get_test_dataloader()

        # Generate predictions and save to submission.csv
        print("Generating predictions and saving submission...")
        predict_and_submit(best_model, test_loader, self.device)
