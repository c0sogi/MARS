import os
import torch
from library.config import Config
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import MultiScaleResUNet1D
from library.train import Trainer
from library.loss import DeepSupervisionMAELoss


class InferenceRunner:
    """
    Handles the inference pipeline for the GNSS localization model.
    Loads the trained model, prepares test data, and generates the submission file.
    """

    def __init__(self, checkpoint_path=None):
        """
        Initialize the InferenceRunner.

        Args:
            checkpoint_path (str, optional): Path to the model checkpoint.
                                             Defaults to Config.WORKING_DIR/best_model.pth.
        """
        self.device = torch.device(Config.DEVICE)
        self.checkpoint_path = checkpoint_path or os.path.join(
            Config.WORKING_DIR, "best_model.pth"
        )

    def run_inference(self, debug=False):
        """
        Executes the inference process.

        Args:
            debug (bool): If True, runs on a subset of the data for debugging purposes.
        """
        print(f"Starting Inference Run (Debug={debug})...")

        # 1. Get Normalization Statistics
        # We invoke get_dataloaders to process/load the training data and calculate
        # the mean and std used during training. This is crucial for correct inference.
        print("Retrieving normalization statistics from training data...")
        # We don't need the train/val loaders themselves, just the stats
        _, _, stats = get_dataloaders(debug=debug)

        # 2. Prepare Test DataLoader
        print("Initializing Test DataLoader...")
        test_loader = get_test_dataloader(stats, debug=debug)

        # 3. Load Model Architecture and Weights
        print(f"Loading model architecture: {Config.MODEL_NAME}...")
        model = MultiScaleResUNet1D().to(self.device)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.checkpoint_path}"
            )

        print(f"Loading weights from {self.checkpoint_path}...")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)

        # 4. Initialize Trainer for Prediction
        # We use the Trainer class to handle the prediction loop and coordinate conversion.
        # Optimizer and Scheduler are not needed for inference, so we pass None.
        # Criterion is required by __init__ but unused in predict(), so we pass the standard loss.
        criterion = DeepSupervisionMAELoss()

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
        )

        # 5. Generate Predictions
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        print(f"Generating predictions -> {output_path}")
        trainer.predict(test_loader, output_path)

        print("Inference process completed successfully.")
