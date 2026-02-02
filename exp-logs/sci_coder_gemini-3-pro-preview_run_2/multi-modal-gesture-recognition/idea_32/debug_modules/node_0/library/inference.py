import os
import torch
from torch.utils.data import DataLoader
import library.config as config
import library.utils as utils
import library.model as model_lib
import library.data_loader as data_loader
import library.trainer as trainer_lib


class InferenceEngine:
    """
    Engine to handle inference process using the trained BMGCN model.
    """

    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the InferenceEngine.

        Args:
            checkpoint_path (str): Path to the model checkpoint.
            device (torch.device, optional): Device to run inference on.
        """
        self.device = device if device else utils.get_device()
        self.checkpoint_path = checkpoint_path

        # Initialize Model Architecture
        self.model = model_lib.BMGCN()

        # Load weights
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        utils.load_checkpoint(self.checkpoint_path, self.model, device=self.device)

        # Initialize Trainer to leverage its prediction logic
        self.trainer = trainer_lib.Trainer(self.model, self.device)

    def run(
        self,
        test_metadata_path=config.TEST_METADATA_PATH,
        output_path=os.path.join(config.SUBMISSION_DIR, "submission.csv"),
        batch_size=config.TRAIN_CONFIG["batch_size"],
        num_workers=config.TRAIN_CONFIG["num_workers"],
    ):
        """
        Runs inference on the test dataset and generates the submission file.

        Args:
            test_metadata_path (str): Path to test metadata CSV.
            output_path (str): Path to save the submission CSV.
            batch_size (int): Batch size for inference.
            num_workers (int): Number of workers for data loading.
        """
        # Create Test Dataset and Loader directly to avoid overhead of loading train/val
        # We use the existing GestureDataset class from data_loader
        test_ds = data_loader.GestureDataset(
            metadata_path=test_metadata_path,
            mode="test",
            load_cached_data=True,
            limit=None,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=True,
        )

        print(f"Running inference on {len(test_ds)} samples...")

        # Use the trainer's method to generate the submission file
        # This handles prediction, median filtering, decoding, and file writing
        self.trainer.generate_submission_file(test_loader, output_path)
