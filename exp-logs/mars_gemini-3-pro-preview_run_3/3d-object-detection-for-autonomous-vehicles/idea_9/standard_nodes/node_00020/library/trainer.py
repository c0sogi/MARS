import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import LidarDataset, collate_fn
from library.detector import PointPillarsDetector


class Trainer:
    """
    Manages the training lifecycle, validation, and submission generation
    for the Two-Stage PointPillars object detection model.
    """

    def __init__(self, debug=False, subset_size=None, epochs=None):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, runs in debug mode with smaller subsets.
            subset_size (int): Optional override for subset size.
            epochs (int): Optional override for number of epochs.
        """
        self.debug = debug
        self.subset_size = (
            subset_size
            if subset_size is not None
            else (Config.SUBSET_SIZE if debug else None)
        )

        # Override Config epochs if provided
        if epochs is not None:
            Config.EPOCHS = epochs
        self.epochs = Config.EPOCHS

        self.device = torch.device(Config.DEVICE)
        self._set_seeds()

        # Ensure working directories exist
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    def _set_seeds(self):
        """Sets fixed random seeds for reproducibility."""
        seed = Config.SEED
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_dataloader(self, split):
        """
        Creates a DataLoader for the specified split.

        Args:
            split (str): 'train', 'val', or 'test'.

        Returns:
            DataLoader: Configured data loader.
        """
        # Determine if we should use a subset
        use_subset = self.subset_size

        # Initialize dataset (handles caching internally)
        dataset = LidarDataset(
            split=split, subset_size=use_subset, load_cached_data=True
        )

        is_train = split == "train"

        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=is_train,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def run(self):
        """
        Executes the full pipeline: Training -> Validation -> Inference.
        """
        print(
            f"Initializing Trainer (Debug={self.debug}, Device={self.device}, Epochs={self.epochs})"
        )

        # 1. Setup DataLoaders
        print("Setting up data loaders...")
        train_loader = self._get_dataloader("train")
        val_loader = self._get_dataloader("val")
        test_loader = self._get_dataloader("test")

        # 2. Initialize Model
        print("Initializing PointPillars Detector...")
        detector = PointPillarsDetector()

        # 3. Training Loop
        print("Starting training...")

        # Initialize Scheduler manually since we are controlling the loop
        steps_per_epoch = len(train_loader)
        detector.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            detector.optimizer,
            max_lr=Config.LR,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            pct_start=0.3,
        )

        best_val_loss = float("inf")
        checkpoint_path = os.path.join(Config.WORKING_DIR, "model_checkpoint.pth")

        for epoch in range(self.epochs):
            # Train Epoch
            train_loss = detector.train_epoch(train_loader, epoch)

            # Validate
            val_loss = detector.validate(val_loader)

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpoint Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                detector.best_val_loss = best_val_loss
                detector.save_checkpoint(checkpoint_path)

            # Early Stopping Logic
            # Trigger if val loss explodes relative to best (divergence)
            if epoch > 5 and val_loss > best_val_loss * 1.2:
                print("Early stopping triggered.")
                break

        # 4. Load Best Model
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from {checkpoint_path}...")
            detector.load_checkpoint(checkpoint_path)
        else:
            print("Warning: No checkpoint found. Using current model state.")

        # 5. Generate Submission
        print("Generating submission for test set...")
        detector.generate_submission(test_loader, Config.SUBMISSION_PATH)
        print("Pipeline complete.")
