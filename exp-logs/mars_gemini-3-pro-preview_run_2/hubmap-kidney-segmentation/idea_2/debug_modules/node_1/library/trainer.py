import os
import time
import gc
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

from library.config import Config
from library.utils import setup_logger, rle_encode
from library.loss_metrics import HybridBCEDiceLoss, DiceScore
from library.model import AnatomyAwareUNetPlusPlus
from library.data_processing import prepare_train_val_loaders, get_test_loader


class WarmupEarlyStopping:
    """
    Early stopping mechanism with a warmup period.

    This class monitors the validation loss. For the first `warmup_epochs`,
    it tracks the best loss but does not increment the patience counter,
    preventing premature stopping due to early training volatility.
    """

    def __init__(
        self, patience=10, min_delta=0, warmup_epochs=15, verbose=True, logger=None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        self.logger = logger
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # Only enforce early stopping after the warmup period
            if epoch > self.warmup_epochs:
                self.counter += 1
                if self.logger and self.verbose:
                    self.logger.info(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class Trainer:
    """
    Trainer class encapsulating the training loop, validation, and inference logic.
    """

    def __init__(self, load_cached_data=True):
        self.device = torch.device(Config.DEVICE)
        self.logger = setup_logger()

        # Prepare Data
        self.logger.info("Initializing DataLoaders...")
        self.train_loader, self.val_loader = prepare_train_val_loaders(
            load_cached_data=load_cached_data
        )

        # Initialize Model
        self.logger.info(
            f"Initializing {Config.MODEL_ARCH} model with {Config.ENCODER_NAME} encoder..."
        )
        self.model = AnatomyAwareUNetPlusPlus().to(self.device)

        # Loss and Metrics
        self.criterion = HybridBCEDiceLoss()
        self.metric = DiceScore()

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Early Stopping
        self.early_stopping = WarmupEarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            warmup_epochs=Config.WARMUP_EPOCHS,
            logger=self.logger,
        )

    def train_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self, epoch):
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = self.metric(outputs, masks)

                running_loss += loss.item()
                running_dice += dice.item()

        val_loss = running_loss / len(self.val_loader)
        val_dice = running_dice / len(self.val_loader)
        return val_loss, val_dice

    def fit(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        best_val_loss = float("inf")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_dice = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            duration = time.time() - start_time

            # Log metrics (full precision)
            self.logger.info(f"Epoch {epoch}/{Config.EPOCHS} - Time: {duration:.2f}s")
            self.logger.info(f"Train Loss: {train_loss}")
            self.logger.info(f"Val Loss: {val_loss}")
            self.logger.info(f"Val Dice: {val_dice}")

            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                self.logger.info(
                    f"Model saved at epoch {epoch} with Val Loss: {val_loss}"
                )

            # Check Early Stopping
            if self.early_stopping(val_loss, epoch):
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss}")

    def _get_gaussian_window(self, size):
        """
        Generates a 2D Gaussian window to weight predictions during reconstruction.
        This helps reduce edge artifacts when merging overlapping tiles.
        """
        sigma = size / 2.0
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        x, y = np.meshgrid(x, y)
        d = np.sqrt(x * x + y * y)
        g = np.exp(-(d**2) / (2.0 * 0.5**2))
        return g

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves the submission CSV.
        Reconstructs full images from tiles using Gaussian weighted averaging.
        """
        self.logger.info("Starting inference and submission generation...")

        # Load the best model
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
            self.logger.info(f"Loaded model from {Config.MODEL_PATH}")
        else:
            self.logger.warning("No saved model found. Using current model state.")

        self.model.eval()
        test_loader = get_test_loader()

        # Gaussian window for smooth blending
        window = self._get_gaussian_window(Config.TILE_SIZE)

        # Buffers for image reconstruction
        current_id = None
        buffer_pred = None
        buffer_weight = None

        results = []

        with torch.no_grad():
            for images, coords, ids, shapes in test_loader:
                images = images.to(self.device)

                # Predict
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()  # (B, 1, H, W)

                coords = coords.numpy()

                # Handle shapes (can be list of tensors or tensor)
                if isinstance(shapes, torch.Tensor):
                    shapes = shapes.numpy()
                else:
                    shapes = np.array([s.numpy() for s in shapes])

                # Iterate through batch
                for i in range(len(images)):
                    img_id = ids[i]
                    y, x = coords[i]
                    h, w = shapes[i]
                    prob_tile = probs[i, 0]

                    # Check if we switched to a new image
                    if img_id != current_id:
                        # Process and save the previous image
                        if current_id is not None:
                            self.logger.info(
                                f"Processing prediction for {current_id}..."
                            )

                            # Normalize by weight map
                            mask = buffer_pred / (buffer_weight + 1e-6)

                            # Threshold
                            mask = (mask > Config.MASK_THRESHOLD).astype(np.uint8)

                            # Encode
                            rle = rle_encode(mask)
                            results.append({"id": current_id, "predicted": rle})

                            # Free memory
                            del buffer_pred, buffer_weight, mask
                            gc.collect()

                        # Initialize buffers for new image
                        current_id = img_id
                        buffer_pred = np.zeros((h, w), dtype=np.float32)
                        buffer_weight = np.zeros((h, w), dtype=np.float32)

                    # Accumulate prediction and weights
                    # Ensure dimensions match (handling potential edge cases if any)
                    th, tw = prob_tile.shape

                    buffer_pred[y : y + th, x : x + tw] += prob_tile * window[:th, :tw]
                    buffer_weight[y : y + th, x : x + tw] += window[:th, :tw]

        # Process the final image
        if current_id is not None:
            self.logger.info(f"Processing prediction for {current_id}...")
            mask = buffer_pred / (buffer_weight + 1e-6)
            mask = (mask > Config.MASK_THRESHOLD).astype(np.uint8)
            rle = rle_encode(mask)
            results.append({"id": current_id, "predicted": rle})

        # Save to CSV
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
