import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda import amp

from library.config import Config
from library.utils import (
    set_seed,
    get_dice_coef,
    get_3d_hausdorff,
    keep_largest_connected_component_3d,
)
from library.loss import WeightedDeepSupervisionLoss
from library.model import UnetPlusPlus
from library.dataset import prepare_loaders


class Trainer:
    """
    Manages the training, validation, and checkpointing of the U-Net++ model.
    """

    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE

        # Initialize Model
        self.model = UnetPlusPlus().to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=1e-6
        )

        # Loss Function
        self.criterion = WeightedDeepSupervisionLoss()

        # Mixed Precision Scaler
        self.scaler = amp.GradScaler("cuda")

        # Tracking
        self.best_score = -np.inf

        # Load Validation Metadata for 3D Reconstruction
        # We need this to map flat predictions back to 3D volumes (Case + Day)
        self.val_df = pd.read_csv(Config.VAL_CSV, keep_default_na=False)

        # Align validation metadata with the loader if in DEBUG mode
        if Config.DEBUG:
            self.val_df = self.val_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(
                drop=True
            )

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(self.train_loader)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with amp.autocast():
                # Model returns list of tensors if Deep Supervision is active
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward Pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / dataset_size

    def validate(self):
        """
        Runs validation using the 'reconstruct-then-score' methodology.
        1. Predicts 2D masks for all validation slices.
        2. Reconstructs 3D volumes based on metadata.
        3. Applies 3D CCA post-processing.
        4. Computes 3D Hausdorff and Dice metrics.
        """
        self.model.eval()

        all_preds = []
        all_masks = []

        # 1. Inference Loop (Slice-wise)
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)

                # Forward (Eval mode returns single tensor)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                # Move to CPU to save GPU memory
                all_preds.append(probs.cpu().numpy())
                all_masks.append(masks.numpy())

        # Concatenate all batches -> (N, 3, H, W)
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # 2. 3D Reconstruction & Scoring
        # Group slices by (Case, Day)
        groups = self.val_df.groupby(["case", "day"])

        metrics = {"dice": [], "hd": []}

        for (case, day), group in groups:
            # Get indices for this volume
            indices = group.index.to_numpy()

            # Extract slices
            vol_preds = all_preds[indices]  # (Depth, 3, H, W)
            vol_masks = all_masks[indices]  # (Depth, 3, H, W)

            # Ensure slices are sorted by Z-position (slice number)
            slice_nums = group["slice"].astype(int).values
            sort_idx = np.argsort(slice_nums)

            vol_preds = vol_preds[sort_idx]
            vol_masks = vol_masks[sort_idx]

            # Process each class independently
            for class_idx in range(Config.NUM_CLASSES):
                # Extract class volume: (Depth, H, W)
                p_vol = vol_preds[:, class_idx, :, :]
                t_vol = vol_masks[:, class_idx, :, :]

                # Binarize
                p_vol_bin = (p_vol > Config.MASK_THRESHOLD).astype(np.uint8)
                t_vol_bin = (t_vol > 0.5).astype(np.uint8)

                # 3. Post-Processing: 3D Connected Component Analysis
                # Keep only the largest object to remove noise
                p_vol_processed = keep_largest_connected_component_3d(
                    p_vol_bin, min_size=Config.MIN_COMPONENT_SIZE
                )

                # 4. Compute Metrics
                dice = get_dice_coef(t_vol_bin, p_vol_processed)
                hd = get_3d_hausdorff(t_vol_bin, p_vol_processed)

                metrics["dice"].append(dice)
                metrics["hd"].append(hd)

        # Aggregate Metrics
        mean_dice = np.mean(metrics["dice"])
        mean_hd = np.mean(metrics["hd"])

        # Competition Score: 0.4 * Dice + 0.6 * (1 - HD)
        # Note: HD is normalized distance (0=good, 1=bad), so we use (1-HD) for scoring.
        score = 0.4 * mean_dice + 0.6 * (1.0 - mean_hd)

        return score, mean_dice, mean_hd

    def save_model(self, filename):
        """Saves the model weights."""
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def fit(self, epochs, patience=5):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs (Patience={patience})...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_score, val_dice, val_hd = self.validate()

            # Scheduler Step
            self.scheduler.step()

            # Logging
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.1f}s")
            print(f"  Train Loss: {train_loss:.5f}")
            print(
                f"  Val Score: {val_score:.5f} | Dice: {val_dice:.5f} | HD: {val_hd:.5f}"
            )

            # Checkpointing & Early Stopping
            if val_score > self.best_score:
                print(
                    f"  [+] Score improved: {self.best_score:.5f} -> {val_score:.5f}. Saving model."
                )
                self.best_score = val_score
                self.save_model("best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"  [-] Score did not improve. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Score: {self.best_score:.5f}")


def run_training():
    """
    Entry point to initialize data and start training.
    """
    # Reproducibility
    set_seed(Config.SEED)

    # Prepare Data
    print("Initializing DataLoaders...")
    train_loader, val_loader = prepare_loaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Start Training
    trainer.fit(epochs=Config.EPOCHS, patience=5)
