import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import (
    device,
    CHECKPOINT_DIR,
    LOG_DIR,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    SCHEDULER,
    MIN_LR,
    T_MAX,
    THR_LARGE_BOWEL,
    THR_SMALL_BOWEL,
    THR_STOMACH,
    NUM_CLASSES,
    SEED,
)
from library.utils import set_seed, keep_largest_component_3d
from library.dataset import get_loaders
from library.model import UnetPlusPlus
from library.losses import BCETverskyLoss
from library.metrics import get_competition_score


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = device
        self.output_dir = CHECKPOINT_DIR

        # Initialize DataLoaders
        print(f"Initializing DataLoaders (Debug={debug})...")
        self.train_loader, self.val_loader = get_loaders(
            load_cached_data=True, debug=debug
        )

        # Initialize Model
        print("Initializing U-Net++ Model...")
        self.model = UnetPlusPlus().to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_MAX, eta_min=MIN_LR
        )

        # Loss & Scaler
        self.criterion = BCETverskyLoss().to(self.device)
        self.scaler = GradScaler()

        # Metrics
        self.best_score = -np.inf

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        # Set seed for reproducibility per epoch
        set_seed(SEED + epoch)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            with autocast():
                # Forward pass
                # In training, U-Net++ with Deep Supervision returns a list of tensors
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate_volumetric(self):
        """
        Performs validation by reconstructing 3D volumes and calculating
        the competition metric (Dice + Hausdorff) with 3D CCA post-processing.
        """
        self.model.eval()

        # Containers for full validation set predictions
        # We store flattened arrays to reconstruct volumes later
        all_preds = []
        all_targets = []

        print("Generating validation predictions...")
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)

                with autocast():
                    # In eval mode, model returns single tensor (final output)
                    outputs = self.model(images)
                    outputs = torch.sigmoid(outputs)

                # Move to CPU to save GPU memory
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        # Concatenate all batches: (N, C, H, W)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Get metadata to group slices into volumes
        val_df = self.val_loader.dataset.df

        # Ensure alignment
        if len(val_df) != len(all_preds):
            raise ValueError(
                f"Mismatch: DF len {len(val_df)} vs Preds len {len(all_preds)}"
            )

        # Group by Case + Day
        groups = val_df.groupby(["case", "day"])

        scores = []

        # Thresholds per class
        thresholds = [THR_LARGE_BOWEL, THR_SMALL_BOWEL, THR_STOMACH]

        print(f"Evaluating {len(groups)} volumes...")

        for (case, day), group_df in groups:
            # Get indices for this volume
            indices = group_df.index.values

            # Extract volume slices: (D, C, H, W)
            vol_preds_raw = all_preds[indices]
            vol_targets = all_targets[indices]

            # Sort by slice number to ensure correct 3D structure
            # The dataset might be sorted, but we enforce it here using the 'slice' column
            slice_nums = group_df["slice"].values
            sort_idx = np.argsort(slice_nums)

            vol_preds_raw = vol_preds_raw[sort_idx]
            vol_targets = vol_targets[sort_idx]

            # Iterate over classes
            for cls_idx in range(NUM_CLASSES):
                # Extract specific class volume: (D, H, W)
                y_pred_prob = vol_preds_raw[:, cls_idx, :, :]
                y_true = vol_targets[:, cls_idx, :, :]

                # Binarize
                y_pred = (y_pred_prob > thresholds[cls_idx]).astype(np.uint8)
                y_true = (y_true > 0.5).astype(np.uint8)

                # Apply 3D CCA (Post-processing)
                # This keeps only the largest connected component
                y_pred_processed = keep_largest_component_3d(y_pred)

                # Calculate metric
                score = get_competition_score(y_true, y_pred_processed)
                scores.append(score)

        # Average score across all (Case, Day, Class) tuples
        final_metric = np.mean(scores)
        return final_metric

    def save_checkpoint(self, score, filename):
        save_path = os.path.join(self.output_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "score": score,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")

    def fit(self, epochs=EPOCHS):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate (Volumetric)
            val_score = self.validate_volumetric()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score (3D): {val_score:.6f}"
            )

            # Checkpointing
            if val_score > self.best_score:
                print(
                    f"Score Improved ({self.best_score:.6f} -> {val_score:.6f}). Saving Best Model..."
                )
                self.best_score = val_score
                self.save_checkpoint(val_score, "best_model.pth")

            # Save Last Model
            self.save_checkpoint(val_score, "last_model.pth")

            # Memory Cleanup
            gc.collect()
            torch.cuda.empty_cache()


def run_training(debug=False, epochs=EPOCHS):
    set_seed(SEED)
    trainer = Trainer(debug=debug)
    trainer.fit(epochs=epochs)
