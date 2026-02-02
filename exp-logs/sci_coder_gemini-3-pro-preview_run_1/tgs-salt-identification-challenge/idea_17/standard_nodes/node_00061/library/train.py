import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import Config
from library.utils import seed_everything, do_kaggle_metric
from library.model import DeepResUNet
from library.loss import BCEDiceLoss, BCELovaszLoss, DeepSupervisionLoss
from library.dataset import get_loaders


class Trainer:
    """
    Manages the training lifecycle for the Deep Residual U-Net.
    Implements the Extended Homogeneous Lovasz-Cycle curriculum.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = Config.DEVICE

        # Ensure reproducibility
        seed_everything(Config.SEED)

        # Setup directories
        Config.setup_directories()

        # Initialize DataLoaders
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader = get_loaders(
            debug=self.debug, load_cached_data=True
        )

        # Initialize Model
        print("Initializing Model...")
        self.model = DeepResUNet().to(self.device)

        # Initialize Optimizer
        # Using AdamW as specified in the idea
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
        )

        # Initialize Scheduler
        # Cosine Annealing Warm Restarts
        # T_0 = 50 epochs (length of one cycle)
        # T_mult = 1 (constant cycle length)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=Config.LR_MIN
        )

        # Initialize Loss Functions
        # Phase 1: Structure (BCE + Dice)
        self.loss_phase1 = DeepSupervisionLoss(
            main_loss_fn=BCEDiceLoss(),
            aux_loss_fn=BCEDiceLoss(),
            aux_weights=[0.5, 0.5],  # Weights for aux heads
        )

        # Phase 2: Metric (BCE + Lovasz)
        self.loss_phase2 = DeepSupervisionLoss(
            main_loss_fn=BCELovaszLoss(),
            aux_loss_fn=BCELovaszLoss(),  # Aux heads also switch to Lovasz
            aux_weights=[0.5, 0.5],
        )

        # State tracking
        self.best_map_cycle_1 = 0.0
        self.best_map_cycle_2 = 0.0
        self.best_map_cycle_3 = 0.0
        self.best_map_cycle_4 = 0.0

        # Calculate cropping indices for validation (128 -> 101)
        pad_h = Config.IMG_H - Config.ORIG_H
        pad_w = Config.IMG_W - Config.ORIG_W
        self.crop_top = pad_h // 2
        self.crop_bottom = self.crop_top + Config.ORIG_H
        self.crop_left = pad_w // 2
        self.crop_right = self.crop_left + Config.ORIG_W

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        # Select Loss Function based on Phase
        if epoch_idx < Config.CYCLE_1_END_EPOCH:
            criterion = self.loss_phase1
        else:
            criterion = self.loss_phase2

        for batch_idx, (images, masks, depths, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns list of outputs due to deep supervision)
            outputs = self.model(images, depths)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        # Validation always uses the Phase 2 metric (Lovasz) or just BCE for loss tracking
        # Ideally, we track the loss relevant to the current phase, but for consistency
        # let's just track BCE+Dice for stability in logging, or mirror training.
        # Let's mirror training phase for loss calculation, but mAP is the real metric.
        # However, since validation doesn't use deep supervision outputs (returns single tensor),
        # we need to handle that.

        # Simple BCE+Dice for validation loss tracking
        val_criterion = BCEDiceLoss()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks, depths, _ in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                # Forward pass (eval mode returns single tensor)
                output = self.model(images, depths)

                # Calculate Loss
                loss = val_criterion(output, masks)
                val_loss += loss.item()

                # Sigmoid to get probabilities
                probs = torch.sigmoid(output)

                # Crop to original size for accurate metric calculation
                # Output shape: (B, 1, 128, 128)
                probs_cropped = probs[
                    :,
                    :,
                    self.crop_top : self.crop_bottom,
                    self.crop_left : self.crop_right,
                ]
                masks_cropped = masks[
                    :,
                    :,
                    self.crop_top : self.crop_bottom,
                    self.crop_left : self.crop_right,
                ]

                all_preds.append(probs_cropped.cpu().numpy())
                all_targets.append(masks_cropped.cpu().numpy())

        # Concatenate
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Remove channel dim: (N, 1, H, W) -> (N, H, W)
        all_preds = all_preds.squeeze(1)
        all_targets = all_targets.squeeze(1)

        # Calculate mAP
        map_score = do_kaggle_metric(all_preds, all_targets, threshold=0.5)

        return val_loss / len(self.val_loader), map_score

    def save_checkpoint(self, filename):
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")

    def train(self):
        print(f"Starting training for {Config.TOTAL_EPOCHS} epochs...")
        print(f"Phase 1 (Structure): Epochs 1-{Config.CYCLE_1_END_EPOCH}")
        print(
            f"Phase 2 (Metric): Epochs {Config.CYCLE_1_END_EPOCH+1}-{Config.TOTAL_EPOCHS}"
        )

        start_time = time.time()

        for epoch in range(Config.TOTAL_EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_map = self.validate()

            # Step Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Determine Cycle
            # Cycle 1: 0-49
            # Cycle 2: 50-99
            # Cycle 3: 100-149
            # Cycle 4: 150-199
            cycle_idx = epoch // Config.EPOCHS_PER_CYCLE + 1

            # Logging
            print(
                f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS} [Cycle {cycle_idx}] | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val mAP: {val_map:.10f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpointing Logic
            # We save the best model for each cycle independently

            if cycle_idx == 1:
                if val_map > self.best_map_cycle_1:
                    self.best_map_cycle_1 = val_map
                    self.save_checkpoint("best_cycle_1.pth")

            elif cycle_idx == 2:
                # Reset tracking at start of cycle? No, just track max within this window.
                # Since variable is initialized to 0, it works.
                if val_map > self.best_map_cycle_2:
                    self.best_map_cycle_2 = val_map
                    self.save_checkpoint("best_cycle_2.pth")

            elif cycle_idx == 3:
                if val_map > self.best_map_cycle_3:
                    self.best_map_cycle_3 = val_map
                    self.save_checkpoint("best_cycle_3.pth")

            elif cycle_idx == 4:
                if val_map > self.best_map_cycle_4:
                    self.best_map_cycle_4 = val_map
                    self.save_checkpoint("best_cycle_4.pth")
                    # Also save as best_model.pth for generic usage if needed,
                    # though ensemble will use specific cycle files.
                    self.save_checkpoint("best_model.pth")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time/60:.2f} minutes.")
        print(f"Best mAP Cycle 1: {self.best_map_cycle_1:.10f}")
        print(f"Best mAP Cycle 2: {self.best_map_cycle_2:.10f}")
        print(f"Best mAP Cycle 3: {self.best_map_cycle_3:.10f}")
        print(f"Best mAP Cycle 4: {self.best_map_cycle_4:.10f}")
