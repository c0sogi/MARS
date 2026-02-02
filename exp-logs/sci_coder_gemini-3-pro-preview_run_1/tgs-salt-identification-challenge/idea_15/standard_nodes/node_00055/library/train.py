import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import Config
from library.utils import set_seed, do_kaggle_metric, SWAHandler
from library.losses import BCEDiceLoss, BCELovaszLoss
from library.model import DeepResUNet
from library.dataset import SaltDataset


class Trainer:
    """
    Manages the training lifecycle of the Deep Residual U-Net.
    Implements the curriculum learning strategy (Dice -> Lovasz) and SWA.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # 1. Data Loaders
        print("Initializing Datasets...")
        self.train_dataset = SaltDataset(mode="train", load_cached_data=True)
        self.val_dataset = SaltDataset(mode="val", load_cached_data=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # 2. Model
        print("Initializing Model...")
        self.model = DeepResUNet().to(self.device)

        # 3. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Cosine Annealing Warm Restarts
        # T_0 is the number of epochs for the first restart.
        # We want 3 cycles of 50 epochs each.
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=1e-6
        )

        # 4. Losses
        self.criterion_phase1 = BCEDiceLoss().to(self.device)
        self.criterion_phase2 = BCELovaszLoss().to(self.device)
        self.current_criterion = self.criterion_phase1

        # 5. SWA Handler
        self.swa_handler = SWAHandler(self.model) if Config.USE_SWA else None

        # 6. Tracking
        self.best_map = 0.0
        self.best_cycle_2_map = 0.0

        # Calculate cropping indices for validation (128 -> 101)
        pad_total = Config.IMG_SIZE - Config.ORIG_SIZE
        self.pad_top = pad_total // 2
        self.pad_bottom = pad_total - self.pad_top
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left

    def crop_to_original(self, tensor):
        """Crops a (B, C, H, W) tensor from 128x128 back to 101x101."""
        # H is dim 2, W is dim 3
        return tensor[
            ...,
            self.pad_top : Config.IMG_SIZE - self.pad_bottom,
            self.pad_left : Config.IMG_SIZE - self.pad_right,
        ]

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        # Switch Loss function based on Curriculum
        if epoch == Config.LOVASZ_SWITCH_EPOCH:
            print(f"\n[Curriculum] Switching to BCE + Lovasz Loss at Epoch {epoch}")
            self.current_criterion = self.criterion_phase2

        for i, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with Deep Supervision
            # Returns: (logits, logits_aux2, logits_aux1)
            outputs = self.model(images)

            # Unpack outputs
            if Config.DEEP_SUPERVISION:
                logits, aux2, aux1 = outputs
                loss_main = self.current_criterion(logits, masks)
                loss_aux2 = self.current_criterion(aux2, masks)
                loss_aux1 = self.current_criterion(aux1, masks)

                # Weighted sum of losses
                loss = loss_main + 0.5 * loss_aux2 + 0.5 * loss_aux1
            else:
                logits = outputs
                loss = self.current_criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        # Update SWA if active and in the correct epoch range
        if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
            self.swa_handler.update(self.model)

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward (In eval mode, model returns only logits)
                logits = self.model(images)

                # Calculate validation loss on padded data for consistency
                loss = self.current_criterion(logits, masks)
                running_loss += loss.item()

                # Sigmoid for predictions
                probs = torch.sigmoid(logits)

                # Crop back to 101x101 for Metric Calculation
                probs_cropped = self.crop_to_original(probs)
                masks_cropped = self.crop_to_original(masks)

                preds_list.append(probs_cropped.cpu().numpy())
                targets_list.append(masks_cropped.cpu().numpy())

        # Concatenate
        preds_arr = np.concatenate(preds_list, axis=0)
        targets_arr = np.concatenate(targets_list, axis=0)

        # Remove channel dim: (B, 1, H, W) -> (B, H, W)
        preds_arr = preds_arr.squeeze(1)
        targets_arr = targets_arr.squeeze(1)

        # Calculate Kaggle Metric (mAP over IoU thresholds)
        score = do_kaggle_metric(preds_arr, targets_arr)

        return running_loss / len(self.val_loader), score

    def save_checkpoint(self, name):
        path = os.path.join(Config.CHECKPOINT_DIR, name)
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {name}")

    def save_swa_checkpoint(self, name):
        if self.swa_handler:
            path = os.path.join(Config.CHECKPOINT_DIR, name)
            torch.save(self.swa_handler.get_model().state_dict(), path)
            print(f"Saved SWA checkpoint: {name}")

    def run(self):
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
        print(f"Device: {self.device}")

        start_time = time.time()

        for epoch in range(Config.NUM_EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Step Scheduler
            self.scheduler.step()

            # Validate
            val_loss, val_map = self.validate()

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val mAP: {val_map:.10f}"
            )

            # Checkpoint Logic

            # 1. Overall Best Model
            if val_map > self.best_map:
                self.best_map = val_map
                self.save_checkpoint("best_model.pth")

            # 2. Best Model from Cycle 2 (Epochs 50-99)
            # Cycle 2 is index 1 (0-based), so epochs 50 to 99.
            if 50 <= epoch < 100:
                if val_map > self.best_cycle_2_map:
                    self.best_cycle_2_map = val_map
                    self.save_checkpoint("best_cycle_2.pth")

        # End of Training
        print("\nTraining Complete.")

        # Final SWA Handling
        if Config.USE_SWA:
            print("Updating SWA BatchNorm statistics...")
            self.swa_handler.update_bn(self.train_loader, device=self.device)
            self.save_swa_checkpoint("swa_model.pth")

        total_time = time.time() - start_time
        print(f"Total Runtime: {total_time/60:.2f} minutes.")
        print(f"Best Overall mAP: {self.best_map:.10f}")
        print(f"Best Cycle 2 mAP: {self.best_cycle_2_map:.10f}")


def main():
    # Ensure directories exist
    Config.setup()

    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
