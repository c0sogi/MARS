import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import Config
from library.utils import set_seed, compute_map_score
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.model import DeepResUNet
from library.dataset import get_dataloaders


class Trainer:
    def __init__(self):
        # 1. Setup
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # 2. Data
        self.train_loader, self.val_loader, _ = get_dataloaders(load_cached_data=True)

        # 3. Model
        self.model = DeepResUNet().to(self.device)

        # 4. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Scheduler (Cosine Annealing Warm Restarts)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=Config.CYCLE_LEN,
            T_mult=Config.T_MULT,
            eta_min=Config.ETA_MIN,
        )

        # 6. Losses
        self.criterion_phase1 = BCEDiceLoss().to(self.device)
        self.criterion_phase2 = LovaszHingeLoss(bce_weight=0.5).to(self.device)

        # 7. Tracking
        self.best_global_map = -1.0
        self.best_cycle_maps = {c: -1.0 for c in range(1, Config.CYCLES + 1)}

    def get_criterion(self, epoch):
        """Selects loss function based on Curriculum Schedule."""
        if epoch < Config.LOSS_SWITCH_EPOCH:
            return self.criterion_phase1
        return self.criterion_phase2

    def train_one_epoch(self, epoch):
        self.model.train()
        criterion = self.get_criterion(epoch)

        running_loss = 0.0

        for batch_idx, (images, masks, depths, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            depths = depths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with Deep Supervision
            logits, aux2, aux1 = self.model(images, depths)

            # Calculate Loss (Main + Aux)
            loss_main = criterion(logits, masks)
            loss_aux2 = criterion(aux2, masks)
            loss_aux1 = criterion(aux1, masks)

            # Weighted sum of losses
            loss = loss_main + 0.5 * loss_aux2 + 0.5 * loss_aux1

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        # Step scheduler at epoch level
        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        # We use Phase 1 criterion for consistent validation loss tracking,
        # but mAP is the primary metric.
        criterion = self.criterion_phase1

        running_loss = 0.0
        all_preds = []
        all_targets = []

        # Calculate cropping indices to revert 128x128 -> 101x101
        # Pad was (128-101)//2 = 13
        start_idx = (Config.IMG_HEIGHT - Config.ORIG_HEIGHT) // 2
        end_idx = start_idx + Config.ORIG_HEIGHT

        with torch.no_grad():
            for images, masks, depths, _ in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                depths = depths.to(self.device)

                # Forward pass (Inference only returns logits)
                logits = self.model(images, depths)

                loss = criterion(logits, masks)
                running_loss += loss.item()

                # Apply sigmoid
                preds = torch.sigmoid(logits)

                # Crop back to original 101x101 resolution for accurate mAP calculation
                preds_cropped = preds[:, :, start_idx:end_idx, start_idx:end_idx]
                masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

                all_preds.append(preds_cropped.cpu().numpy())
                all_targets.append(masks_cropped.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate mAP
        val_map = compute_map_score(all_preds, all_targets)

        return running_loss / len(self.val_loader), val_map

    def save_checkpoint(self, filename):
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")
        print(f"Device: {self.device}")
        print(f"Loss Switch Epoch: {Config.LOSS_SWITCH_EPOCH}")

        start_time = time.time()

        for epoch in range(Config.EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_map = self.evaluate()

            # Determine Cycle
            # Cycle 1: 0-49, Cycle 2: 50-99, Cycle 3: 100-149
            current_cycle = (epoch // Config.CYCLE_LEN) + 1

            # Logging
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} [Cycle {current_cycle}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val mAP: {val_map:.10f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Checkpointing Logic

            # 1. Global Best
            if val_map > self.best_global_map:
                self.best_global_map = val_map
                self.save_checkpoint("best_model.pth")

            # 2. Cycle Best (Snapshot Ensembling)
            if val_map > self.best_cycle_maps[current_cycle]:
                self.best_cycle_maps[current_cycle] = val_map
                # Only save to disk if it's one of the target cycles for ensembling
                if current_cycle in Config.SNAPSHOT_CYCLES:
                    self.save_checkpoint(f"best_cycle_{current_cycle}.pth")

        total_time = time.time() - start_time
        print(f"\nTraining Complete. Total Time: {total_time/60:.2f} minutes.")
        print(f"Best Global mAP: {self.best_global_map:.10f}")
        for c in Config.SNAPSHOT_CYCLES:
            print(f"Best Cycle {c} mAP: {self.best_cycle_maps[c]:.10f}")


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    trainer = Trainer()
    trainer.fit()
