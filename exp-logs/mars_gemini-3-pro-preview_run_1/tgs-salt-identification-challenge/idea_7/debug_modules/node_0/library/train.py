import os
import time
import torch
import numpy as np
import torch.optim as optim
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HyperResUNet
from library.losses import BCEDiceLoss, BCELovaszLoss, DeepSupervisionLoss
from library.utils import do_kaggle_metric, unpad_image


class Trainer:
    def __init__(self):
        # 1. Setup Device and Config
        self.device = torch.device(Config.DEVICE)
        Config.set_seed(Config.SEED)

        # 2. Data Loaders
        self.train_loader, self.val_loader, _ = get_dataloaders()

        # 3. Model
        self.model = HyperResUNet().to(self.device)

        # 4. Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Scheduler: Cosine Annealing Warm Restarts
        # T_0 = 50 epochs per cycle
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.CYCLE_LEN, T_mult=1, eta_min=1e-6
        )

        # 6. Losses
        # Phase 1: BCE + Dice
        self.criterion_phase1 = DeepSupervisionLoss(BCEDiceLoss())
        # Phase 2: BCE + Lovasz
        self.criterion_phase2 = DeepSupervisionLoss(BCELovaszLoss())

        # 7. Tracking
        self.best_map_global = 0.0
        self.best_map_cycle_2 = 0.0
        self.best_map_cycle_3 = 0.0

        # Ensure checkpoint directory exists
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        # Select criterion based on curriculum
        if epoch_idx < Config.PHASE_1_EPOCHS:
            criterion = self.criterion_phase1
        else:
            criterion = self.criterion_phase2

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Model returns list of tensors because deep_supervision=True in training
            outputs = self.model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()

        preds_all = []
        targets_all = []

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)

                # Forward pass
                # Model returns single tensor (main head) in eval mode
                output = self.model(images)
                pred_prob = torch.sigmoid(output)

                # Move to CPU
                pred_prob = pred_prob.cpu().numpy()  # (B, 1, 128, 128)
                masks = masks.numpy()  # (B, 1, 128, 128)

                # Unpad to original 101x101 for accurate metric calculation
                for i in range(pred_prob.shape[0]):
                    # Squeeze channel dim for unpad utility
                    p_img = pred_prob[i, 0, :, :]
                    m_img = masks[i, 0, :, :]

                    p_unpadded = unpad_image(
                        p_img, (Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                    )
                    m_unpadded = unpad_image(
                        m_img, (Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
                    )

                    preds_all.append(p_unpadded)
                    targets_all.append(m_unpadded)

        # Calculate mAP
        # Stack into arrays (N, 101, 101)
        preds_all = np.array(preds_all)
        targets_all = np.array(targets_all)

        score = do_kaggle_metric(preds_all, targets_all)
        return score

    def fit(self):
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
        print(f"Phase 1 (BCE+Dice): Epochs 0-{Config.PHASE_1_EPOCHS-1}")
        print(f"Phase 2 (Lovasz): Epochs {Config.PHASE_1_EPOCHS}-{Config.NUM_EPOCHS-1}")

        start_time = time.time()

        for epoch in range(Config.NUM_EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_map = self.validate()

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_duration = time.time() - epoch_start

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val mAP: {val_map:.10f} | "
                f"Time: {epoch_duration:.2f}s"
            )

            # Checkpointing Logic

            # Global Best
            if val_map > self.best_map_global:
                self.best_map_global = val_map
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
                )

            # Cycle 2 Best (Epochs 50-99)
            # Indices are 0-based, so Cycle 2 is 50 to 99
            if 50 <= epoch < 100:
                if val_map > self.best_map_cycle_2:
                    self.best_map_cycle_2 = val_map
                    torch.save(self.model.state_dict(), Config.CYCLE_2_BEST_MODEL)
                    print(f"  -> New Cycle 2 Best: {val_map:.6f}")

            # Cycle 3 Best (Epochs 100-149)
            if 100 <= epoch < 150:
                if val_map > self.best_map_cycle_3:
                    self.best_map_cycle_3 = val_map
                    torch.save(self.model.state_dict(), Config.CYCLE_3_BEST_MODEL)
                    print(f"  -> New Cycle 3 Best: {val_map:.6f}")

        total_time = time.time() - start_time
        print(f"\nTraining Complete. Total Time: {total_time/60:.2f} min")
        print(f"Best Global mAP: {self.best_map_global:.6f}")
        print(f"Best Cycle 2 mAP: {self.best_map_cycle_2:.6f}")
        print(f"Best Cycle 3 mAP: {self.best_map_cycle_3:.6f}")


def train_model():
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    train_model()
