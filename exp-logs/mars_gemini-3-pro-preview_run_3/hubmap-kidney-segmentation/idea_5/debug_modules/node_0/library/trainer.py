import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from library.utils import set_seed


class DeepSupervisionLoss(nn.Module):
    """
    Combined BCE and Dice Loss for Deep Supervision.
    Calculates weighted loss across multiple model outputs.
    """

    def __init__(self, weights=[1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, target):
        loss = 0.0
        # preds is a list of tensors from the model [fine, medium, coarse, very_coarse]
        # target is the ground truth mask (B, 1, H, W)
        # The model outputs are already upsampled to (H, W), so we can compare directly.

        for i, pred in enumerate(preds):
            if i < len(self.weights):
                w = self.weights[i]
                l_bce = self.bce(pred, target)
                l_dice = self.dice_loss(pred, target)
                # Combined loss: 0.5 BCE + 0.5 Dice
                loss += w * (0.5 * l_bce + 0.5 * l_dice)

        return loss


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Args:
            model: The PyTorch model (ConvNeXtUNetPlusPlus).
            train_dataset: Instance of HubmapDataset for training.
            val_dataset: Instance of HubmapDataset for validation.
            config: Dictionary containing hyperparameters (lr, batch_size, epochs, etc.).
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Optimization
        self.learning_rate = config.get("lr", 1e-4)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get("weight_decay", 1e-2),
        )
        self.criterion = DeepSupervisionLoss()

        # Training State
        self.best_dice = 0.0
        self.start_epoch = 0
        self.working_dir = config.get("working_dir", "./working/idea_5")
        os.makedirs(self.working_dir, exist_ok=True)
        self.save_path = os.path.join(self.working_dir, "best_model.pth")

        # Progressive Resizing Config
        self.num_epochs = config.get("num_epochs", 20)
        self.resize_epoch = int(self.num_epochs * 0.6)  # Switch at 60%
        self.phase1_size = config.get("tile_size", 512)
        self.phase2_size = 768  # Higher resolution for fine-tuning

        # Scheduler (Initialized for Phase 1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.resize_epoch, eta_min=1e-6
        )

    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=True,
            drop_last=shuffle,
        )

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns list of outputs)
            outputs = self.model(images)

            # Calculate loss
            loss = self.criterion(outputs, masks)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        dice_scores = []

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                # Use the finest resolution output (index 0) for evaluation
                pred = outputs[0]
                pred = torch.sigmoid(pred)

                # Binarize for Dice calculation
                pred_binary = (pred > 0.5).float()

                # Calculate Dice
                intersection = (pred_binary * masks).sum(dim=(2, 3))
                union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2.0 * intersection) / (union + 1e-7)

                dice_scores.extend(dice.cpu().numpy().tolist())

        return np.mean(dice_scores)

    def fit(self):
        print(f"Starting training on device: {self.device}")

        # Initial DataLoaders (Phase 1)
        batch_size = self.config.get("batch_size", 8)
        train_loader = self.get_dataloader(self.train_dataset, batch_size, shuffle=True)
        val_loader = self.get_dataloader(self.val_dataset, batch_size, shuffle=False)

        patience = self.config.get("patience", 5)
        patience_counter = 0

        for epoch in range(self.start_epoch, self.num_epochs):
            # --- Progressive Resizing Logic ---
            if epoch == self.resize_epoch:
                print(
                    f"\n[Progressive Resizing] Switching to resolution {self.phase2_size}x{self.phase2_size}"
                )

                # Update Datasets
                self.train_dataset.update_resolution(self.phase2_size)
                self.val_dataset.update_resolution(self.phase2_size)

                # Reduce batch size for larger images to avoid OOM
                new_batch_size = max(1, batch_size // 2)
                print(f"Reducing batch size from {batch_size} to {new_batch_size}")

                # Re-create DataLoaders
                train_loader = self.get_dataloader(
                    self.train_dataset, new_batch_size, shuffle=True
                )
                val_loader = self.get_dataloader(
                    self.val_dataset, new_batch_size, shuffle=False
                )

                # Reset Scheduler for Phase 2
                remaining_epochs = self.num_epochs - self.resize_epoch
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=remaining_epochs, eta_min=1e-6
                )

            # --- Training ---
            start_time = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_dice = self.evaluate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Time: {duration:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Dice: {val_dice:.16f}"
            )

            # --- Checkpointing & Early Stopping ---
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved to {self.save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val Dice: {self.best_dice:.16f}")
