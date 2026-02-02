import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.utils import CFG, seed_everything, dice_coef
from library.model import UNetPlusPlus
from library.losses import DeepSupervisionLoss


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(self, patience=5, delta=0, path="checkpoint.pth", verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(
                f"Validation score improved ({self.val_score_max} --> {val_score}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score


class Trainer:
    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else CFG.device

        # Ensure reproducibility
        seed_everything(CFG.seed)

        # Initialize Model
        self.model = UNetPlusPlus(
            backbone_name=CFG.backbone,
            in_channels=3,
            classes=CFG.num_classes,
            pretrained=True,
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=CFG.T_0, T_mult=CFG.T_mult, eta_min=CFG.min_lr
        )

        # Loss Function
        # We give slightly more weight to the final output (index 0 in the list passed to loss,
        # but index 0 in the list returned by model is actually the deepest/final one?
        # Checking model.py: return [out4, out3, out2, out1]. out4 is the final refined output.
        # DeepSupervisionLoss expects a list. We can weight them.
        # Let's assign weights: [1.0, 0.5, 0.25, 0.125] for [out4, out3, out2, out1]
        self.criterion = DeepSupervisionLoss(weights=[1.0, 0.5, 0.25, 0.125])

        # Directories
        self.working_dir = CFG.cache_dir
        os.makedirs(self.working_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.working_dir, "best_model.pth")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, data in enumerate(self.train_loader):
            images = data["image"].to(self.device, dtype=torch.float32)
            masks = data["mask"].to(self.device, dtype=torch.float32)

            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Forward pass
            # Returns list of tensors [out4, out3, out2, out1]
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        dataset_size = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                images = data["image"].to(self.device, dtype=torch.float32)
                masks = data["mask"].to(self.device, dtype=torch.float32)

                batch_size = images.size(0)

                # Forward pass
                # Returns single tensor out4
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)

                running_loss += loss.item() * batch_size

                # Compute Dice Metric
                # Apply sigmoid
                preds = torch.sigmoid(outputs)
                # Threshold
                preds = (preds > CFG.threshold).float()

                # Calculate Dice for this batch
                # We calculate per image or per batch? dice_coef function flattens inputs.
                # It calculates global dice over the batch if passed directly.
                # To be precise, we often want mean dice over samples, but global dice is also common.
                # Given dice_coef implementation flattens everything, it computes 'Global Dice'.
                # We will stick to that for the batch, then average over batches weighted by size.
                batch_dice = dice_coef(masks.cpu().numpy(), preds.cpu().numpy())
                running_dice += batch_dice * batch_size

                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size
        return epoch_loss, epoch_dice

    def fit(self, epochs=CFG.epochs, patience=5):
        print(f"Starting training on device: {self.device}")

        early_stopping = EarlyStopping(
            patience=patience, verbose=False, path=self.checkpoint_path
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_dice = self.validate()

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            end_time = time.time()
            epoch_mins = int((end_time - start_time) / 60)
            epoch_secs = int((end_time - start_time) % 60)

            # Print metrics with full precision
            print(f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss}")
            print(f"\tVal Loss: {val_loss}")
            print(f"\tVal Dice: {val_dice}")

            # Early Stopping check
            early_stopping(val_dice, self.model)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        print(f"Training complete. Best model saved to {self.checkpoint_path}")
