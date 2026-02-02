import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import calculate_macro_f1, save_checkpoint


class Trainer:
    """
    Trainer class for the Hierarchical Multi-Task Network.
    Manages training, validation, logging, and early stopping.
    """

    def __init__(self, model, train_loader, val_loader, device=Config.DEVICE):
        """
        Args:
            model (nn.Module): The hierarchical model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run training on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()
        self.loss_weights = Config.LOSS_WEIGHTS

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
        )

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Early Stopping & Checkpointing
        self.patience = Config.PATIENCE
        self.best_score = -1.0
        self.counter = 0

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        for i, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)

            # Unpack targets: (species, genus, family)
            species_targets = targets[0].to(self.device)
            genus_targets = targets[1].to(self.device)
            family_targets = targets[2].to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)

                loss_species = self.criterion(outputs["species"], species_targets)
                loss_genus = self.criterion(outputs["genus"], genus_targets)
                loss_family = self.criterion(outputs["family"], family_targets)

                # Weighted Sum of Losses
                total_loss = (
                    self.loss_weights["species"] * loss_species
                    + self.loss_weights["genus"] * loss_genus
                    + self.loss_weights["family"] * loss_family
                )

            # Backward Pass with Scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += total_loss.item()

        return running_loss / num_batches

    def validate(self):
        """
        Runs validation on the validation set.
        Returns:
            tuple: (Average Validation Loss, Macro F1 Score)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)

                species_targets = targets[0].to(self.device)
                genus_targets = targets[1].to(self.device)
                family_targets = targets[2].to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)

                    loss_species = self.criterion(outputs["species"], species_targets)
                    loss_genus = self.criterion(outputs["genus"], genus_targets)
                    loss_family = self.criterion(outputs["family"], family_targets)

                    total_loss = (
                        self.loss_weights["species"] * loss_species
                        + self.loss_weights["genus"] * loss_genus
                        + self.loss_weights["family"] * loss_family
                    )

                running_loss += total_loss.item()

                # Collect predictions for Macro F1 (Species only)
                preds = torch.argmax(outputs["species"], dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(species_targets.cpu())

        avg_loss = running_loss / num_batches

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate Metric
        macro_f1 = calculate_macro_f1(all_targets, all_preds)

        return avg_loss, macro_f1

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop.
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch()

            # Validate
            val_loss, val_f1 = self.validate()

            # Scheduler Step
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Logging
            print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val F1:     {val_f1}")

            # Early Stopping & Checkpointing
            if val_f1 > self.best_score:
                print(
                    f"  Validation F1 improved from {self.best_score} to {val_f1}. Saving model..."
                )
                self.best_score = val_f1
                save_checkpoint(self.model, self.optimizer, epoch, val_f1)
                self.counter = 0
            else:
                self.counter += 1
                print(
                    f"  Validation F1 did not improve. Counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                print("Early stopping triggered.")
                break
