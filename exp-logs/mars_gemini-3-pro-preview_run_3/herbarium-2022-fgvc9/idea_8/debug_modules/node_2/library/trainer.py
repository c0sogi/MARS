import os
import torch
import numpy as np
from library.utils import get_logger, calculate_macro_f1, seed_everything
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss


class Trainer:
    """
    Trainer class for the Hierarchical Plant Classification task.
    Manages model initialization, training loops, validation, and checkpointing.
    """

    def __init__(self, config, train_loader, val_loader, hierarchy_info):
        """
        Args:
            config (dict): Dictionary containing hyperparameters and configuration.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            hierarchy_info (dict): Dictionary containing hierarchy counts (num_genera, num_families).
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hierarchy_info = hierarchy_info

        self.logger = get_logger("Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract configuration
        self.num_species = config.get("num_species", 15501)
        self.num_genera = hierarchy_info["num_genera"]
        self.num_families = hierarchy_info["num_families"]

        self.epochs = config.get("epochs", 20)
        self.lr = config.get("lr", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.patience = config.get("patience", 5)
        self.checkpoint_dir = config.get("checkpoint_dir", "./working/idea_8")

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize Model
        self.model = HierarchicalEfficientNet(
            num_species=self.num_species,
            num_genera=self.num_genera,
            num_families=self.num_families,
            pretrained=True,
        ).to(self.device)

        # Initialize Loss
        self.criterion = HierarchicalLoss(
            genus_weight=config.get("genus_weight", 0.1),
            family_weight=config.get("family_weight", 0.1),
            label_smoothing=config.get("label_smoothing", 0.1),
        )

        # Initialize Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Initialize Scheduler
        # Steps per epoch is required for OneCycleLR
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        self.best_f1 = -1.0

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (images, species_ids, genus_ids, family_ids) in enumerate(
            self.train_loader
        ):
            images = images.to(self.device)
            species_ids = species_ids.to(self.device)
            genus_ids = genus_ids.to(self.device)
            family_ids = family_ids.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, (species_ids, genus_ids, family_ids))

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Returns:
            avg_loss (float): Average validation loss.
            macro_f1 (float): Macro F1 score on species predictions.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, species_ids, genus_ids, family_ids in self.val_loader:
                images = images.to(self.device)
                species_ids = species_ids.to(self.device)
                genus_ids = genus_ids.to(self.device)
                family_ids = family_ids.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, (species_ids, genus_ids, family_ids))

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Get predictions for the primary task (Species)
                species_logits = outputs["species"]
                preds = torch.argmax(species_logits, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(species_ids.cpu().numpy())

        avg_loss = running_loss / count if count > 0 else 0.0

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            macro_f1 = calculate_macro_f1(all_preds, all_targets)
        else:
            macro_f1 = 0.0

        return avg_loss, macro_f1

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        self.logger.info("Starting training...")

        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_f1 = self.validate()

            # Print metrics with full precision
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
            )

            # Checkpoint and Early Stopping
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                patience_counter = 0

                save_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f"New best model saved with F1: {val_f1}")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch} epochs. Best F1: {self.best_f1}"
                )
                break

        self.logger.info(f"Training complete. Final Best F1: {self.best_f1}")
