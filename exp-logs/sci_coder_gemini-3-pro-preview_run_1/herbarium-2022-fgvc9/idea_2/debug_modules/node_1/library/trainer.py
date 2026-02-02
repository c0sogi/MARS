import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from library.utils import compute_loss_weights, seed_everything
from library.model import HierarchicalConvNeXt


class HierarchicalTrainer:
    def __init__(
        self,
        model,
        device,
        num_families,
        num_genera,
        num_species,
        learning_rate_backbone=1e-4,
        learning_rate_head=1e-3,
        lambda_genus=0.5,
        lambda_family=0.5,
        label_smoothing=0.1,
    ):
        """
        Args:
            model (nn.Module): The HierarchicalConvNeXt model.
            device (torch.device): Device to run training on.
            num_families (int): Number of family classes.
            num_genera (int): Number of genus classes.
            num_species (int): Number of species classes.
            learning_rate_backbone (float): LR for the ConvNeXt backbone.
            learning_rate_head (float): LR for the classification heads.
            lambda_genus (float): Loss weight for the genus head.
            lambda_family (float): Loss weight for the family head.
            label_smoothing (float): Label smoothing factor for species loss.
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_genus = lambda_genus
        self.lambda_family = lambda_family

        # 1. Loss Functions
        # Compute class weights for species to handle imbalance
        species_weights = compute_loss_weights(
            num_classes=num_species, load_cached_data=True
        )
        species_weights = species_weights.to(device)

        self.criterion_species = nn.CrossEntropyLoss(
            weight=species_weights, label_smoothing=label_smoothing
        )
        self.criterion_genus = nn.CrossEntropyLoss()
        self.criterion_family = nn.CrossEntropyLoss()

        # 2. Optimizer with Differential Learning Rates
        # Separate parameters into backbone and heads
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": learning_rate_backbone},
                {"params": head_params, "lr": learning_rate_head},
            ]
        )

        self.best_f1 = -1.0
        self.checkpoint_dir = "./working/idea_2/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(self.device)
            target_species = targets["species"].to(self.device)
            target_genus = targets["genus"].to(self.device)
            target_family = targets["family"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Calculate losses
            loss_species = self.criterion_species(outputs["species"], target_species)
            loss_genus = self.criterion_genus(outputs["genus"], target_genus)
            loss_family = self.criterion_family(outputs["family"], target_family)

            # Weighted sum
            total_loss = (
                loss_species
                + (self.lambda_genus * loss_genus)
                + (self.lambda_family * loss_family)
            )

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        running_loss = 0.0

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(self.device)
                target_species = targets["species"].to(self.device)

                # We also compute loss for validation to monitor convergence
                target_genus = targets["genus"].to(self.device)
                target_family = targets["family"].to(self.device)

                outputs = self.model(images)

                # Loss calculation
                loss_species = self.criterion_species(
                    outputs["species"], target_species
                )
                loss_genus = self.criterion_genus(outputs["genus"], target_genus)
                loss_family = self.criterion_family(outputs["family"], target_family)
                total_loss = (
                    loss_species
                    + (self.lambda_genus * loss_genus)
                    + (self.lambda_family * loss_family)
                )

                running_loss += total_loss.item() * images.size(0)

                # Predictions (only species matters for the metric)
                preds = torch.argmax(outputs["species"], dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target_species.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        # Calculate Macro F1 Score
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        return val_loss, macro_f1

    def fit(self, train_loader, val_loader, num_epochs=10, patience=3):
        """
        Main training loop with Early Stopping and Scheduler.
        """
        # Scheduler: Cosine Annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_f1 = self.validate(val_loader)

            # Step Scheduler (strictly after validation)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_f1}"
            )  # Full precision print

            # Early Stopping & Checkpointing
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved with F1: {self.best_f1}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation F1: {self.best_f1}")
