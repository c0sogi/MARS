import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import get_logger
from library.model import CascadedTaxonomicNetwork
from library.loss import HierarchicalLoss

logger = get_logger(__name__)


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the Cascaded Taxonomic Network.
    """

    def __init__(self, meta_counts):
        """
        Initializes the Trainer with model, optimizer, scheduler, and loss function.

        Args:
            meta_counts (dict): Dictionary containing 'num_families', 'num_genera', and 'num_species'.
        """
        self.device = Config.DEVICE
        self.meta_counts = meta_counts
        self.num_species = meta_counts.get("num_species", Config.NUM_CLASSES)
        self.num_families = meta_counts["num_families"]
        self.num_genera = meta_counts["num_genera"]

        # Initialize Model
        self.model = CascadedTaxonomicNetwork(
            num_species=self.num_species,
            num_families=self.num_families,
            num_genera=self.num_genera,
            pretrained=True,
        ).to(self.device)

        # Initialize Loss
        self.criterion = HierarchicalLoss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LR_START,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.LR_MIN
        )

        # Initialize AMP Scaler
        self.scaler = GradScaler()

        # Checkpointing and Early Stopping state
        self.best_f1 = -1.0
        self.best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    def train_one_epoch(self, loader, epoch):
        """
        Executes one epoch of training using Automatic Mixed Precision (AMP).

        Args:
            loader (DataLoader): Training data loader.
            epoch (int): Current epoch number.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        running_acc_sp = 0.0
        total_samples = 0

        for images, (species_labels, genus_labels, family_labels) in loader:
            images = images.to(self.device)
            species_labels = species_labels.to(self.device)
            genus_labels = genus_labels.to(self.device)
            family_labels = family_labels.to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                # Pass labels tuple for ArcFace handling in the model
                outputs = self.model(
                    images, labels=(species_labels, genus_labels, family_labels)
                )
                loss = self.criterion(
                    outputs, (species_labels, genus_labels, family_labels)
                )

            # Scaled Backward Pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            running_loss += loss.item() * batch_size

            # Calculate Species Accuracy
            sp_logits = outputs[0]
            _, preds_sp = torch.max(sp_logits, 1)
            running_acc_sp += torch.sum(preds_sp == species_labels.data)
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_acc_sp.double() / total_samples

        return epoch_loss, epoch_acc.item()

    def validate(self, loader):
        """
        Evaluates the model on the validation set.

        Args:
            loader (DataLoader): Validation data loader.

        Returns:
            tuple: (average_loss, macro_f1_score)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        total_samples = 0

        with torch.no_grad():
            for images, (species_labels, genus_labels, family_labels) in loader:
                images = images.to(self.device)
                species_labels = species_labels.to(self.device)
                genus_labels = genus_labels.to(self.device)
                family_labels = family_labels.to(self.device)
                batch_size = images.size(0)

                # Inference: labels=None ensures ArcFace returns cosine similarity (logits)
                outputs = self.model(images, labels=None)

                # Calculate validation loss
                loss = self.criterion(
                    outputs, (species_labels, genus_labels, family_labels)
                )
                running_loss += loss.item() * batch_size

                # Collect predictions for F1 Score
                sp_logits = outputs[0]
                _, preds = torch.max(sp_logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(species_labels.cpu().numpy())
                total_samples += batch_size

        epoch_loss = running_loss / total_samples
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        return epoch_loss, macro_f1

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS, patience=5):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Maximum number of epochs.
            patience (int): Epochs to wait for improvement before stopping.
        """
        logger.info(
            f"Starting training for {epochs} epochs with patience {patience}..."
        )

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_f1 = self.validate(val_loader)

            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]["lr"]

            # Print metrics with full precision
            logger.info(
                f"Epoch {epoch}: LR={curr_lr} | "
                f"Train Loss={train_loss} | Train Acc={train_acc} | "
                f"Val Loss={val_loss} | Val F1={val_f1}"
            )

            # Checkpointing and Early Stopping
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with F1: {self.best_f1}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        logger.info(f"Training finished. Best Validation F1: {self.best_f1}")

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves them to the submission file.

        Args:
            test_loader (DataLoader): Test data loader.
        """
        # Load the best model weights
        if os.path.exists(self.best_model_path):
            logger.info(
                f"Loading best model from {self.best_model_path} for inference..."
            )
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            logger.warning(
                "Best model not found. Using current model weights for inference."
            )

        self.model.eval()
        predictions = []
        image_ids = []

        logger.info("Generating predictions...")
        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)

                # Forward pass
                outputs = self.model(images, labels=None)
                sp_logits = outputs[0]

                # Get predictions
                _, preds = torch.max(sp_logits, 1)

                predictions.extend(preds.cpu().numpy())
                image_ids.extend(ids)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

        # Ensure Id column is integer for correct sorting, if applicable
        try:
            df_sub["Id"] = df_sub["Id"].astype(int)
            df_sub = df_sub.sort_values("Id")
        except ValueError:
            pass  # Keep as is if IDs are not integers

        # Save to CSV
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
