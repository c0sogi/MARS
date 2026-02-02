import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from library.utils import set_seed, get_device, AverageMeter, Logger
from library.losses import FocalLoss


class Trainer:
    """
    Trainer class for Hierarchical Multi-Task Learning.
    Manages training, validation, early stopping, and submission generation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        label_map,
        mixup_alpha=0.2,
        aux_weight_genus=1.0,
        aux_weight_family=1.0,
        save_dir="./working/demo_run",
    ):
        """
        Args:
            model (nn.Module): The hierarchical classification model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): PyTorch optimizer.
            scheduler (LRScheduler): PyTorch learning rate scheduler.
            device (torch.device): Device to run training on.
            label_map (dict): Mapping from category_id to contiguous index.
            mixup_alpha (float): Alpha parameter for Beta distribution in Mixup.
            aux_weight_genus (float): Weight for Genus head loss.
            aux_weight_family (float): Weight for Family head loss.
            save_dir (str): Directory to save checkpoints and logs.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.label_map = label_map
        self.mixup_alpha = mixup_alpha
        self.aux_weight_genus = aux_weight_genus
        self.aux_weight_family = aux_weight_family
        self.save_dir = save_dir

        # Invert label map for submission (Index -> Category ID)
        self.idx_to_category = {v: k for k, v in label_map.items()}

        # Define Loss Functions
        # Species head uses Focal Loss to handle imbalance
        self.species_criterion = FocalLoss()
        # Auxiliary heads use standard Cross Entropy
        self.aux_criterion = nn.CrossEntropyLoss()

        # Setup logging
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = Logger(os.path.join(self.save_dir, "train.log"))
        self.best_model_path = os.path.join(self.save_dir, "best_model.pth")

    def mixup_data(self, x, y_s, y_g, y_f):
        """
        Performs Mixup augmentation on inputs and hierarchical labels.
        """
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]

        # Return pairs of targets
        y_s_a, y_s_b = y_s, y_s[index]
        y_g_a, y_g_b = y_g, y_g[index]
        y_f_a, y_f_b = y_f, y_f[index]

        return mixed_x, y_s_a, y_s_b, y_g_a, y_g_b, y_f_a, y_f_b, lam

    def mixup_criterion(
        self, pred_s, pred_g, pred_f, y_s_a, y_s_b, y_g_a, y_g_b, y_f_a, y_f_b, lam
    ):
        """
        Calculates the compound loss for mixed targets.
        Loss = L_species + w_g * L_genus + w_f * L_family
        """
        # Loss for set A
        loss_s_a = self.species_criterion(pred_s, y_s_a)
        loss_g_a = self.aux_criterion(pred_g, y_g_a)
        loss_f_a = self.aux_criterion(pred_f, y_f_a)
        total_a = (
            loss_s_a
            + self.aux_weight_genus * loss_g_a
            + self.aux_weight_family * loss_f_a
        )

        # Loss for set B
        loss_s_b = self.species_criterion(pred_s, y_s_b)
        loss_g_b = self.aux_criterion(pred_g, y_g_b)
        loss_f_b = self.aux_criterion(pred_f, y_f_b)
        total_b = (
            loss_s_b
            + self.aux_weight_genus * loss_g_b
            + self.aux_weight_family * loss_f_b
        )

        return lam * total_a + (1 - lam) * total_b

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for i, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            target_species = batch["species"].to(self.device)
            target_genus = batch["genus"].to(self.device)
            target_family = batch["family"].to(self.device)

            # Apply Mixup
            mixed_images, s_a, s_b, g_a, g_b, f_a, f_b, lam = self.mixup_data(
                images, target_species, target_genus, target_family
            )

            self.optimizer.zero_grad()

            # Forward pass
            pred_species, pred_genus, pred_family = self.model(mixed_images)

            # Calculate loss
            loss = self.mixup_criterion(
                pred_species, pred_genus, pred_family, s_a, s_b, g_a, g_b, f_a, f_b, lam
            )

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                target_species = batch["species"].to(self.device)
                target_genus = batch["genus"].to(self.device)
                target_family = batch["family"].to(self.device)

                # Forward pass
                pred_species, pred_genus, pred_family = self.model(images)

                # Calculate loss (no mixup)
                loss_s = self.species_criterion(pred_species, target_species)
                loss_g = self.aux_criterion(pred_genus, target_genus)
                loss_f = self.aux_criterion(pred_family, target_family)

                total_loss = (
                    loss_s
                    + self.aux_weight_genus * loss_g
                    + self.aux_weight_family * loss_f
                )
                losses.update(total_loss.item(), images.size(0))

                # Store predictions for F1 Score
                preds = torch.argmax(pred_species, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target_species.cpu().numpy())

        # Calculate Macro F1
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        return losses.avg, macro_f1

    def fit(self, num_epochs, patience=5):
        """
        Main training loop with early stopping.
        """
        self.logger.log(f"Starting training on device: {self.device}")
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_f1 = self.validate(epoch)

            # Update Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log metrics
            self.logger.log(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val Macro F1: {val_f1:.9f}"
            )

            # Early Stopping and Checkpointing
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.log(f"New best model saved with F1: {best_f1:.9f}")
            else:
                patience_counter += 1
                self.logger.log(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    self.logger.log("Early stopping triggered.")
                    break

        self.logger.log(f"Training complete. Best F1: {best_f1:.9f}")

    def predict(self, test_loader, output_dir="./submission"):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        self.logger.log("Loading best model for inference...")

        # Load best model weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            self.logger.log(
                "Warning: Best model not found. Using current model weights."
            )

        self.model.eval()
        results = []

        self.logger.log("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                # Test loader returns image and image_id
                images, image_ids = batch
                images = images.to(self.device)

                # Forward pass
                pred_species, _, _ = self.model(images)

                # Get predicted class indices
                pred_indices = torch.argmax(pred_species, dim=1).cpu().numpy()

                # Map indices back to original category_ids
                for img_id, pred_idx in zip(image_ids, pred_indices):
                    category_id = self.idx_to_category.get(pred_idx, 0)
                    results.append({"Id": int(img_id), "Predicted": int(category_id)})

        # Create DataFrame
        df_submission = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        submission_path = os.path.join(output_dir, "submission.csv")

        # Save to CSV
        df_submission.to_csv(submission_path, index=False)
        self.logger.log(f"Submission saved to {submission_path}")
