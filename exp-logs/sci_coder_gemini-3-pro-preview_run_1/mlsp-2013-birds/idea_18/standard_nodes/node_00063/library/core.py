import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim.swa_utils import AveragedModel, update_bn
from library.utils import calculate_roc_auc, save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer class for training, validating, and predicting with the bird species classification model.
    Handles standard training loops, SWA (Stochastic Weight Averaging), and Early Stopping.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
        checkpoint_dir="./working/idea_18/checkpoints",
        use_swa=False,
        swa_start_epoch=None,
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
            device (torch.device): Device to run on (cuda/cpu).
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            checkpoint_dir (str): Directory to save checkpoints.
            use_swa (bool): Whether to use Stochastic Weight Averaging.
            swa_start_epoch (int, optional): Epoch to start SWA updates.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch

        self.swa_model = None
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)

        self.best_auc = -1.0
        self.patience_counter = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader, model_to_use=None):
        """
        Evaluates the model on the validation set.
        """
        if model_to_use is None:
            model_to_use = self.model

        model_to_use.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model_to_use(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Apply sigmoid for AUC calculation
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        epoch_loss = running_loss / count if count > 0 else 0.0

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            auc = calculate_roc_auc(all_targets, all_preds)
        else:
            auc = 0.5

        return epoch_loss, auc

    def fit(self, train_loader, val_loader, epochs, patience=10):
        """
        Main training loop with SWA and Early Stopping logic.
        """
        print(f"Starting training for {epochs} epochs on device {self.device}...")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # SWA Update Logic
            is_swa_phase = False
            if (
                self.use_swa
                and self.swa_start_epoch is not None
                and epoch >= self.swa_start_epoch
            ):
                is_swa_phase = True
                self.swa_model.update_parameters(self.model)

            # Validation (Main Model)
            val_loss, val_auc = self.validate(val_loader, self.model)

            # Scheduler Step
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Checkpointing & Early Stopping
            is_best = val_auc > self.best_auc
            if is_best:
                self.best_auc = val_auc
                # Disable early stopping counter reset during SWA phase to strictly follow SWA schedule
                # but still track best model.
                if not is_swa_phase:
                    self.patience_counter = 0

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_auc": self.best_auc,
                    },
                    is_best=True,
                    checkpoint_dir=self.checkpoint_dir,
                    filename="model_last.pth",
                )
            else:
                if not is_swa_phase:
                    self.patience_counter += 1

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_auc": self.best_auc,
                    },
                    is_best=False,
                    checkpoint_dir=self.checkpoint_dir,
                    filename="model_last.pth",
                )

            # Trigger Early Stopping only if NOT in SWA phase
            if not is_swa_phase and self.patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # End of training
        if self.use_swa:
            print("Updating SWA Batch Normalization statistics...")
            update_bn(train_loader, self.swa_model, device=self.device)

            # Save SWA model
            torch.save(
                self.swa_model.module.state_dict(),
                os.path.join(self.checkpoint_dir, "model_swa.pth"),
            )
            print("SWA model saved.")

    def predict(self, test_loader, use_swa_model=False):
        """
        Generates predictions for the test set.
        """
        if use_swa_model and self.swa_model is not None:
            model_to_use = self.swa_model
        else:
            model_to_use = self.model

        model_to_use.eval()

        predictions = []
        ids = []

        with torch.no_grad():
            for images, _, rec_ids in test_loader:
                images = images.to(self.device)
                outputs = model_to_use(images)
                probs = torch.sigmoid(outputs)

                predictions.append(probs.cpu().numpy())
                ids.extend(rec_ids.numpy())

        predictions = np.concatenate(predictions, axis=0)
        return ids, predictions


def generate_submission(ids, predictions, output_path="./submission/submission.csv"):
    """
    Generates the submission CSV in the required format.
    Format:
    Id,Probability
    rec_id * 100 + species_id, probability
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_rows = []
    num_classes = predictions.shape[1]

    for i, rec_id in enumerate(ids):
        probs = predictions[i]
        for species_id in range(num_classes):
            row_id = int(rec_id * 100 + species_id)
            prob = probs[species_id]
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
