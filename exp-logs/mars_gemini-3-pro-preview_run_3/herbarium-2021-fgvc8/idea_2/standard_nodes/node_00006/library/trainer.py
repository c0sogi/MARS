import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, update_bn
from library.config import Config
from library.utils import calculate_metric
from library.network import PlantClassifier


class Trainer:
    """
    Trainer class encapsulating training, validation, SWA logic, and submission generation.
    """

    def __init__(self, model=None):
        self.device = torch.device(Config.DEVICE)

        # Initialize model
        if model is None:
            self.model = PlantClassifier(pretrained=Config.PRETRAINED)
        else:
            self.model = model

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=Config.USE_AMP)

        # SWA Setup
        self.use_swa = Config.USE_SWA
        self.swa_start = Config.SWA_START_EPOCH
        self.swa_model = None
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)

        self.working_dir = Config.WORKING_DIR
        self.best_score = -1.0

    def train_one_epoch(self, train_loader, epoch):
        """Run one epoch of training."""
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            with autocast(enabled=Config.USE_AMP):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader, model_to_eval=None):
        """Run validation and calculate Macro F1."""
        if model_to_eval is None:
            model_to_eval = self.model

        model_to_eval.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=Config.USE_AMP):
                    outputs = model_to_eval(images)

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if len(all_preds) == 0:
            return 0.0

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return calculate_metric(all_labels, all_preds)

    def fit(self, train_loader, val_loader, patience=3):
        """
        Execute the training pipeline with SWA and Early Stopping.
        """
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")

        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            # Adjust LR for SWA
            if self.use_swa and epoch == self.swa_start:
                print(f"SWA Starting at Epoch {epoch}. Adjusting LR to {Config.SWA_LR}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = Config.SWA_LR

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Update SWA
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)

            # Validate (Standard Model)
            val_score = self.validate(val_loader, self.model)

            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val F1: {val_score}"
            )

            # Checkpoint Best Model
            if val_score > self.best_score:
                self.best_score = val_score
                patience_counter = 0
                save_path = os.path.join(self.working_dir, "model_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  New best model saved to {save_path}")
            else:
                patience_counter += 1

            # Early Stopping
            # We skip early stopping if we are in the SWA phase to ensure SWA collects enough models
            if patience_counter >= patience:
                if self.use_swa and epoch >= self.swa_start:
                    print(
                        f"  Patience limit reached, but continuing for SWA (Epoch {epoch})"
                    )
                else:
                    print(f"  Early stopping triggered at epoch {epoch}")
                    break

        # Finalize SWA
        if self.use_swa:
            print("Updating SWA BatchNorm statistics (this may take a while)...")
            update_bn(train_loader, self.swa_model, device=self.device)

            swa_save_path = os.path.join(self.working_dir, "model_swa.pth")
            # Save the underlying module state dict for compatibility
            torch.save(self.swa_model.module.state_dict(), swa_save_path)
            print(f"SWA model saved to {swa_save_path}")

            swa_score = self.validate(val_loader, self.swa_model)
            print(f"Final SWA Val F1: {swa_score}")

    def generate_submission(self, test_loader):
        """
        Generate predictions for the test set and save to submission.csv.
        Uses the SWA model if available, otherwise the best standard model.
        """
        # Determine which model to use
        swa_path = os.path.join(self.working_dir, "model_swa.pth")
        best_path = os.path.join(self.working_dir, "model_best.pth")

        inference_model = PlantClassifier(pretrained=False)

        if self.use_swa and os.path.exists(swa_path):
            print("Loading SWA model for prediction...")
            state_dict = torch.load(swa_path, map_location=self.device)
            inference_model.load_state_dict(state_dict)
        elif os.path.exists(best_path):
            print("Loading best standard model for prediction...")
            state_dict = torch.load(best_path, map_location=self.device)
            inference_model.load_state_dict(state_dict)
        else:
            print("Warning: No checkpoint found. Using current model state.")
            inference_model = self.model

        inference_model = inference_model.to(self.device)
        inference_model.eval()

        ids = []
        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(self.device, non_blocking=True)

                with autocast(enabled=Config.USE_AMP):
                    outputs = inference_model(images)

                preds = torch.argmax(outputs, dim=1)

                ids.extend(image_ids.numpy())
                predictions.extend(preds.cpu().numpy())

        # Create submission DataFrame
        df_submission = pd.DataFrame({"Id": ids, "Predicted": predictions})

        # Save to CSV
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
