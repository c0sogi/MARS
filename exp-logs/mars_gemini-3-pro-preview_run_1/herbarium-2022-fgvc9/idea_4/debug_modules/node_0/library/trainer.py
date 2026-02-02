import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

from library.utils import Config, set_seed, get_device, setup_logger, ensure_dirs
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt
from library.loss import HierarchicalLoss, get_class_weights


class Trainer:
    """
    Manages the training, validation, and inference processes for the
    Hierarchical ConvNeXt model.
    """

    def __init__(self):
        self.device = get_device()
        ensure_dirs()
        self.logger = setup_logger(
            "Trainer", os.path.join(Config.WORKING_DIR, "training.log")
        )
        self.best_f1 = 0.0
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, model, loader, optimizer, loss_fn, scaler, epoch):
        """
        Runs one epoch of training with gradient accumulation and mixed precision.
        """
        model.train()
        running_loss = 0.0
        dataset_size = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            # Move data to device
            images = batch["image"].to(self.device)
            targets = {
                "species": batch["species"].to(self.device),
                "genus": batch["genus"].to(self.device),
                "family": batch["family"].to(self.device),
            }
            batch_size = images.size(0)

            # Mixed Precision Forward Pass
            with autocast():
                outputs = model(images)
                loss, _ = loss_fn(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / Config.ACCUMULATION_STEPS

            # Backward Pass
            scaler.scale(loss).backward()

            # Update weights after accumulation steps
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Track metrics (multiply by accumulation steps to get actual loss back)
            running_loss += loss.item() * Config.ACCUMULATION_STEPS * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        self.logger.info(f"Epoch {epoch} Training Loss: {epoch_loss}")
        return epoch_loss

    def validate(self, model, loader, loss_fn):
        """
        Evaluates the model on the validation set.
        Computes Macro F1 score for the Species head.
        """
        model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = {
                    "species": batch["species"].to(self.device),
                    "genus": batch["genus"].to(self.device),
                    "family": batch["family"].to(self.device),
                }
                batch_size = images.size(0)

                # Forward pass
                outputs = model(images)
                loss, _ = loss_fn(outputs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Collect predictions for Macro F1 (Species only)
                preds = torch.argmax(outputs["species"], dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets["species"].cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate Macro F1
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        self.logger.info(f"Validation Loss: {epoch_loss}")
        self.logger.info(f"Validation Macro F1: {macro_f1}")

        return macro_f1, epoch_loss

    def run_training(self):
        """
        Main execution loop for training with SWA strategy.
        """
        set_seed(Config.SEED)

        # 1. Data Loading
        self.logger.info("Loading datasets...")
        train_loader, val_loader, test_loader, maps = get_dataloaders(
            load_cached_data=True
        )

        # 2. Model Initialization
        self.logger.info(f"Initializing model: {Config.MODEL_NAME}")
        model = HierarchicalConvNeXt(pretrained=True)
        model.to(self.device)

        # 3. Loss Function setup
        self.logger.info("Computing class weights...")
        class_weights = get_class_weights(load_cached_data=True)
        loss_fn = HierarchicalLoss(self.device, class_weights=class_weights)

        # 4. Optimizer with Differential Learning Rates
        # Separate backbone parameters from head parameters
        backbone_params = list(model.backbone.parameters())
        head_params = (
            list(model.head_species.parameters())
            + list(model.head_genus.parameters())
            + list(model.head_family.parameters())
        )

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        scaler = GradScaler()

        # 5. Schedulers
        # Standard scheduler for initial phase
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # SWA setup
        swa_model = AveragedModel(model).to(self.device)
        swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

        # 6. Training Loop
        self.logger.info("Starting training loop...")

        for epoch in range(1, Config.EPOCHS + 1):
            self.logger.info(f"--- Epoch {epoch}/{Config.EPOCHS} ---")

            # Train
            self.train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch)

            # SWA Logic
            is_swa_phase = Config.USE_SWA and (epoch >= Config.SWA_START_EPOCH)

            if is_swa_phase:
                self.logger.info("Updating SWA model parameters...")
                swa_model.update_parameters(model)
                swa_scheduler.step()

                # Update BN statistics for SWA model before validation
                self.logger.info("Updating SWA Batch Normalization statistics...")
                update_bn(train_loader, swa_model, device=self.device)

                # Validate SWA model
                val_f1, val_loss = self.validate(swa_model, val_loader, loss_fn)
                current_model_state = swa_model.state_dict()

            else:
                # Standard Phase
                scheduler.step()
                val_f1, val_loss = self.validate(model, val_loader, loss_fn)
                current_model_state = model.state_dict()

            # Save Best Model
            if val_f1 > self.best_f1:
                self.logger.info(f"New best F1: {val_f1} (Previous: {self.best_f1})")
                self.best_f1 = val_f1
                torch.save(current_model_state, self.best_model_path)
                self.logger.info(f"Saved model to {self.best_model_path}")
            else:
                self.logger.info(f"F1 did not improve (Best: {self.best_f1})")

        self.logger.info("Training completed.")

        # Generate submission with the best model
        self.generate_submission(test_loader, maps)

    def generate_submission(self, test_loader, maps):
        """
        Generates predictions for the test set using Test-Time Augmentation (TTA).
        Saves the result to ./submission/submission.csv.
        """
        self.logger.info("Generating submission...")

        # Load best model
        model = HierarchicalConvNeXt(pretrained=False)
        model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()

        # Prepare mapping: Model Index -> Original Category ID
        idx_to_species_raw = maps["idx_to_species"]
        # JSON keys are strings, convert to int for lookup
        idx_to_species = {int(k): int(v) for k, v in idx_to_species_raw.items()}

        predictions = []
        image_ids = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)

                # TTA: Original + Horizontal Flip
                images_flipped = torch.flip(
                    images, dims=[3]
                )  # Flip width dimension (B, C, H, W)

                # Forward pass for both
                out_orig = model(images)["species"]
                out_flip = model(images_flipped)["species"]

                # Average logits
                avg_logits = (out_orig + out_flip) / 2.0

                # Get predictions
                preds = torch.argmax(avg_logits, dim=1).cpu().numpy()

                predictions.extend(preds)
                image_ids.extend(ids)

        # Map predictions back to original category_ids
        final_preds = [idx_to_species[p] for p in predictions]

        # Create DataFrame
        submission_df = pd.DataFrame({"Id": image_ids, "Predicted": final_preds})

        # Save
        submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        self.logger.info(f"Submission saved to {submission_path}")
