import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.dataset import get_dataloaders
from library.model import MultiTaskEfficientNet
from library.loss import StableFocalLoss


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_pf1(preds, targets):
    """
    Computes Probabilistic F1 Score (pF1).
    Args:
        preds: Array of predicted probabilities.
        targets: Array of binary ground truth labels.
    """
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    # pTP = Sum(p_i * y_i)
    pTP = np.sum(preds * targets)

    # pFP = Sum(p_i * (1 - y_i))
    pFP = np.sum(preds * (1 - targets))

    # Recall Denominator = TP + FN = Sum(y_i)
    total_positives = np.sum(targets)

    # Precision Denominator = pTP + pFP = Sum(p_i)
    precision_denom = pTP + pFP

    p_precision = pTP / precision_denom if precision_denom > 0 else 0.0
    p_recall = pTP / total_positives if total_positives > 0 else 0.0

    if p_precision + p_recall == 0:
        return 0.0

    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall)
    return pf1


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = MultiTaskEfficientNet(pretrained=True).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        self.cancer_criterion = StableFocalLoss()
        # ignore_index=-1 handles missing aux labels (set in dataset.py)
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Mixed Precision Scaler
        self.scaler = GradScaler()

    def train(self):
        set_seed(Config.SEED)

        # Load Data
        print("Loading data...")
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

        if not train_loader:
            print("Error: Train loader is None. Check metadata generation.")
            return

        # Scheduler Configuration
        # Calculate steps per epoch based on gradient accumulation
        steps_per_epoch = len(train_loader) // Config.GRAD_ACCUMULATION_STEPS
        total_steps = steps_per_epoch * Config.NUM_EPOCHS

        print(
            f"Training for {Config.NUM_EPOCHS} epochs. Steps per epoch: {steps_per_epoch}"
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            epoch_start = time.time()

            # --- Training Phase ---
            self.model.train()
            train_loss_accum = 0.0

            self.optimizer.zero_grad()

            for i, (images, meta, targets) in enumerate(train_loader):
                images = images.to(self.device)
                meta = meta.to(self.device)

                t_cancer = targets["cancer"].to(self.device)
                t_birads = targets["birads"].to(self.device)
                t_density = targets["density"].to(self.device)

                # Mixed Precision Forward Pass
                with autocast():
                    outputs = self.model(images, meta)

                    # Compute Losses
                    loss_c = self.cancer_criterion(outputs["cancer"], t_cancer)
                    loss_b = 0.0
                    loss_d = 0.0

                    if Config.USE_AUX_HEADS:
                        loss_b = self.aux_criterion(outputs["birads"], t_birads)
                        loss_d = self.aux_criterion(outputs["density"], t_density)

                    total_loss = loss_c + loss_b + loss_d

                    # Scale loss for gradient accumulation
                    loss_scaled = total_loss / Config.GRAD_ACCUMULATION_STEPS

                # Backward Pass
                self.scaler.scale(loss_scaled).backward()

                # Optimization Step (every N batches)
                if (i + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), Config.MAX_GRAD_NORM
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                train_loss_accum += total_loss.item()

            avg_train_loss = train_loss_accum / len(train_loader)

            # --- Validation Phase ---
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["loss"]
            val_pf1 = val_metrics["pf1"]

            epoch_time = time.time() - epoch_start

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1} | Time: {epoch_time:.1f}s | "
                f"Train Loss: {avg_train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | Val pF1: {val_pf1:.8f}"
            )

            # --- Model Checkpointing & Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"  -> Saved Best Model (Loss: {val_loss:.8f})")
            else:
                patience_counter += 1
                print(
                    f"  -> Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def validate(self, loader):
        self.model.eval()
        loss_accum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, meta, targets in loader:
                images = images.to(self.device)
                meta = meta.to(self.device)

                t_cancer = targets["cancer"].to(self.device)
                t_birads = targets["birads"].to(self.device)
                t_density = targets["density"].to(self.device)

                with autocast():
                    outputs = self.model(images, meta)

                    loss_c = self.cancer_criterion(outputs["cancer"], t_cancer)
                    loss_b = 0.0
                    loss_d = 0.0

                    if Config.USE_AUX_HEADS:
                        loss_b = self.aux_criterion(outputs["birads"], t_birads)
                        loss_d = self.aux_criterion(outputs["density"], t_density)

                    total_loss = loss_c + loss_b + loss_d

                loss_accum += total_loss.item()

                # Collect predictions for pF1
                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs["cancer"]).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(t_cancer.cpu().numpy())

        avg_loss = loss_accum / len(loader)
        pf1 = calculate_pf1(all_preds, all_targets)

        return {"loss": avg_loss, "pf1": pf1}


def run_training():
    """Helper function to run the training process."""
    trainer = Trainer()
    trainer.train()
