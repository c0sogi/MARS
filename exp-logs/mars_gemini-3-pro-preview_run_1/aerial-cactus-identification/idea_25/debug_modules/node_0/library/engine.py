import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger, calculate_roc_auc

logger = get_logger("engine")


def mixup_data(x, y_cls, y_qual, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns:
        mixed_x: The mixed images.
        y_cls_a, y_cls_b: The classification labels for the two mixed images.
        y_qual_a, y_qual_b: The quality targets for the two mixed images.
        lam: The mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_cls_a, y_cls_b = y_cls, y_cls[index]
    y_qual_a, y_qual_b = y_qual, y_qual[index]
    return mixed_x, y_cls_a, y_cls_b, y_qual_a, y_qual_b, lam


class Engine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss Functions
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_reg = nn.MSELoss()

        # SWA Attributes
        self.swa_model = None
        self.swa_start_epoch = Config.SWA_START_EPOCH

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Trains the model for one epoch using Mixup and Multi-Task Loss.
        """
        self.model.train()
        running_loss = 0.0
        running_loss_tex = 0.0
        running_loss_sem = 0.0
        running_loss_qual = 0.0

        dataset_size = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)  # (B, 1)
            quality = batch["quality_target"].to(self.device).unsqueeze(1)  # (B, 1)

            batch_size = images.size(0)
            dataset_size += batch_size

            # Apply Mixup
            mixed_images, lbl_a, lbl_b, qual_a, qual_b, lam = mixup_data(
                images, labels, quality, Config.MIXUP_ALPHA, self.device
            )

            # Forward Pass
            outputs = self.model(mixed_images)

            # --- Calculate Losses ---

            # 1. Texture Head (Classification)
            if Config.USE_TEXTURE_HEAD and outputs["texture"] is not None:
                loss_tex = lam * self.criterion_cls(outputs["texture"], lbl_a) + (
                    1 - lam
                ) * self.criterion_cls(outputs["texture"], lbl_b)
            else:
                loss_tex = torch.tensor(0.0, device=self.device)

            # 2. Semantic Head (Classification)
            if Config.USE_SEMANTIC_HEAD and outputs["semantic"] is not None:
                loss_sem = lam * self.criterion_cls(outputs["semantic"], lbl_a) + (
                    1 - lam
                ) * self.criterion_cls(outputs["semantic"], lbl_b)
            else:
                loss_sem = torch.tensor(0.0, device=self.device)

            # 3. Quality Head (Regression)
            if Config.USE_QUALITY_HEAD and outputs["quality"] is not None:
                loss_qual = lam * self.criterion_reg(outputs["quality"], qual_a) + (
                    1 - lam
                ) * self.criterion_reg(outputs["quality"], qual_b)
            else:
                loss_qual = torch.tensor(0.0, device=self.device)

            # Weighted Sum
            total_loss = (
                (Config.LOSS_WEIGHT_TEXTURE * loss_tex)
                + (Config.LOSS_WEIGHT_SEMANTIC * loss_sem)
                + (Config.LOSS_WEIGHT_QUALITY * loss_qual)
            )

            # Backward Pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Logging
            running_loss += total_loss.item() * batch_size
            running_loss_tex += loss_tex.item() * batch_size
            running_loss_sem += loss_sem.item() * batch_size
            running_loss_qual += loss_qual.item() * batch_size

        return {
            "loss": running_loss / dataset_size,
            "loss_tex": running_loss_tex / dataset_size,
            "loss_sem": running_loss_sem / dataset_size,
            "loss_qual": running_loss_qual / dataset_size,
        }

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Computes ROC AUC for Texture, Semantic, and Ensemble predictions.
        """
        self.model.eval()
        running_loss = 0.0

        all_labels = []
        all_preds_tex = []
        all_preds_sem = []

        dataset_size = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)
                quality = batch["quality_target"].to(self.device).unsqueeze(1)

                batch_size = images.size(0)
                dataset_size += batch_size

                outputs = self.model(images)

                # Calculate Raw Losses (No Mixup)
                loss_tex = torch.tensor(0.0, device=self.device)
                loss_sem = torch.tensor(0.0, device=self.device)
                loss_qual = torch.tensor(0.0, device=self.device)

                if Config.USE_TEXTURE_HEAD and outputs["texture"] is not None:
                    loss_tex = self.criterion_cls(outputs["texture"], labels)
                    all_preds_tex.append(
                        torch.sigmoid(outputs["texture"]).cpu().numpy()
                    )

                if Config.USE_SEMANTIC_HEAD and outputs["semantic"] is not None:
                    loss_sem = self.criterion_cls(outputs["semantic"], labels)
                    all_preds_sem.append(
                        torch.sigmoid(outputs["semantic"]).cpu().numpy()
                    )

                if Config.USE_QUALITY_HEAD and outputs["quality"] is not None:
                    loss_qual = self.criterion_reg(outputs["quality"], quality)

                total_loss = (
                    (Config.LOSS_WEIGHT_TEXTURE * loss_tex)
                    + (Config.LOSS_WEIGHT_SEMANTIC * loss_sem)
                    + (Config.LOSS_WEIGHT_QUALITY * loss_qual)
                )

                running_loss += total_loss.item() * batch_size
                all_labels.append(labels.cpu().numpy())

        # Concatenate results
        all_labels = np.concatenate(all_labels)

        # Calculate Metrics
        auc_tex = 0.5
        auc_sem = 0.5
        auc_ens = 0.5

        if Config.USE_TEXTURE_HEAD and all_preds_tex:
            all_preds_tex = np.concatenate(all_preds_tex)
            auc_tex = calculate_roc_auc(all_labels, all_preds_tex)

        if Config.USE_SEMANTIC_HEAD and all_preds_sem:
            all_preds_sem = np.concatenate(all_preds_sem)
            auc_sem = calculate_roc_auc(all_labels, all_preds_sem)

        # Ensemble Logic
        if Config.USE_TEXTURE_HEAD and Config.USE_SEMANTIC_HEAD:
            all_preds_ens = (all_preds_tex + all_preds_sem) / 2.0
            auc_ens = calculate_roc_auc(all_labels, all_preds_ens)
        elif Config.USE_TEXTURE_HEAD:
            auc_ens = auc_tex
        elif Config.USE_SEMANTIC_HEAD:
            auc_ens = auc_sem

        return {
            "loss": running_loss / dataset_size,
            "auc_tex": auc_tex,
            "auc_sem": auc_sem,
            "auc_ens": auc_ens,
        }

    def fit(self, train_loader, val_loader, fold_idx):
        """
        Main training routine.
        Phase 1: Convergence (AdamW + Cosine Annealing)
        Phase 2: Exploration (SWA with Cyclic LR)
        """
        logger.info(f"Starting training for Fold {fold_idx}")

        best_auc = 0.0
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_fold{fold_idx}.pth"
        )
        swa_model_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_fold{fold_idx}.pth")

        # Initialize SWA Model
        self.swa_model = AveragedModel(self.model)

        for epoch in range(1, Config.TOTAL_EPOCHS + 1):

            # --- LR Scheduling ---
            if epoch > Config.CONVERGENCE_EPOCHS:
                # Manual Cyclic LR for SWA Phase
                swa_epoch_idx = epoch - Config.CONVERGENCE_EPOCHS

                # Determine position in cycle (0 to CYCLE_LEN - 1)
                cycle_pos = (swa_epoch_idx - 1) % Config.SWA_CYCLE_LEN

                # Cosine decay from Max to Min within one cycle
                frac = (
                    cycle_pos / float(Config.SWA_CYCLE_LEN - 1)
                    if Config.SWA_CYCLE_LEN > 1
                    else 1.0
                )
                current_lr = Config.SWA_LR_MIN + 0.5 * (
                    Config.SWA_LR_MAX - Config.SWA_LR_MIN
                ) * (1 + np.cos(np.pi * frac))

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr

            # --- Training ---
            train_metrics = self.train_one_epoch(train_loader, epoch)

            # --- Validation ---
            val_metrics = self.validate(val_loader)

            # --- Logging ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{Config.TOTAL_EPOCHS} [LR: {current_lr:.2e}] - "
                f"Train Loss: {train_metrics['loss']:.6f} "
                f"(T:{train_metrics['loss_tex']:.4f}, S:{train_metrics['loss_sem']:.4f}, Q:{train_metrics['loss_qual']:.4f}) - "
                f"Val Loss: {val_metrics['loss']:.6f} - "
                f"Val AUC Ens: {val_metrics['auc_ens']:.10f}"
            )

            # --- Phase Logic ---
            if epoch <= Config.CONVERGENCE_EPOCHS:
                # Convergence Phase
                if self.scheduler:
                    self.scheduler.step()

                # Save Best Model
                if val_metrics["auc_ens"] > best_auc:
                    best_auc = val_metrics["auc_ens"]
                    torch.save(self.model.state_dict(), best_model_path)
                    logger.info(f"New best model saved with AUC: {best_auc:.10f}")

            else:
                # SWA Phase
                # Update SWA model at the end of each cycle (when LR is lowest)
                cycle_pos = (
                    epoch - Config.CONVERGENCE_EPOCHS - 1
                ) % Config.SWA_CYCLE_LEN
                if cycle_pos == Config.SWA_CYCLE_LEN - 1:
                    logger.info(f"SWA: Updating averaged model at epoch {epoch}")
                    self.swa_model.update_parameters(self.model)

        # --- Finalization ---
        logger.info("SWA: Updating BatchNorm statistics...")
        update_bn(train_loader, self.swa_model, device=self.device)

        # Save SWA Model (saving the averaged parameters)
        torch.save(self.swa_model.state_dict(), swa_model_path)
        logger.info(f"SWA model saved to {swa_model_path}")

        return best_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        Returns:
            ids: List of image IDs.
            probs: List of predicted probabilities (Ensemble of Texture and Semantic).
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                outputs = self.model(images)

                # Ensemble Prediction
                prob_tex = (
                    torch.sigmoid(outputs["texture"])
                    if outputs["texture"] is not None
                    else 0
                )
                prob_sem = (
                    torch.sigmoid(outputs["semantic"])
                    if outputs["semantic"] is not None
                    else 0
                )

                if Config.USE_TEXTURE_HEAD and Config.USE_SEMANTIC_HEAD:
                    prob_ens = (prob_tex + prob_sem) / 2.0
                elif Config.USE_TEXTURE_HEAD:
                    prob_ens = prob_tex
                else:
                    prob_ens = prob_sem

                all_probs.append(prob_ens.cpu().numpy())

        return np.concatenate(all_probs).flatten()
