import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from library.config import Config
from library.utils import get_logger, calculate_log_loss, save_checkpoint

logger = get_logger("engine")


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits and Label Smoothing.
    Target smoothing: y_ls = y * (1 - epsilon) + 0.5 * epsilon
    """

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if self.smoothing > 0:
            with torch.no_grad():
                targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=Config.LABEL_SMOOTHING)

    for batch_idx, (images, angles, targets) in enumerate(dataloader):
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)

        batch_size = images.size(0)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Step scheduler if it's per-iteration (not used in this specific config, but good practice)
    # Note: CosineAnnealingLR and ReduceLROnPlateau are usually stepped per epoch.

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate_tta(model, dataloader, device):
    """
    Evaluates the model using Exhaustive Closed-Group TTA.
    TTA Variants: Original, H-Flip, V-Flip, Rot180.
    """
    model.eval()
    preds_list = []
    targets_list = []
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, angles, targets in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)
            batch_size = images.size(0)

            # --- TTA Generation ---
            # 1. Original
            img_orig = images
            # 2. Horizontal Flip
            img_hflip = torch.flip(images, [3])
            # 3. Vertical Flip
            img_vflip = torch.flip(images, [2])
            # 4. Rotate 180 (H + V flip)
            img_rot180 = torch.flip(images, [2, 3])

            # Stack all variants: (4 * B, C, H, W)
            # Processing in one batch is faster if memory allows
            # If OOM occurs, process sequentially. ResNet18 is small, so stacking is fine.
            combined_images = torch.cat(
                [img_orig, img_hflip, img_vflip, img_rot180], dim=0
            )
            combined_angles = torch.cat([angles] * 4, dim=0)

            # Forward pass
            logits = model(combined_images, combined_angles)
            probs = torch.sigmoid(logits)

            # Unstack and Average
            # probs shape: (4 * B, 1) -> (4, B, 1)
            probs_reshaped = probs.view(4, batch_size, 1)
            avg_probs = torch.mean(probs_reshaped, dim=0)  # (B, 1)

            # Compute Loss on averaged probabilities
            # We need to clamp to avoid log(0) issues if calculating manual log loss,
            # but usually we want to evaluate the loss w.r.t targets.
            # Since we have probs, we can't use BCEWithLogitsLoss directly without logits.
            # However, for metric reporting, we use the sklearn log_loss on the CPU side later.
            # Here we just collect predictions.

            # For the running_loss metric returned by this function, we use the averaged probs.
            # We convert back to logits for numerical stability in BCE or compute manually.
            # Let's use the calculate_log_loss utility for the final scalar.

            preds_list.append(avg_probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            dataset_size += batch_size

    # Concatenate all
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)

    # Calculate final metric
    final_loss = calculate_log_loss(y_true, y_pred)

    return final_loss, y_pred, y_true


def predict_test_tta(model, dataloader, device):
    """
    Generates predictions for the test set using TTA.
    """
    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for images, angles, ids in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            batch_size = images.size(0)

            # TTA Variants
            img_orig = images
            img_hflip = torch.flip(images, [3])
            img_vflip = torch.flip(images, [2])
            img_rot180 = torch.flip(images, [2, 3])

            combined_images = torch.cat(
                [img_orig, img_hflip, img_vflip, img_rot180], dim=0
            )
            combined_angles = torch.cat([angles] * 4, dim=0)

            logits = model(combined_images, combined_angles)
            probs = torch.sigmoid(logits)

            probs_reshaped = probs.view(4, batch_size, 1)
            avg_probs = torch.mean(probs_reshaped, dim=0)

            preds_list.append(avg_probs.cpu().numpy())
            ids_list.append(ids)

    return np.concatenate(ids_list), np.concatenate(preds_list, axis=0)


class SWAHandler:
    """
    Manages Stochastic Weight Averaging.
    """

    def __init__(self, model, swa_lr=Config.SWA_LR, device=Config.DEVICE):
        self.device = device
        self.swa_model = AveragedModel(model).to(device)
        self.swa_scheduler = SWALR(
            torch.optim.SGD(model.parameters(), lr=swa_lr), swa_lr=swa_lr
        )
        self.start_epoch = 0

    def update(self, model):
        self.swa_model.update_parameters(model)
        self.swa_scheduler.step()

    def update_bn(self, dataloader):
        """
        Updates Batch Normalization statistics for the SWA model.
        Requires a custom forward pass logic because our model takes (x, angle).
        Standard update_bn only passes x. We need to patch it or run manually.
        """
        # Standard update_bn doesn't support multi-input models easily.
        # We implement a manual BN update loop.
        logger.info("Updating SWA Batch Normalization statistics...")
        self.swa_model.train()
        with torch.no_grad():
            for images, angles, _ in dataloader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                # Forward pass updates running mean/var in BN layers
                _ = self.swa_model(images, angles)

    def get_model(self):
        return self.swa_model


class IcebergTrainer:
    """
    Orchestrates the two-phase training protocol.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LR_INIT, weight_decay=Config.WEIGHT_DECAY
        )

    def fit_phase1_calibration(self, train_loader, val_loader, fold_idx):
        """
        Phase 1: Adaptive Calibration.
        Finds the optimal convergence epoch using ReduceLROnPlateau and Early Stopping.
        """
        logger.info(f"[Fold {fold_idx}] Phase 1: Adaptive Calibration Started")

        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.P1_FACTOR,
            patience=Config.P1_PATIENCE,
            min_lr=Config.P1_MIN_LR,
            verbose=True,
        )

        best_loss = float("inf")
        best_epoch = 0
        early_stop_counter = 0

        # We allow a generous max epoch count, relying on early stopping
        max_epochs = Config.P1_MAX_EPOCHS

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.device, epoch
            )

            val_loss, _, _ = evaluate_tta(self.model, val_loader, self.device)

            scheduler.step(val_loss)

            # Checkpoint and Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                # Save best phase 1 model (optional, mostly for analysis)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "best_loss": best_loss,
                    },
                    is_best=True,
                    checkpoint_dir=Config.CHECKPOINT_DIR,
                    filename=f"phase1_fold{fold_idx}_checkpoint.pth",
                )
            else:
                early_stop_counter += 1

            logger.info(
                f"Ep {epoch}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}, Best={best_loss:.5f} (Ep {best_epoch})"
            )

            if early_stop_counter >= (Config.P1_PATIENCE * 2):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info(
            f"[Fold {fold_idx}] Calibration Complete. Optimal Epochs: {best_epoch}"
        )
        return best_epoch

    def fit_phase2_production(self, full_train_loader, num_epochs, fold_idx):
        """
        Phase 2: Production (Cosine-SWA).
        Trains on full data for num_epochs with CosineAnnealing, then SWA.
        """
        logger.info(
            f"[Fold {fold_idx}] Phase 2: Production Training Started (Target Epochs: {num_epochs})"
        )

        # Reset Optimizer for Phase 2
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LR_INIT, weight_decay=Config.WEIGHT_DECAY
        )

        # Cosine Schedule mapped to the calibrated epoch count
        # We set T_max to num_epochs and eta_min to SWA_LR to ensure smooth transition
        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=Config.SWA_LR
        )

        # Initialize SWA Handler
        swa_handler = SWAHandler(self.model, swa_lr=Config.SWA_LR, device=self.device)

        # 1. Standard Training with Cosine Schedule
        for epoch in range(1, num_epochs + 1):
            loss = train_one_epoch(
                self.model, full_train_loader, self.optimizer, self.device, epoch
            )
            scheduler.step()
            logger.info(
                f"Ep {epoch}/{num_epochs} (Cosine): Loss={loss:.5f}, LR={scheduler.get_last_lr()[0]:.2e}"
            )

        # 2. SWA Phase
        logger.info(
            f"[Fold {fold_idx}] Entering SWA Phase for {Config.SWA_EPOCHS} epochs..."
        )
        for i in range(Config.SWA_EPOCHS):
            swa_epoch = num_epochs + i + 1
            loss = train_one_epoch(
                self.model, full_train_loader, self.optimizer, self.device, swa_epoch
            )
            swa_handler.update(self.model)
            logger.info(f"Ep {swa_epoch} (SWA): Loss={loss:.5f}")

        # 3. Update BN
        swa_handler.update_bn(full_train_loader)

        # 4. Save Final SWA Model
        final_swa_model = swa_handler.get_model()
        save_checkpoint(
            {"state_dict": final_swa_model.state_dict(), "fold": fold_idx},
            is_best=False,
            checkpoint_dir=Config.CHECKPOINT_DIR,
            filename=f"model_fold{fold_idx}_swa.pth",
        )

        return final_swa_model
