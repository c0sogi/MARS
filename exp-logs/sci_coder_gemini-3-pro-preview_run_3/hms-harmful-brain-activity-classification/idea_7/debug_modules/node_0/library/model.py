import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import AverageMeter, kl_divergence_score, get_logger

# Initialize logger
logger = get_logger(os.path.join(Config.WORKING_DIR, "training.log"))


class DualScaleSpectrogramNet(nn.Module):
    """
    Dual-Scale Spectrogram Fusion Network.

    Stream A: High-Resolution Focus Encoder (50s EEG Window)
              Input: (Batch, 19, 128, 512) -> Adapter -> EfficientNet-B2
    Stream B: Long-Term Context Encoder (10m Spectrogram Window)
              Input: (Batch, 4, 256, 256) -> Adapter -> EfficientNet-B0
    Fusion:   Concatenation -> Dropout -> Dense -> Softmax
    """

    def __init__(self, config=Config):
        super().__init__()

        # ==========================
        # Stream A: EEG (High-Res)
        # ==========================
        # Adapter: Project 19 EEG channels to 3 channels for ImageNet backbone
        self.adapter_eeg = nn.Conv2d(
            config.IN_CHANNELS_EEG, 3, kernel_size=1, bias=False
        )

        # Backbone: EfficientNet-B2 (Pretrained)
        self.backbone_eeg = timm.create_model(
            config.BACKBONE_EEG,
            pretrained=config.PRETRAINED,
            num_classes=0,  # Return pooled features
            in_chans=3,
            global_pool="avg",
        )

        # ==========================
        # Stream B: Spec (Context)
        # ==========================
        # Adapter: Project 4 Spectrogram regions to 3 channels
        self.adapter_spec = nn.Conv2d(
            config.IN_CHANNELS_SPEC, 3, kernel_size=1, bias=False
        )

        # Backbone: EfficientNet-B0 (Pretrained)
        self.backbone_spec = timm.create_model(
            config.BACKBONE_SPEC,
            pretrained=config.PRETRAINED,
            num_classes=0,  # Return pooled features
            in_chans=3,
            global_pool="avg",
        )

        # ==========================
        # Fusion Head
        # ==========================
        # Calculate feature dimension dynamically
        self.num_features = (
            self.backbone_eeg.num_features + self.backbone_spec.num_features
        )

        self.head = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.num_features, config.NUM_CLASSES),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Args:
            x (tuple): ((Batch, 19, 128, 512), (Batch, 4, 256, 256))
        Returns:
            torch.Tensor: Probabilities (Batch, 6)
        """
        x_eeg, x_spec = x

        # Stream A Processing
        x_eeg = self.adapter_eeg(x_eeg)
        feat_eeg = self.backbone_eeg(x_eeg)  # (Batch, 1408)

        # Stream B Processing
        x_spec = self.adapter_spec(x_spec)
        feat_spec = self.backbone_spec(x_spec)  # (Batch, 1280)

        # Fusion
        combined = torch.cat([feat_eeg, feat_spec], dim=1)
        output = self.head(combined)

        return output


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using MixUp augmentation if enabled.
    """
    model.train()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move data to device
        x_eeg = inputs[0].to(device, non_blocking=True)
        x_spec = inputs[1].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # MixUp Augmentation
        if Config.USE_MIXUP and np.random.random() < 0.5:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(x_eeg.size(0)).to(device)

            x_eeg = lam * x_eeg + (1 - lam) * x_eeg[index]
            x_spec = lam * x_spec + (1 - lam) * x_spec[index]
            targets_a, targets_b = targets, targets[index]

            # Forward pass
            outputs = model((x_eeg, x_spec))

            # Loss calculation (KLDivLoss expects LogProbs)
            log_outputs = torch.log(outputs + 1e-15)
            loss = lam * criterion(log_outputs, targets_a) + (1 - lam) * criterion(
                log_outputs, targets_b
            )
        else:
            # Standard Forward pass
            outputs = model((x_eeg, x_spec))
            log_outputs = torch.log(outputs + 1e-15)
            loss = criterion(log_outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), x_eeg.size(0))
        with torch.no_grad():
            # Calculate KL on original targets for monitoring
            kl = kl_divergence_score(targets, outputs)
            kl_meter.update(kl, x_eeg.size(0))

    return loss_meter.avg, kl_meter.avg


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    with torch.no_grad():
        for inputs, targets in loader:
            x_eeg = inputs[0].to(device, non_blocking=True)
            x_spec = inputs[1].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model((x_eeg, x_spec))

            log_outputs = torch.log(outputs + 1e-15)
            loss = criterion(log_outputs, targets)

            loss_meter.update(loss.item(), x_eeg.size(0))
            kl = kl_divergence_score(targets, outputs)
            kl_meter.update(kl, x_eeg.size(0))

    return loss_meter.avg, kl_meter.avg


def run_training(model, train_loader, val_loader, config):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    device = config.DEVICE
    model.to(device)

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # Loss: KL Divergence (expects log-probs input)
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_kl = float("inf")
    patience_counter = 0

    logger.info(f"Starting training on device: {device}")

    for epoch in range(config.EPOCHS):
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS}")

        train_loss, train_kl = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_kl = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.6f} | Train KL: {train_kl:.6f}")
        logger.info(f"  Val Loss:   {val_loss:.6f} | Val KL:   {val_kl:.6f}")

        # Checkpointing
        if val_kl < best_kl:
            best_kl = val_kl
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            logger.info(f"  New Best Model Saved! KL: {best_kl:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"Training Complete. Best Val KL: {best_kl:.6f}")


def generate_submission(model, test_loader, test_df, config):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = config.DEVICE
    model.to(device)

    # Load best model weights
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        logger.info(f"Loaded model from {config.MODEL_PATH} for inference.")
    else:
        logger.warning("Model path not found! Using initialized weights (random).")

    model.eval()
    preds = []

    logger.info("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            # inputs is tuple (x_eeg, x_spec)
            x_eeg = inputs[0].to(device)
            x_spec = inputs[1].to(device)

            outputs = model((x_eeg, x_spec))
            preds.append(outputs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Prepare Submission DataFrame
    # Map prob columns to vote columns
    sub_cols = [c.replace("_prob", "_vote") for c in config.TARGET_COLS]

    sub_df = pd.DataFrame(preds, columns=sub_cols)
    sub_df["eeg_id"] = test_df["eeg_id"].values

    # Reorder to match submission format
    cols = ["eeg_id"] + sub_cols
    sub_df = sub_df[cols]

    sub_df.to_csv(config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_FILE}")
