import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_iou_map
from library.model import ResNeXtUNet
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass (Model outputs logits)
        logits = model(images)

        # Compute loss
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using mAP metric.
    Unpads images to original size (101x101) for accurate metric calculation.
    """
    model.eval()
    all_preds = []
    all_masks = []

    # Calculate slicing indices for unpadding (128 -> 101)
    # Padding logic in utils.py: top = (128-101)//2 = 13
    pad_top = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    pad_left = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    h_orig = Config.ORIG_SIZE
    w_orig = Config.ORIG_SIZE

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)

            # Squeeze channel dim if present for slicing convenience
            if preds.dim() == 4:
                preds = preds.squeeze(1)
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            # Unpad / Crop to central 101x101 region
            # This ensures we validate on the actual target area, ignoring padding artifacts
            preds_cropped = preds[
                :, pad_top : pad_top + h_orig, pad_left : pad_left + w_orig
            ]
            masks_cropped = masks[
                :, pad_top : pad_top + h_orig, pad_left : pad_left + w_orig
            ]

            all_preds.append(preds_cropped.cpu())
            all_masks.append(masks_cropped.cpu())

    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate mAP
    score = calculate_iou_map(all_preds, all_masks)
    return score


class Trainer:
    def __init__(
        self, debug=False, max_samples=None, epochs_stage1=None, epochs_stage2=None
    ):
        """
        Initializes the Trainer.

        Args:
            debug (bool): If True, runs in debug mode with small data.
            max_samples (int): Limit number of samples for debugging.
            epochs_stage1 (int): Override number of epochs for Stage 1.
            epochs_stage2 (int): Override number of epochs for Stage 2.
        """
        set_seed(Config.SEED)
        self.device = Config.DEVICE
        self.debug = debug

        # Override Config values if provided
        if max_samples is not None:
            Config.MAX_SAMPLES = max_samples

        self.epochs_stage1 = (
            epochs_stage1 if epochs_stage1 is not None else Config.EPOCHS_STAGE1
        )
        self.epochs_stage2 = (
            epochs_stage2 if epochs_stage2 is not None else Config.EPOCHS_STAGE2
        )

        # Initialize DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=debug, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )

        # Initialize Model
        self.model = ResNeXtUNet().to(self.device)

        # Setup Checkpoint Directory
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def run(self):
        """
        Executes the two-stage training pipeline.
        """
        # ==========================
        # Stage 1: BCE + Dice Loss
        # ==========================
        print(
            f"\n=== Starting Stage 1 (BCE + Dice) for {self.epochs_stage1} epochs ==="
        )

        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=False
        )
        criterion = BCEDiceLoss()

        best_score = 0.0
        patience_counter = 0

        for epoch in range(1, self.epochs_stage1 + 1):
            train_loss = train_one_epoch(
                self.model, self.train_loader, optimizer, criterion, self.device
            )
            val_score = validate(self.model, self.val_loader, self.device)

            # Step scheduler based on validation score
            scheduler.step(val_score)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Stage 1 | Epoch {epoch}/{self.epochs_stage1} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val mAP: {val_score:.6f}"
            )

            if val_score > best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered in Stage 1 at epoch {epoch}.")
                break

        # ==========================
        # Stage 2: Lovasz-Hinge Loss
        # ==========================
        print(
            f"\n=== Starting Stage 2 (Lovasz-Hinge) for {self.epochs_stage2} epochs ==="
        )

        # Load best weights from Stage 1 to ensure we start fine-tuning from the best point
        if os.path.exists(self.best_model_path):
            print("Loading best model from Stage 1...")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print(
                "Warning: No checkpoint found from Stage 1. Continuing with current weights."
            )

        # Re-initialize optimizer for fine-tuning
        # We use a slightly lower learning rate (0.5x) for the Lovasz optimization
        finetune_lr = Config.LEARNING_RATE * 0.5
        optimizer = optim.Adam(self.model.parameters(), lr=finetune_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5, verbose=False
        )
        criterion = LovaszHingeLoss()

        # Reset patience counter for Stage 2
        patience_counter = 0
        # Note: We keep best_score from Stage 1 to ensure we only save if Stage 2 improves the model

        for epoch in range(1, self.epochs_stage2 + 1):
            train_loss = train_one_epoch(
                self.model, self.train_loader, optimizer, criterion, self.device
            )
            val_score = validate(self.model, self.val_loader, self.device)

            scheduler.step(val_score)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Stage 2 | Epoch {epoch}/{self.epochs_stage2} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val mAP: {val_score:.6f}"
            )

            if val_score > best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered in Stage 2 at epoch {epoch}.")
                break

        print(f"\nTraining Complete. Best Validation mAP: {best_score}")
        return best_score
