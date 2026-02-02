import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np

from library.config import (
    GeneralConfig,
    PathConfig,
    ModelConfig,
    TrainConfig,
    DataConfig,
)
from library.utils import AverageMeter, calculate_iou_map, rle_encode
from library.losses import BCEDiceLoss, LovaszHingeLoss, DeepSupervisionLoss
from library.dataset import get_loaders, get_test_loader
from library.model import SaltUNetPlusPlus


class ModelTrainer:
    """
    Manages the training, validation, and checkpointing of the Salt Segmentation model.
    """

    def __init__(self, fold_idx, debug=False):
        self.fold_idx = fold_idx
        self.debug = debug
        self.device = torch.device(GeneralConfig.DEVICE)

        # Initialize Data Loaders
        self.train_loader, self.val_loader = get_loaders(fold_idx, debug=debug)

        # Initialize Model
        self.model = SaltUNetPlusPlus()
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TrainConfig.LR,
            weight_decay=TrainConfig.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=TrainConfig.SCHEDULER_FACTOR,
            patience=TrainConfig.SCHEDULER_PATIENCE,
            min_lr=TrainConfig.MIN_LR,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Loss Functions
        # Phase 1: Deep Supervision with BCE + Dice
        self.criterion_warmup = DeepSupervisionLoss(BCEDiceLoss())
        # Phase 2: Lovasz Hinge on final output
        self.criterion_finetune = LovaszHingeLoss()

        # Checkpointing
        self.best_map = 0.0
        self.best_epoch = 0
        self.checkpoint_path = os.path.join(
            PathConfig.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth"
        )

    def train_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        meter = AverageMeter()

        # Determine Training Phase
        if epoch < TrainConfig.WARMUP_EPOCHS:
            phase = "Warmup"
        else:
            phase = "Finetune"

        for batch_idx, (images, masks, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast():
                # Model returns a list of outputs in training mode (Deep Supervision)
                outputs = self.model(images)

                if phase == "Warmup":
                    # Calculate loss on all deep supervision heads
                    loss = self.criterion_warmup(outputs, masks)
                else:
                    # Calculate Lovasz loss only on the final, most refined output
                    # outputs[-1] corresponds to the final decoder output
                    loss = self.criterion_finetune(outputs[-1], masks)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            meter.update(loss.item(), images.size(0))

        return meter.avg

    def validate(self):
        """Runs validation and calculates mAP on original image size."""
        self.model.eval()

        # Calculate cropping indices to revert padding (128 -> 101)
        # Assuming symmetric padding used in dataset
        start_idx = (DataConfig.IMG_H - DataConfig.ORIG_H) // 2
        end_idx = start_idx + DataConfig.ORIG_H

        # Cite solution_lesson_node_00011: Decouple Discriminative Performance from Calibration
        # Accumulate all predictions to perform threshold optimization
        probs_list = []
        masks_list = []

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast():
                    # Model returns single tensor in eval mode
                    output = self.model(images)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(output)

                # Crop predictions and masks to original 101x101 size
                probs_cropped = probs[:, :, start_idx:end_idx, start_idx:end_idx]
                masks_cropped = masks[:, :, start_idx:end_idx, start_idx:end_idx]

                probs_list.append(probs_cropped.cpu().numpy())
                masks_list.append(masks_cropped.cpu().numpy())

        probs_all = np.concatenate(probs_list, axis=0)
        masks_all = np.concatenate(masks_list, axis=0)

        # Sweep thresholds to find the best mAP for this epoch
        best_map = 0.0
        thresholds = np.arange(0.3, 0.76, 0.05)

        for th in thresholds:
            score = calculate_iou_map(probs_all, masks_all, threshold=th)
            if score > best_map:
                best_map = score

        return best_map

    def run(self):
        """Main training loop."""
        print(f"Starting training for Fold {self.fold_idx}...")
        early_stopping_counter = 0

        for epoch in range(TrainConfig.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_map = self.validate()

            # Update Scheduler based on Validation mAP
            self.scheduler.step(val_map)

            elapsed = time.time() - start_time

            # Log Metrics
            print(
                f"Epoch {epoch+1}/{TrainConfig.EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val mAP: {val_map:.8f} | "
                f"Time: {elapsed:.2f}s"
            )

            # Save Best Model
            if val_map > self.best_map:
                self.best_map = val_map
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  >>> New Best Model Saved! (mAP: {val_map:.8f})")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Early Stopping
            if early_stopping_counter >= TrainConfig.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(
            f"Fold {self.fold_idx} finished. Best mAP: {self.best_map:.8f} at epoch {self.best_epoch+1}"
        )
        return self.checkpoint_path


def train_fold(fold_idx, debug=False):
    """Helper function to train a specific fold."""
    trainer = ModelTrainer(fold_idx, debug=debug)
    return trainer.run()


def generate_submission(model_paths, output_path=None, debug=False):
    """
    Generates submission file using an ensemble of models and TTA.
    """
    if output_path is None:
        output_path = os.path.join(PathConfig.SUBMISSION_DIR, "submission.csv")

    device = torch.device(GeneralConfig.DEVICE)
    test_loader = get_test_loader(debug=debug)

    # Load all models in the ensemble
    models = []
    print(f"Loading {len(model_paths)} models for ensemble inference...")
    for path in model_paths:
        model = SaltUNetPlusPlus()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    # Crop parameters (128 -> 101)
    start_idx = (DataConfig.IMG_H - DataConfig.ORIG_H) // 2
    end_idx = start_idx + DataConfig.ORIG_H

    # Threshold for binarization
    threshold = 0.5
    predictions = {}

    print("Generating predictions...")
    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)

            # Accumulate probabilities
            avg_preds = torch.zeros(
                (images.size(0), 1, DataConfig.IMG_H, DataConfig.IMG_W), device=device
            )

            for model in models:
                # 1. Standard Inference
                out = model(images)
                pred = torch.sigmoid(out)
                avg_preds += pred

                # 2. Test-Time Augmentation (Horizontal Flip)
                images_flipped = torch.flip(images, dims=[3])
                out_flipped = model(images_flipped)
                pred_flipped = torch.sigmoid(out_flipped)
                # Flip back
                pred_flipped_back = torch.flip(pred_flipped, dims=[3])
                avg_preds += pred_flipped_back

            # Average over (Num_Models * 2 views)
            avg_preds /= len(models) * 2

            # Crop to original size
            avg_preds = avg_preds[:, :, start_idx:end_idx, start_idx:end_idx]

            # Binarize
            binary_preds = (avg_preds > threshold).byte().cpu().numpy()

            # Encode RLE
            for i, img_id in enumerate(ids):
                mask = binary_preds[i, 0]  # (H, W)
                rle = rle_encode(mask)
                predictions[img_id] = rle

    # Create Submission DataFrame
    sub_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["rle_mask"])
    sub_df.index.name = "id"
    sub_df.reset_index(inplace=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return sub_df
