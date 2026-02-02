import os
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from library.config import Config
from library.utils import (
    set_seed,
    unpad_image,
    compute_map_score,
    rle_encode,
)
from library.dataset import get_dataloader
from library.model import ResNeXt50UNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss


class Trainer:
    def __init__(self):
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # Data Loaders
        self.train_loader = get_dataloader(
            "train", batch_size=Config.BATCH_SIZE, shuffle=True
        )
        self.val_loader = get_dataloader(
            "val", batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Model
        self.model = ResNeXt50UNetPlusPlus(n_classes=1, deep_supervision=True)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Losses
        self.criterion_bce_dice = BCEDiceLoss()
        self.criterion_lovasz = LovaszHingeLoss()

        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Training State
        self.best_map = 0.0
        self.best_threshold = 0.0  # To store the optimal binarization threshold
        self.start_epoch = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        # Determine Loss Function based on Curriculum
        if epoch < Config.LOVASZ_SWITCH_EPOCH:
            criterion = self.criterion_bce_dice
            loss_name = "BCE+Dice"
        else:
            criterion = self.criterion_lovasz
            loss_name = "Lovasz-Hinge"

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{Config.EPOCHS} [{loss_name}]",
            leave=False,
        )

        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # AMP Forward pass
            with torch.cuda.amp.autocast():
                # Model returns list of tensors if deep_supervision=True and training=True
                outputs = self.model(images)

                # Calculate Loss
                if epoch < Config.LOVASZ_SWITCH_EPOCH:
                    # Deep Supervision: Loss averages over all outputs
                    loss = criterion(outputs, masks)
                else:
                    # Fine-tuning: Apply Lovasz only to the final output
                    # outputs is a list [out1, out2, out3, out4], we want out4
                    final_output = outputs[-1]
                    loss = criterion(final_output, masks)

            # AMP Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (batch_idx + 1)})

        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0

        # Store all predictions and targets for threshold optimization
        all_preds = []
        all_targets = []

        # Use BCE+Dice for validation loss tracking consistency, or Lovasz if in that phase
        # Usually better to stick to one metric for loss tracking, but let's follow the phase
        if epoch < Config.LOVASZ_SWITCH_EPOCH:
            criterion = self.criterion_bce_dice
        else:
            criterion = self.criterion_lovasz

        with torch.no_grad():
            for images, masks, _ in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # AMP Inference
                with torch.cuda.amp.autocast():
                    # Forward pass (Inference returns only final output if training=False)
                    preds = self.model(images)

                    # Test-Time Augmentation (Horizontal Flip)
                    if Config.TTA_FLIP:
                        images_flipped = torch.flip(images, dims=[3])
                        preds_flipped = self.model(images_flipped)
                        preds_flipped = torch.flip(preds_flipped, dims=[3])
                        preds = (preds + preds_flipped) / 2.0

                    loss = criterion(preds, masks)

                running_loss += loss.item()

                # Move to CPU for metric calculation
                preds_np = (
                    preds.float().sigmoid().cpu().numpy()
                )  # Convert logits to probs
                masks_np = masks.cpu().numpy()

                # Unpad to original size (101x101) for accurate metric calculation
                for i in range(len(preds_np)):
                    # preds_np[i] shape is (1, 128, 128) or (128, 128)
                    p = preds_np[i].squeeze()
                    m = masks_np[i].squeeze()

                    p_orig = unpad_image(p, Config.ORIG_SIZE)
                    m_orig = unpad_image(m, Config.ORIG_SIZE)

                    all_preds.append(p_orig)
                    all_targets.append(m_orig)

        # Threshold Optimization
        # Sweep probability thresholds to find best mAP
        thresholds = np.arange(0.3, 0.75, 0.05)
        best_epoch_map = 0.0
        best_epoch_thresh = 0.5

        # Convert lists to arrays for faster processing if memory allows,
        # otherwise loop carefully. 600 images is small enough for numpy.
        all_preds_arr = np.array(all_preds)
        all_targets_arr = np.array(all_targets)

        for t in thresholds:
            # Binarize
            preds_bin = (all_preds_arr > t).astype(np.float32)
            # Compute mAP
            score = compute_map_score(
                torch.tensor(preds_bin), torch.tensor(all_targets_arr)
            )
            if score > best_epoch_map:
                best_epoch_map = score
                best_epoch_thresh = t

        return running_loss / len(self.val_loader), best_epoch_map, best_epoch_thresh

    def fit(self):
        print(f"Starting training on device: {self.device}")
        early_stopping_counter = 0

        for epoch in range(self.start_epoch, Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_map, val_thresh = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step(val_map)

            duration = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val mAP: {val_map:.10f} (Thresh: {val_thresh:.2f}) | "
                f"Time: {duration:.0f}s"
            )

            # Checkpointing
            if val_map > self.best_map:
                print(
                    f"mAP Improved ({self.best_map:.5f} -> {val_map:.5f}). Saving model..."
                )
                self.best_map = float(val_map)
                self.best_threshold = float(val_thresh)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_map": self.best_map,
                        "best_threshold": self.best_threshold,
                    },
                    Config.BEST_MODEL_PATH,
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Early Stopping
            if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {Config.EARLY_STOPPING_PATIENCE} epochs of no improvement."
                )
                break

    def generate_submission(self):
        print("Generating submission...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print("No checkpoint found. Skipping submission generation.")
            return

        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Use the optimized threshold from the best epoch
        optimal_threshold = checkpoint.get("best_threshold", 0.5)
        print(f"Using optimal threshold: {optimal_threshold}")

        test_loader = get_dataloader(
            "test", batch_size=Config.BATCH_SIZE, shuffle=False
        )

        submission_data = []

        with torch.no_grad():
            for images, ids in tqdm(test_loader, desc="Inference"):
                images = images.to(self.device)

                # Forward
                preds = self.model(images)

                # TTA
                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, dims=[3])
                    preds_flipped = self.model(images_flipped)
                    preds_flipped = torch.flip(preds_flipped, dims=[3])
                    preds = (preds + preds_flipped) / 2.0

                preds_prob = preds.sigmoid().cpu().numpy()

                for i in range(len(ids)):
                    # Unpad
                    pred_img = preds_prob[i].squeeze()
                    pred_orig = unpad_image(pred_img, Config.ORIG_SIZE)

                    # Binarize
                    mask_bin = (pred_orig > optimal_threshold).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(mask_bin)
                    submission_data.append([ids[i], rle])

        # Save
        df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
