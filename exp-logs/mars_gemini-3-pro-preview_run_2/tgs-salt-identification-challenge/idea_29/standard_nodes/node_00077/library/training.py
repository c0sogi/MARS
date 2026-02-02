import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import inspect
from library.config import Config
from library.utils import rle_encode, unpad_image, calc_map
from library.losses import LovaszHingeLoss


class Trainer:
    """
    Handles training and validation logic.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss()

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()

        total_loss = 0.0
        num_batches = len(loader)

        for batch in loader:
            images = batch["image"].to(self.device).float()
            depths = batch["depth"].to(self.device).float()
            masks = batch["mask"].to(self.device).float()  # (B, 1, H, W)

            self.optimizer.zero_grad()

            # Forward: Input is Image + Depth
            logits = self.model(images, depths)

            # Loss: Lovasz + BCE
            # Cite Lesson 00021: Lovasz-Hinge Loss optimizes IoU directly.
            l_lovasz = self.lovasz(logits, masks)
            l_bce = self.bce(logits, masks)
            loss = l_lovasz + l_bce

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Performs threshold optimization to maximize mAP.
        """
        self.model.eval()
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device).float()
                depths = batch["depth"].to(self.device).float()
                masks = batch["mask"].numpy()

                # Forward pass
                logits = self.model(images, depths)

                # Convert to probabilities
                probs = torch.sigmoid(logits).cpu().numpy()

                # Unpad predictions and masks to original size (101x101)
                for i in range(len(probs)):
                    p = probs[i].squeeze(0)
                    m = masks[i].squeeze(0)

                    p_un = unpad_image(p)
                    m_un = unpad_image(m)

                    all_preds.append(p_un)
                    all_masks.append(m_un)

        all_preds = np.array(all_preds)
        all_masks = np.array(all_masks)

        # Threshold Optimization
        # Cite Lesson 00018: Always tune binarization threshold for mAP metrics.
        best_map = 0.0
        best_thresh = 0.5
        thresholds = np.arange(0.3, 0.72, 0.02)

        for t in thresholds:
            binary_preds = (all_preds > t).astype(np.uint8)
            score = calc_map(binary_preds, all_masks)
            if score > best_map:
                best_map = score
                best_thresh = t

        return best_map, best_thresh, all_preds, all_masks


def train_model(model, train_loader, val_loader, device, config, fold_idx=0):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    trainer = Trainer(model, device, optimizer, scheduler)

    best_map = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.CACHE_DIR, f"best_model_fold{fold_idx}.pth")

    # Store best OOF predictions for ensemble thresholding
    best_oof_preds = None
    best_oof_masks = None

    print(f"Starting training Fold {fold_idx}...")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        val_map, val_thresh, val_preds, val_masks = trainer.validate(val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {train_loss:.8f} | Val mAP: {val_map:.10f} | Best Thresh: {val_thresh:.4f}"
        )

        # Checkpointing
        # Cite Lesson 00033: Decouple prediction probability from decision boundaries. Save based on max score at optimal threshold.
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            best_oof_preds = val_preds
            best_oof_masks = val_masks
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model state
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return model, best_oof_preds, best_oof_masks


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()

    # Load best threshold
    thresh_path = os.path.join(Config.CACHE_DIR, "best_threshold.txt")
    threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            try:
                threshold = float(f.read().strip())
            except ValueError:
                threshold = 0.5

    print(f"Generating submission with threshold: {threshold}")

    # Determine model input requirements
    sig = inspect.signature(model.forward)
    needs_depth = "depth" in sig.parameters

    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device).float()
            depths = batch["depth"].to(device).float()
            ids = batch["id"]

            # Forward pass
            if needs_depth:
                logits = model(images, depths)
            else:
                outputs = model(images)
                logits = outputs["logits"]

            probs = torch.sigmoid(logits).cpu().numpy()

            for i, img_id in enumerate(ids):
                # Process each image in batch
                p = probs[i].squeeze(0)  # (128, 128)

                # Unpad to original size (101, 101)
                p_un = unpad_image(p)

                # Threshold and Encode
                mask = (p_un > threshold).astype(np.uint8)
                rle = rle_encode(mask)

                submission_data.append([img_id, rle])

    # Save to CSV
    df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
