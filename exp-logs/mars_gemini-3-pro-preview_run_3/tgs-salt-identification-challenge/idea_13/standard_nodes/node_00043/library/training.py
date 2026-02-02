import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import calculate_iou_map, rle_encode
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.model import SaltUNetPlusPlus


class Trainer:
    """
    Trainer class for the Salt Segmentation Task.
    Encapsulates training logic, mixed precision, loss scheduling, and validation.
    """

    def __init__(self, device=Config.DEVICE, checkpoint_path=None, resume=False):
        self.device = device
        self.model = SaltUNetPlusPlus().to(device)

        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Optimizer
        lr = Config.LEARNING_RATE
        if resume:
            lr = lr * 0.1  # Reduce LR for fine-tuning

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Reduce LR when mAP plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        # Losses
        self.criterion_bce_dice = BCEDiceLoss()
        self.criterion_lovasz = LovaszHingeLoss(per_image=True)

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        self.best_score = -float("inf")
        self.best_model_path = None

    def train_epoch(self, loader, criterion):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, masks, ids) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast():
                # Model returns a list of outputs [out1, out2, out3, out4] due to deep supervision
                outputs = self.model(images)

                loss = 0
                if isinstance(outputs, list):
                    # Sum loss over all deep supervision heads
                    for output in outputs:
                        loss += criterion(output, masks)
                else:
                    loss = criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        preds = []
        gts = []

        with torch.no_grad():
            for images, masks, ids in loader:
                images = images.to(self.device, non_blocking=True)

                # Inference: Model returns single tensor (highest fidelity head) in eval mode
                output = self.model(images)

                # Resize logits back to original 101x101 resolution
                output = F.interpolate(
                    output,
                    size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                    mode="bilinear",
                    align_corners=True,
                )

                # Sigmoid and Threshold
                prob = torch.sigmoid(output)
                pred_mask = (prob > 0.5).float()

                # Resize Ground Truth to 101x101 for accurate metric calculation
                # Masks from loader are 128x128 (transformed)
                if masks.shape[-2:] != (Config.ORIG_HEIGHT, Config.ORIG_WIDTH):
                    masks_resized = F.interpolate(
                        masks.float(),
                        size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                        mode="nearest",
                    )
                else:
                    masks_resized = masks

                # Move to CPU and flatten batch
                pred_numpy = pred_mask.cpu().numpy()
                gt_numpy = masks_resized.cpu().numpy()

                for p, g in zip(pred_numpy, gt_numpy):
                    preds.append(p.squeeze())
                    gts.append(g.squeeze())

        # Calculate mAP
        score = calculate_iou_map(preds, gts)
        return score

    def fit(self, train_loader, val_loader, fold_idx=0, epochs=Config.EPOCHS):
        print(f"Starting training for Fold {fold_idx} (Epochs: {epochs})...")

        for epoch in range(1, epochs + 1):
            # Loss Schedule
            if epoch <= Config.LOVASZ_EPOCH:
                criterion = self.criterion_bce_dice
                loss_name = "BCE+Dice"
            else:
                criterion = self.criterion_lovasz
                loss_name = "Lovasz"

                # Enforce conservative LR at the switch point
                if epoch == Config.LOVASZ_EPOCH + 1:
                    print(
                        "Switching to Lovasz Loss. Enforcing conservative LR limit (1e-4)."
                    )
                    for param_group in self.optimizer.param_groups:
                        if param_group["lr"] > 1e-4:
                            param_group["lr"] = 1e-4

            # Train
            train_loss = self.train_epoch(train_loader, criterion)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_score)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging (Full precision as requested)
            print(
                f"Epoch {epoch} | Loss ({loss_name}): {train_loss} | Val mAP: {val_score} | LR: {current_lr}"
            )

            # Save Checkpoint
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model_path = os.path.join(
                    Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth"
                )
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Saved best model to {self.best_model_path}")

        return self.best_score


def train_fold(
    train_loader,
    val_loader,
    fold_idx=0,
    epochs=Config.EPOCHS,
    checkpoint_path=None,
    resume=False,
):
    """
    Helper function to instantiate a trainer and run the training loop for a single fold.
    """
    trainer = Trainer(checkpoint_path=checkpoint_path, resume=resume)
    best_score = trainer.fit(train_loader, val_loader, fold_idx=fold_idx, epochs=epochs)

    # Clean up to free memory
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    return best_score


def ensemble_submission(test_loader, output_path=Config.SUBMISSION_FILE):
    """
    Generates submission by ensembling (averaging) predictions from all available fold checkpoints.
    """
    device = Config.DEVICE
    models = []

    # Load all available checkpoints
    for fold in range(Config.NUM_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}_best.pth")
        if os.path.exists(ckpt_path):
            model = SaltUNetPlusPlus().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            models.append(model)
            print(f"Loaded checkpoint: {ckpt_path}")

    if not models:
        print("No checkpoints found. Cannot generate submission.")
        return

    results = []
    print(f"Generating submission with ensemble of {len(models)} models...")

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)

            # Accumulate probabilities
            avg_prob = torch.zeros(
                (images.size(0), 1, Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
            ).to(device)

            for model in models:
                # Forward
                output = model(images)

                # Resize to original dimensions
                output = F.interpolate(
                    output,
                    size=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH),
                    mode="bilinear",
                    align_corners=True,
                )

                # Add probabilities
                avg_prob += torch.sigmoid(output)

            # Average
            avg_prob /= len(models)

            # Move to CPU for processing
            avg_prob = avg_prob.cpu().numpy()

            for i in range(len(ids)):
                img_id = ids[i]
                prob_map = avg_prob[i].squeeze()

                # Binarize (Global threshold 0.5 as per standard, or optimized if available)
                mask_bin = (prob_map > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask_bin)
                results.append((img_id, rle))

    # Save to CSV
    df = pd.DataFrame(results, columns=["id", "rle_mask"])
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
